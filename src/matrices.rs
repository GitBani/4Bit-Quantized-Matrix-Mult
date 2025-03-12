use rand::Rng;
use std::cmp::Ordering;
use std::ops::Range;
use std::{arch::x86_64::*, mem};

use crate::quantization::{quantize_and_pack, unpack_and_dequantize, Quantizer4Bit};

#[derive(PartialEq, Debug)]
pub struct Matrix<T> {
    pub data: Vec<T>,
    pub rows: usize,
    pub cols: usize,
}

impl<T> Matrix<T>
where
    T: Copy + Default,
{
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![T::default(); rows * cols],
            rows,
            cols,
        }
    }

    pub fn transpose(&self) -> Self {
        let mut transposed = Matrix::new(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }

        transposed
    }
}

impl Matrix<f32> {
    /// Quantize and pack values into a row-major matrix
    pub fn quantize_lhs(&self, quantizer: &impl Quantizer4Bit) -> Matrix<u8> {
        Self::quantize(&self, self.rows, quantizer)
    }

    /// Quantize and pack values into a column-major matrix
    pub fn quantize_rhs(&self, quantizer: &impl Quantizer4Bit) -> Matrix<u8> {
        Self::quantize(&self.transpose(), self.rows, quantizer)
    }

    fn quantize(matrix: &Self, dimension: usize, quantizer: &impl Quantizer4Bit) -> Matrix<u8> {
        let mut data = vec![];
        // for each row/col
        for vec in matrix.data.chunks(dimension) {
            // group in pairs, and if odd pad with 0
            // this ensures that rows/cols boundaries are not in the middle of a byte
            for to_pack in vec.chunks(2) {
                let v1 = to_pack[0];
                let v2 = to_pack.get(1).copied().unwrap_or(0.0);
                data.push(quantize_and_pack(quantizer, v1, v2));
            }
        }

        Matrix {
            data,
            rows: matrix.rows,
            cols: matrix.cols,
        }
    }

    pub fn random_square(dimension: usize, range: Range<f32>) -> Self {
        let size = dimension * dimension;
        let mut data = Vec::<f32>::with_capacity(size);

        let mut rng = rand::rng();

        for _ in 0..size {
            data.push(rng.random_range(range.clone()));
        }

        Matrix {
            data,
            rows: dimension,
            cols: dimension,
        }
    }

    pub fn min_and_max(&self) -> (f32, f32) {
        (
            *self
                .data
                .iter()
                .min_by(|&a, &b| a.partial_cmp(b).unwrap())
                .unwrap(),
            *self
                .data
                .iter()
                .max_by(|&a, &b| a.partial_cmp(b).unwrap())
                .unwrap(),
        )
    }
}

impl Matrix<u8> {
    pub fn dequantize(&self, quantizer: &impl Quantizer4Bit) -> Matrix<f32> {
        let mut data = vec![];
        let rem = self.cols % 2;

        for row in self.data.chunks(self.cols) {
            for i in 0..self.cols - rem {
                let (f1, f2) = unpack_and_dequantize(quantizer, row[i]);
                data.push(f1);
                data.push(f2);
            }
            if rem == 1 {
                let f = quantizer.dequantize(row[self.cols - 1]);
                data.push(f);
            }
        }

        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub fn random_sparse_quantized_square(dimension: usize) -> Self {
        let mut data = vec![];

        let rand_packed = || {
            let mut rng = rand::rng();
            if rng.random::<f32>() > 0.67 {
                let nibble1 = rng.random_range(0_u8..15_u8);
                let nibble2 = rng.random_range(0_u8..15_u8);
                let packed = (nibble2 << 4) + nibble1;
                packed
            } else {
                0u8
            }
        };
        let rand_single = || {
            let mut rng = rand::rng();
            if rng.random::<f32>() > 0.67 {
                rng.random_range(0_u8..15_u8)
            } else {
                0u8
            }
        };

        let d_bytes = (dimension + 1) / 2;
        let rem = dimension % 2;

        // for each row
        for _ in 0..dimension {
            for _ in 0..(d_bytes - rem) {
                data.push(rand_packed());
            }

            // last in odd row is padded with a 0
            if rem == 1 {
                data.push(rand_single());
            }
        }

        Matrix {
            data,
            rows: dimension,
            cols: dimension,
        }
    }

    /// Multiply 4-bit quantized matrices using i32 accumulators, no output pipeline (returns matrix with accumulators directly)
    /// self in row-major, other in column-major (this is processed by quantize_lhs and quantize_rhs)
    ///
    /// Optimization from gemmlowp:
    ///
    /// Let `P` denote the matrix shaped like `lhs`, but filled with 1's.

    /// Let `Q` denote the matrix shaped like `rhs`, but filled with 1's.

    /// Adding lhs_offset to each entry of `lhs`, means adding `lhs_offset * P` to
    /// `lhs`.

    /// Adding rhs_offset to each entry of `rhs`, means adding `rhs_offset * Q` to
    /// `rhs`.

    /// Thus, as far as handling `lhs_offset` and `rhs_offset` goes, the matrix product
    /// to be computed is:

    /// (lhs + lhs_offset * P) * (rhs + rhs_offset * Q)

    /// Expanding this (using distributivity of matrix multiplication over addition), we
    /// see that the above product is equal to the following sum of 4 terms:

    /// lhs * rhs                                 (2)
    /// + lhs_offset * P * rhs
    /// + lhs * rhs_offset * Q
    /// + lhs_offset * rhs_offset * P * Q

    /// The first term, `lhs * rhs`, is just the matrix multiplication ignoring the
    /// offsets, i.e. as if `lhs_offset==rhs_offset==0`. Our claim here is that this is
    /// all what we have to compute in the GEMM kernel.

    /// In the second term, `lhs_offset * P * rhs`, notice that since P is filled with
    /// 1's, `P * rhs` has all its rows equal to each other, and equal to the row-vector
    /// of sums of all the entries in each column of rhs.

    /// Thus, we can compute the second term, `lhs_offset * P * rhs`, by summing each
    /// column of rhs. This produces a single row-vector, and in order to add the second
    /// term, we simply need to add this row-vector (multiplied by lhs_offset) to each
    /// row of the result. This is just a rank one update of the result (equivalently,
    /// the second term is a rank one matrix), and we can efficiently store it as a
    /// single vector.

    /// The third term, `lhs * rhs_offset * Q`, is entirely similar to the second one,
    /// and can be similarly computed by summing each row of lhs, storing this in a
    /// single column-vector, and later multiplying these sums by rhs_offset.

    /// The fourth term is a single constant, repeated into all the entries of the
    /// matrix. The matrix `P * Q` is filled with the single constant value 'depth' (the
    /// depth of the matrix product i.e. the number of columns of the lhs). Thus the
    /// fourth term is simply the rank zero update adding this constant to each matrix
    /// entry:

    /// lhs_offset * rhs_offset * depth
    pub fn qmultiply(
        &self,
        other: &Self,
        lhs_offset: i32,
        rhs_offset: i32,
        result_offset: i32,
        q_multiplier: i32,
        rshift: i32,
    ) -> Matrix<u8> {
        let mut accumulators = Vec::<i32>::with_capacity(self.rows * other.cols);

        // stores extracted nibbles
        let mut lhs_row = Vec::<u8>::with_capacity(self.cols);
        unsafe { lhs_row.set_len(self.cols) };
        let mut rhs_col = Vec::<u8>::with_capacity(other.rows);
        unsafe { rhs_col.set_len(other.rows) };

        // to determine which nibble of byte to get
        let mut lower_bits_lhs = true;

        // store vectors that are a part of the optimization trick for adding offsets described above
        let mut rhs_offset_vec = Vec::with_capacity(self.cols);
        let mut lhs_offset_vec = Vec::with_capacity(other.rows);

        // only calculate lhs_offset_vec on first iteration
        let mut first = true;

        let mut i = 0;
        for _ in 0..self.rows {
            // Get next row from nibbles
            for row_idx in 0..self.cols {
                let next_val;
                if lower_bits_lhs {
                    next_val = self.data[i] & 0x0F
                } else {
                    next_val = self.data[i] >> 4;
                    i += 1;
                }
                lower_bits_lhs = !lower_bits_lhs;
                lhs_row[row_idx] = next_val;
            }

            rhs_offset_vec.push(lhs_row.iter().map(|&x| x as i32).sum::<i32>() * rhs_offset);

            let mut j = 0;
            let mut lower_bits_rhs = true;
            for _ in 0..other.cols {
                // Get next col from nibbles
                for col_idx in 0..other.rows {
                    let next_val;
                    if lower_bits_rhs {
                        next_val = other.data[j] & 0x0F
                    } else {
                        next_val = other.data[j] >> 4;
                        j += 1;
                    }
                    lower_bits_rhs = !lower_bits_rhs;
                    rhs_col[col_idx] = next_val;
                }

                if first {
                    lhs_offset_vec
                        .push(rhs_col.iter().map(|&x| x as i32).sum::<i32>() * lhs_offset);
                }

                let mut accumulator: i32 = 0;
                for k in 0..self.cols {
                    accumulator += lhs_row[k] as i32 * rhs_col[k] as i32;
                }
                accumulators.push(accumulator);
            }

            first = false;
        }

        // add offsets and multiplier, repack the 4-bit values in the resulting matrix
        let mut result = Vec::with_capacity(accumulators.len());
        let odd_cols = (other.cols & 1) == 1;

        let depth = self.cols as i32;
        for i in 0..self.rows {
            let row_start = i * other.cols;
            // handle pairs to pack
            for j in (0..other.cols - 1).step_by(2) {
                let with_offset_lo = accumulators[row_start + j]
                    + lhs_offset_vec[j]
                    + rhs_offset_vec[i]
                    + lhs_offset * rhs_offset * depth;
                let with_multiplier_lo = result_offset
                    + rounding_rshift(fixed_point_multiply(with_offset_lo, q_multiplier), rshift);

                let with_offset_hi = accumulators[row_start + j + 1]
                    + lhs_offset_vec[j + 1]
                    + rhs_offset_vec[i]
                    + lhs_offset * rhs_offset * depth;
                let with_multiplier_hi = result_offset
                    + rounding_rshift(fixed_point_multiply(with_offset_hi, q_multiplier), rshift);

                let packed = ((with_multiplier_hi.clamp(0, 15) as u8) << 4)
                    + with_multiplier_lo.clamp(0, 15) as u8;
                result.push(packed);
            }
            // handle remainder element
            if odd_cols {
                let with_offset = accumulators[row_start + other.cols - 1]
                    + lhs_offset_vec[other.cols - 1]
                    + rhs_offset_vec[i]
                    + lhs_offset * rhs_offset * depth;
                let with_multiplier = result_offset
                    + rounding_rshift(fixed_point_multiply(with_offset, q_multiplier), rshift);
                result.push(with_multiplier.clamp(0, 15) as u8);
            }
        }

        Matrix {
            data: result,
            rows: self.rows,
            cols: other.cols,
        }
    }

    pub unsafe fn qmultiply_simd(
        &self,
        other: &Self,
        lhs_offset: i32,
        rhs_offset: i32,
        result_offset: i32,
        q_multiplier: i32,
        rshift: i32,
    ) -> Matrix<u8> {
        let mut accumulators = vec![0; self.rows * other.cols];
        let mut result = Vec::with_capacity(accumulators.len());

        let depth = self.cols;
        let blocks = depth / 8;
        let blocked_depth = blocks * 8;
        let extra = depth - blocked_depth;
        let blocked_depth_bytes = blocks * 4;
        let depth_bytes = (depth + 1) / 2;

        let mut lhs_row_regs = vec![mem::zeroed(); blocks];
        let mut rhs_col_regs = vec![mem::zeroed(); blocks];

        // stores extracted nibbles
        // plus one: if row/col is odd, then the last nibble will be in lower nibble, meaning higher nibble (0) will be stored here too
        // the kernel skips this +1 element
        let mut lhs_row = vec![0; extra + 1];
        let mut rhs_col = vec![0; extra + 1];

        // store vectors that are a part of the optimization trick for adding offsets described above
        let mut rhs_offset_vec = vec![0; self.cols];
        let mut lhs_offset_vec = vec![0; other.rows];

        // only calculate lhs_offset_vec on first iteration
        let mut first = true;

        for (row, i) in (0..self.data.len()).step_by(depth_bytes).enumerate() {
            extract_nibbles(
                &self.data[i..i + depth_bytes],
                blocked_depth_bytes,
                depth_bytes,
                &mut lhs_row_regs,
                &mut lhs_row,
            );

            rhs_offset_vec[row] = (lhs_row_regs.iter().map(|&v| hsum_avx2(v)).sum::<i32>()
                + lhs_row.iter().sum::<i32>())
                * rhs_offset;

            for (col, j) in (0..other.data.len()).step_by(depth_bytes).enumerate() {
                extract_nibbles(
                    &other.data[j..j + depth_bytes],
                    blocked_depth_bytes,
                    depth_bytes,
                    &mut rhs_col_regs,
                    &mut rhs_col,
                );
                if first {
                    lhs_offset_vec[col] = (rhs_col_regs.iter().map(|&v| hsum_avx2(v)).sum::<i32>()
                        + rhs_col.iter().sum::<i32>())
                        * lhs_offset;
                }

                let acc_idx = row * self.cols + col;
                let mut vsum = _mm256_setzero_si256();
                for k in 0..blocks {
                    let vlhs = lhs_row_regs[k];
                    let vrhs = rhs_col_regs[k];
                    let vprod = _mm256_mullo_epi32(vlhs, vrhs);
                    vsum = _mm256_add_epi32(vsum, vprod);
                }
                accumulators[acc_idx] += hsum_avx2(vsum);

                // handle remainder sequentially
                for k in 0..extra {
                    accumulators[acc_idx] += lhs_row[k] * rhs_col[k];
                }
            }

            first = false;
        }

        // add offsets and multiplier, repack the 4-bit values in the resulting matrix
        let depth = self.cols as i32;
        let odd_cols = (other.cols & 1) == 1;

        for i in 0..self.rows {
            let row_start = i * other.cols;
            // handle pairs to pack
            for j in (0..other.cols - 1).step_by(2) {
                let with_offset_lo = accumulators[row_start + j]
                    + lhs_offset_vec[j]
                    + rhs_offset_vec[i]
                    + lhs_offset * rhs_offset * depth;
                let with_multiplier_lo = result_offset
                    + rounding_rshift(fixed_point_multiply(with_offset_lo, q_multiplier), rshift);

                let with_offset_hi = accumulators[row_start + j + 1]
                    + lhs_offset_vec[j + 1]
                    + rhs_offset_vec[i]
                    + lhs_offset * rhs_offset * depth;
                let with_multiplier_hi = result_offset
                    + rounding_rshift(fixed_point_multiply(with_offset_hi, q_multiplier), rshift);

                let packed = ((with_multiplier_hi.clamp(0, 15) as u8) << 4)
                    + with_multiplier_lo.clamp(0, 15) as u8;
                result.push(packed);
            }
            // handle remainder element
            if odd_cols {
                let with_offset = accumulators[row_start + other.cols - 1]
                    + lhs_offset_vec[other.cols - 1]
                    + rhs_offset_vec[i]
                    + lhs_offset * rhs_offset * depth;
                let with_multiplier = result_offset
                    + rounding_rshift(fixed_point_multiply(with_offset, q_multiplier), rshift);
                result.push(with_multiplier.clamp(0, 15) as u8);
            }
        }

        Matrix {
            data: result,
            rows: self.rows,
            cols: other.cols,
        }
    }

    pub unsafe fn qmultiply_simd_fp(
        &self,
        other: &Self,
        lhs_offset: i32,
        rhs_offset: i32,
        result_offset: i32,
        q_multiplier: i32,
        rshift: i32,
    ) -> Matrix<u8> {
        let depth = self.cols;
        let blocks = depth / 8;
        let blocked_depth = blocks * 8;
        let extra = depth - blocked_depth;
        let blocked_depth_bytes = blocks * 4;
        let depth_bytes = (depth + 1) / 2;

        let lhs_rows = self.data.chunks(depth_bytes).map(|row_bytes| {
            extract_nibbles_into_regs(row_bytes, blocked_depth_bytes, depth_bytes)
        });
        let rhs_cols = other.data.chunks(depth_bytes).map(|col_bytes| {
            extract_nibbles_into_regs(col_bytes, blocked_depth_bytes, depth_bytes)
        });

        dbg!(&self);
        for x in lhs_rows {
            for reg in x.0 {
                let mut v = [0; 8];
                _mm256_storeu_si256(v.as_mut_ptr() as *mut __m256i, reg);
                println!("{v:?}");
            }
        }

        // dbg!(depth, blocked_depth, blocked_depth_bytes, depth_bytes);
        // dbg!(lhs_rows.len());
        // dbg!(rhs_cols.len());

        // let zipped = lhs_rows.cycle().take(lhs_rows.len() * )

        Matrix {
            data: vec![],
            rows: 0,
            cols: 0,
        }
    }
}

// blocked_count must be a multiple of 8
unsafe fn extract_nibbles(
    data: &[u8],
    blocked_count: usize,
    total_count: usize,
    regs_dst: &mut [__m256i],
    extras_dst: &mut [i32],
) {
    // nibbles packed into m256i's
    let mut blocks = data.chunks_exact(4);
    for (i, _) in (0..blocked_count).step_by(4).enumerate() {
        let block = blocks.next().unwrap();
        let bytes = _mm_set_epi32(
            block[0] as i32,
            block[1] as i32,
            block[2] as i32,
            block[3] as i32,
        );
        let lower_nibbles = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
        let upper_nibbles = _mm_srli_epi32(bytes, 4);

        let unpacked_lo = _mm_unpacklo_epi32(lower_nibbles, upper_nibbles);
        let unpacked_hi = _mm_unpackhi_epi32(lower_nibbles, upper_nibbles);

        let mut appended = _mm256_castsi128_si256(unpacked_lo);
        appended = _mm256_insertf128_si256(appended, unpacked_hi, 1);

        regs_dst[i] = appended;
    }
    // extras go in i32 vector
    for (i, data_idx) in (blocked_count..total_count).enumerate() {
        let byte = data[data_idx];
        let lo = byte & 0x0F;
        let hi = byte >> 4;
        extras_dst[i] = lo as i32;
        extras_dst[i + 1] = hi as i32;
    }
}

// blocked_count must be a multiple of 8
unsafe fn extract_nibbles_into_regs(
    data: &[u8],
    blocked_count: usize,
    total_count: usize,
) -> (
    impl Iterator<Item = __m256i> + '_,
    impl Iterator<Item = i32> + '_,
) {
    // nibbles packed into m256i's
    let mut blocks = data.chunks_exact(4);
    let regs = (0..blocked_count).step_by(4).map(move |_| {
        let block = blocks.next().unwrap();
        let bytes = _mm_set_epi32(
            block[0] as i32,
            block[1] as i32,
            block[2] as i32,
            block[3] as i32,
        );
        let lower_nibbles = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
        let upper_nibbles = _mm_srli_epi32(bytes, 4);

        let unpacked_lo = _mm_unpacklo_epi32(lower_nibbles, upper_nibbles);
        let unpacked_hi = _mm_unpackhi_epi32(lower_nibbles, upper_nibbles);

        let appended = _mm256_castsi128_si256(unpacked_lo);
        _mm256_insertf128_si256(appended, unpacked_hi, 1)
    });
    let extras = (blocked_count..total_count).flat_map(|data_idx| {
        let byte = data[data_idx];
        let lo = byte & 0x0F;
        let hi = byte >> 4;
        [lo as i32, hi as i32]
    });

    (regs, extras)
}

#[inline]
unsafe fn hsum_avx2(v: __m256i) -> i32 {
    let sum = _mm256_hadd_epi32(v, v);
    let sum = _mm256_hadd_epi32(sum, sum);
    _mm256_extract_epi32(sum, 0) + _mm256_extract_epi32(sum, 4)
}

fn rounding_rshift(x: i32, rshift: i32) -> i32 {
    if rshift == 0 {
        return x;
    };

    let rounding_offset = 1 << (rshift - 1);
    (x + rounding_offset) >> rshift
}

fn fixed_point_multiply(a: i32, b: i32) -> i32 {
    let temp = a as i64 * b as i64 + (1_i64 << 30);
    (temp >> 31) as i32
}

#[derive(PartialEq, Debug)]
pub struct CsrMatrix<T> {
    pub data: Vec<T>,
    pub col_index: Vec<usize>,
    pub row_index: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

impl CsrMatrix<u8> {
    pub fn new(rows: usize, cols: usize) -> Self {
        CsrMatrix {
            data: vec![],
            col_index: vec![],
            row_index: vec![],
            rows,
            cols,
        }
    }
    // Convert *quantized* matrix into CSR representation
    pub fn from(m: &Matrix<u8>) -> Self {
        let mut data = vec![];
        let mut col_index = vec![];
        let mut row_index = vec![0];

        let mut r_idx = 0;
        for row in m.data.chunks(m.cols) {
            for (c_idx, &element) in row.iter().enumerate() {
                if element == 0 {
                    continue;
                }

                data.push(element);
                col_index.push(c_idx);
                r_idx += 1;
            }
            row_index.push(r_idx);
        }

        if data.len() == 0 {
            row_index = vec![];
        }

        CsrMatrix {
            data,
            col_index,
            row_index,
            rows: m.rows,
            cols: m.cols,
        }
    }

    pub fn from_packed_quantized(qm: &Matrix<u8>) -> Self {
        // quantized matrices are packed
        let qrowsize = (qm.cols + 1) / 2;

        let mut data = vec![];
        let mut col_index = vec![];
        let mut row_index = vec![0];

        let mut r_idx = 0;
        for row in qm.data.chunks(qrowsize) {
            for (c_idx, &element) in row.iter().enumerate() {
                if element == 0 {
                    continue;
                }

                data.push(element);
                col_index.push(c_idx * 2); // becuase elements are packed
                r_idx += 1;
            }
            row_index.push(r_idx);
        }

        if data.len() == 0 {
            row_index = vec![];
        }

        CsrMatrix {
            data,
            col_index,
            row_index,
            rows: qm.rows,
            cols: qm.cols,
        }
    }

    pub fn csr_qmultiply(
        &self,
        other: &CscMatrix<u8>,
        lhs_offset: i32,
        rhs_offset: i32,
        result_offset: i32,
        q_multiplier: i32,
        rshift: i32,
    ) -> CsrMatrix<u8> {
        let mut accumulators = vec![];
        let mut result_col_index = vec![];
        let mut result_row_index = Vec::<usize>::with_capacity(self.rows + 1);
        result_row_index.push(0);

        let mut rhs_offset_vec = vec![0; self.cols];
        let mut lhs_offset_vec = vec![0; other.rows];

        let row_len = self.row_index.len();
        let col_len = other.col_index.len();

        if row_len == 0 || col_len == 0 {
            return CsrMatrix::new(self.rows, other.cols);
        }

        for (row_num, &row_ptr) in self.row_index[..row_len - 1].iter().enumerate() {
            let row_start = row_ptr;
            let row_end = self.row_index[row_num + 1];

            rhs_offset_vec[row_num] = self.data[row_start..row_end]
                .iter()
                .map(|&x| x as i32)
                .sum::<i32>()
                * rhs_offset;

            for (col_num, &col_ptr) in other.col_index[..col_len - 1].iter().enumerate() {
                let col_start = col_ptr;
                let col_end = other.col_index[col_num + 1];

                lhs_offset_vec[col_num] = other.data[col_start..col_end]
                    .iter()
                    .map(|&x| x as i32)
                    .sum::<i32>()
                    * lhs_offset;

                let mut lhs_ptr = row_start;
                let mut rhs_ptr = col_start;

                let mut sum = 0;
                while lhs_ptr < row_end && rhs_ptr < col_end {
                    match self.col_index[lhs_ptr].cmp(&other.row_index[rhs_ptr]) {
                        Ordering::Equal => {
                            let lhs_byte = self.data[lhs_ptr];
                            let (lhs_lo, lhs_hi) = (lhs_byte & 0x0F, lhs_byte >> 4);

                            let rhs_byte = other.data[rhs_ptr];
                            let (rhs_lo, rhs_hi) = (rhs_byte & 0x0F, rhs_byte >> 4);

                            sum += lhs_lo as i32 * rhs_lo as i32 + lhs_hi as i32 * rhs_hi as i32;
                            lhs_ptr += 1;
                            rhs_ptr += 1;
                        }
                        Ordering::Greater => rhs_ptr += 1,
                        Ordering::Less => lhs_ptr += 1,
                    }
                }

                accumulators.push(sum);
                result_col_index.push(col_num);
            }

            result_row_index.push(accumulators.len());
        }

        // add offsets and multiplier
        let mut result = vec![];
        let depth = self.cols as i32;
        let res_row_len = result_row_index.len();

        for (i, &row_start) in result_row_index[..res_row_len - 1].iter().enumerate() {
            let row_end = result_row_index[i + 1];

            for (j, _) in (row_start..row_end).enumerate() {
                let with_offset = accumulators[row_start + j]
                    + lhs_offset_vec[result_col_index[row_start + j]]
                    + rhs_offset_vec[i]
                    + lhs_offset * rhs_offset * depth;
                let with_multiplier = result_offset
                    + rounding_rshift(fixed_point_multiply(with_offset, q_multiplier), rshift);
                let casted = with_multiplier.clamp(0, 15) as u8;

                if casted != 0 {
                    result.push(casted);
                }
            }
        }

        CsrMatrix {
            data: result,
            col_index: result_col_index,
            row_index: result_row_index,
            rows: self.rows,
            cols: other.cols,
        }
    }
}

#[derive(PartialEq, Debug)]
pub struct CscMatrix<T> {
    pub data: Vec<T>,
    pub row_index: Vec<usize>,
    pub col_index: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

impl CscMatrix<u8> {
    // Convert a *quantized* matrix in *column-major order* to CSC representation
    pub fn from(m: &Matrix<u8>) -> Self {
        let csr_of_col_major = CsrMatrix::from(&m);
        CscMatrix {
            data: csr_of_col_major.data,
            // flip row and col index vectors (function expects row-major and generates CSR, this is col-major thus ends up in CSC)
            row_index: csr_of_col_major.col_index,
            col_index: csr_of_col_major.row_index,
            rows: csr_of_col_major.rows,
            cols: csr_of_col_major.cols,
        }
    }

    pub fn from_packed_quantized(qm: &Matrix<u8>) -> Self {
        let csr_of_col_major = CsrMatrix::from_packed_quantized(&qm);
        CscMatrix {
            data: csr_of_col_major.data,
            // flip row and col index vectors (function expects row-major and generates CSR, this is col-major thus ends up in CSC)
            row_index: csr_of_col_major.col_index,
            col_index: csr_of_col_major.row_index,
            rows: csr_of_col_major.rows,
            cols: csr_of_col_major.cols,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::quantization::{quantize_multiplier, AffineQuantizer};

    use super::*;

    #[test]
    fn quantize_lhs_odd_number_entries() {
        let lhs: Matrix<f32> = Matrix {
            data: vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
            ],
            rows: 3,
            cols: 5,
        };
        let quantizer = AffineQuantizer::new(1.0, 15.0);

        let expected = Matrix {
            data: vec![
                quantize_and_pack(&quantizer, 1., 2.),
                quantize_and_pack(&quantizer, 3., 0.),
                quantize_and_pack(&quantizer, 4., 5.),
                quantize_and_pack(&quantizer, 6., 0.),
                quantize_and_pack(&quantizer, 7., 8.),
                quantize_and_pack(&quantizer, 9., 0.),
                quantize_and_pack(&quantizer, 10., 11.),
                quantize_and_pack(&quantizer, 12., 0.),
                quantize_and_pack(&quantizer, 13., 14.),
                quantize_and_pack(&quantizer, 15., 0.),
            ],
            rows: 3,
            cols: 5,
        };

        assert_eq!(expected, lhs.quantize_lhs(&quantizer));
    }

    #[test]
    fn quantize_lhs_even_number_entries() {
        let lhs: Matrix<f32> = Matrix {
            data: vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            rows: 4,
            cols: 4,
        };
        let quantizer = AffineQuantizer::new(1.0, 16.0);

        let expected = Matrix {
            data: vec![
                quantize_and_pack(&quantizer, 1., 2.),
                quantize_and_pack(&quantizer, 3., 4.),
                quantize_and_pack(&quantizer, 5., 6.),
                quantize_and_pack(&quantizer, 7., 8.),
                quantize_and_pack(&quantizer, 9., 10.),
                quantize_and_pack(&quantizer, 11., 12.),
                quantize_and_pack(&quantizer, 13., 14.),
                quantize_and_pack(&quantizer, 15., 16.),
            ],
            rows: 4,
            cols: 4,
        };

        assert_eq!(expected, lhs.quantize_lhs(&quantizer));
    }

    #[test]
    fn quantize_rhs_odd_number_entries() {
        let rhs: Matrix<f32> = Matrix {
            data: vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
            ],
            rows: 3,
            cols: 5,
        };
        let quantizer = AffineQuantizer::new(1.0, 15.0);

        let expected = Matrix {
            data: vec![
                quantize_and_pack(&quantizer, 1., 6.),
                quantize_and_pack(&quantizer, 11., 0.),
                quantize_and_pack(&quantizer, 2., 7.),
                quantize_and_pack(&quantizer, 12., 0.),
                quantize_and_pack(&quantizer, 3., 8.),
                quantize_and_pack(&quantizer, 13., 0.),
                quantize_and_pack(&quantizer, 4., 9.),
                quantize_and_pack(&quantizer, 14., 0.),
                quantize_and_pack(&quantizer, 5., 10.),
                quantize_and_pack(&quantizer, 15., 0.),
            ],
            rows: 5,
            cols: 3,
        };

        assert_eq!(expected, rhs.quantize_rhs(&quantizer));
    }

    #[test]
    fn quantize_rhs_even_number_entries() {
        let rhs: Matrix<f32> = Matrix {
            data: vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            rows: 4,
            cols: 4,
        };
        let quantizer = AffineQuantizer::new(1.0, 16.0);

        let expected = Matrix {
            data: vec![
                quantize_and_pack(&quantizer, 1., 5.),
                quantize_and_pack(&quantizer, 9., 13.),
                quantize_and_pack(&quantizer, 2., 6.),
                quantize_and_pack(&quantizer, 10., 14.),
                quantize_and_pack(&quantizer, 3., 7.),
                quantize_and_pack(&quantizer, 11., 15.),
                quantize_and_pack(&quantizer, 4., 8.),
                quantize_and_pack(&quantizer, 12., 16.),
            ],
            rows: 4,
            cols: 4,
        };
        assert_eq!(expected, rhs.quantize_rhs(&quantizer));
    }

    #[test]
    fn test() {
        let rhs = Matrix::random_sparse_quantized_square(9);
        let lhs = Matrix::random_sparse_quantized_square(9);
        dbg!(lhs.data.len());
        _ = unsafe { lhs.qmultiply_simd_fp(&rhs, 0, 0, 0, 0, 0) };
    }
    // todo make this actual tests
    #[test]
    fn test_csr() {
        let d = 3;
        let lhs = Matrix::random_sparse_quantized_square(d);
        let rhs = Matrix::random_sparse_quantized_square(d);
        dbg!(&lhs, &rhs);

        let lhs_quantizer = AffineQuantizer::new(-5.0, 5.0);
        let rhs_quantizer = AffineQuantizer::new(-5.0, 5.0);
        let result_quantizer = AffineQuantizer::new(-5.0, 5.0);

        let lhs_offset = -(lhs_quantizer.zero as i32);
        let rhs_offset = -(rhs_quantizer.zero as i32);
        let result_offset = result_quantizer.zero as i32;

        let real_multiplier = lhs_quantizer.scale * rhs_quantizer.scale / result_quantizer.scale;
        let (q_multiplier, rshift) = quantize_multiplier(real_multiplier);

        let res = lhs.qmultiply(
            &rhs,
            lhs_offset,
            rhs_offset,
            result_offset,
            q_multiplier,
            rshift,
        );
        dbg!(&res);
        let expected = CsrMatrix::from(&res);
        dbg!(&expected);

        let csr = CsrMatrix::from_packed_quantized(&lhs);
        let csc = CscMatrix::from_packed_quantized(&rhs);
        dbg!(&csr, &csc);

        let actual = csr.csr_qmultiply(
            &csc,
            lhs_offset,
            rhs_offset,
            result_offset,
            q_multiplier,
            rshift,
        );
        dbg!(&actual);
    }

    #[test]
    fn test_csc() {
        let m = Matrix {
            data: vec![1, 0, 2, 0, 0, 3, 4, 5, 6],
            rows: 3,
            cols: 3,
        };
        let csc = CscMatrix::from_packed_quantized(&m);
        dbg!(&csc);
    }
}
