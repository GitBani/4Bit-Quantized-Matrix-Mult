use std::ops::Range;

use rand::Rng;

use crate::quantization::{quantize_and_pack, Quantizer4Bit};

#[derive(PartialEq, Debug)]
pub struct Matrix<T> {
    pub data: Vec<T>,
    pub rows: usize,
    pub cols: usize,
}

// Todo maybe delete this all and make the multiplication a function on slices
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

    pub fn new_quantized<Q: Copy + Default>(rows: usize, cols: usize) -> Matrix<Q> {
        Matrix {
            data: vec![Q::default(); (rows * cols + 1) / 2],
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
        let mut quantized_lhs = Matrix::<u8>::new_quantized(self.rows, self.cols);
        Self::quantize(&self, &mut quantized_lhs, quantizer);
        quantized_lhs
    }

    /// Quantize and pack values into a column-major matrix
    pub fn quantize_rhs(&self, quantizer: &impl Quantizer4Bit) -> Matrix<u8> {
        let mut quantized_rhs = Matrix::<u8>::new_quantized(self.cols, self.rows);
        Self::quantize(&self.transpose(), &mut quantized_rhs, quantizer);
        quantized_rhs
    }

    fn quantize(matrix: &Self, dst: &mut Matrix<u8>, quantizer: &impl Quantizer4Bit) {
        for (i, chunk) in matrix.data.chunks(2).enumerate() {
            let v1 = chunk[0];
            let v2 = chunk.get(1).copied().unwrap_or(0.0);
            dst.data[i] = quantize_and_pack(quantizer, v1, v2)
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
}

impl Matrix<u8> {
    /// Multiply 4-bit quantized matrices using i32 accumulators, no output pipeline (returns matrix with accumulators directly)
    /// self in row-major, other in column-major (this is processed by quantize_lhs and quantize_rhs)
    pub fn naive_qmultiply(&self, other: &Self) -> Matrix<i32> {
        let mut result = Vec::<i32>::with_capacity(self.rows * other.cols);
        let mut lower_bits_lhs = true;
        let mut lower_bits_rhs = true;

        let mut lhs_row = Vec::<u8>::with_capacity(self.cols);
        unsafe { lhs_row.set_len(self.cols) };
        let mut rhs_col = Vec::<u8>::with_capacity(other.rows);
        unsafe { rhs_col.set_len(other.rows) };

        let mut i = 0;
        while i < self.data.len() {
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

            let mut j = 0;
            while j < other.data.len() {
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

                let mut dot_prod: i32 = 0;
                for k in 0..self.cols {
                    dot_prod += lhs_row[k] as i32 * rhs_col[k] as i32;
                }
                result.push(dot_prod);
            }
        }

        Matrix {
            data: result,
            rows: self.rows,
            cols: other.cols,
        }
    }

    // Try to avoid memory allocations
    // pub fn naive_qmultiply2(&self, other: &Self) -> Matrix<i32> {
    //     let mut result = Vec::<i32>::new();
    //     let mut lower_bits_lhs = true;
    //     let mut lower_bits_rhs = true;

    //     let depth_pairs = self.cols / 2;

    //     let mut i = 0;
    //     while i < self.data.len() {
    //         let mut j = 0;
    //         while j < other.data.len() {
    //             let mut result_entry: i32 = 0;

    //             // process highest even number <= depth, moving to the next byte each time
    //             for k in 0..depth_pairs {
    //                 let lhs_byte = self.data[i];
    //                 let rhs_byte = other.data[j];

    //                 let lhs_low = lhs_byte & 0x0F;
    //                 let lhs_high = lhs_byte >> 4;
    //                 let rhs_low = rhs_byte & 0x0F;
    //                 let rhs_high = rhs_byte >> 4;

    //                 result_entry += lhs_low as i32 * rhs_low as i32;
    //                 result_entry += lhs_high as i32 * rhs_high as i32;

    //                 i += 1;
    //                 j += 1;
    //             }

    //             // if depth is odd, last element of the vectors is in the next nibble

    //             result.push(dot_prod);
    //         }
    //     }

    //     Matrix {
    //         data: result,
    //         rows: self.rows,
    //         cols: other.cols,
    //     }
    // }
}

#[cfg(test)]
mod tests {
    use crate::quantization::AffineQuantizer;

    use super::*;

    #[test]
    fn naive_qmultiply_odd_depth() {
        // Case 1
        let a: Matrix<u8> = Matrix {
            data: vec![(2 << 4) + 1, (4 << 4) + 3, (6 << 4) + 5],
            rows: 2,
            cols: 3,
        };
        let b: Matrix<u8> = Matrix {
            data: vec![(8 << 4) + 7, (10 << 4) + 9, (12 << 4) + 11],
            rows: 3,
            cols: 2,
        };
        let c = Matrix {
            data: vec![50, 68, 122, 167],
            rows: 2,
            cols: 2,
        };

        assert_eq!(a.naive_qmultiply(&b), c);
    }

    #[test]
    fn naive_qmultiply_even() {
        let a: Matrix<u8> = Matrix {
            data: vec![
                (2 << 4) + 1,
                (4 << 4) + 3,
                (6 << 4) + 5,
                (8 << 4) + 7,
                (10 << 4) + 9,
                (12 << 4) + 11,
                (14 << 4) + 13,
                (15 << 4) + 15,
            ],
            rows: 4,
            cols: 4,
        };
        let c = Matrix {
            data: vec![
                30, 70, 110, 146, 70, 174, 278, 374, 110, 278, 446, 602, 146, 374, 602, 815,
            ],
            rows: 4,
            cols: 4,
        };

        assert_eq!(a.naive_qmultiply(&a), c);
    }

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
                quantize_and_pack(&quantizer, 3., 4.),
                quantize_and_pack(&quantizer, 5., 6.),
                quantize_and_pack(&quantizer, 7., 8.),
                quantize_and_pack(&quantizer, 9., 10.),
                quantize_and_pack(&quantizer, 11., 12.),
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
                quantize_and_pack(&quantizer, 11., 2.),
                quantize_and_pack(&quantizer, 7., 12.),
                quantize_and_pack(&quantizer, 3., 8.),
                quantize_and_pack(&quantizer, 13., 4.),
                quantize_and_pack(&quantizer, 9., 14.),
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
}
