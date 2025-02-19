use std::ops::{AddAssign, Mul};

use crate::quantization::{quantize_and_pack, Quantizer4Bit};

#[derive(PartialEq, Debug)]
pub struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

// Todo maybe delete this all and make the multiplication a function on slices
impl<T> Matrix<T>
where
    T: Copy + Default + AddAssign + Mul<Output = T>,
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

    pub fn naive_multiply(&self, other: &Matrix<T>) -> Matrix<T> {
        let mut result = Matrix::<T>::new(self.rows, other.cols);

        // for i in 0..self.rows {
        //     for j in 0..other.cols {
        //         for k in 0..self.cols {
        //             let increment_by = self.get_element(i, k) * other.get_element(k, j);
        //             result.increment_element(increment_by, i, j);
        //         }
        //     }
        // }

        result
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
}

#[cfg(test)]
mod tests {
    use crate::quantization::affine_quantizer::AffineQuantizer;

    use super::*;

    // #[test]
    // fn naive_multiply() {
    //     let a = Matrix {
    //         data: vec![1, 2, 3, 4, 5, 6],
    //         rows: 2,
    //         cols: 3,
    //     };
    //     let b = Matrix {
    //         data: vec![7, 8, 9, 10, 11, 12],
    //         rows: 3,
    //         cols: 2,
    //     };
    //     let c = Matrix {
    //         data: vec![58, 64, 139, 154],
    //         rows: 2,
    //         cols: 2,
    //     };

    //     assert_eq!(a.naive_multiply(&b), c)
    // }

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
