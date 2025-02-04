use std::ops::{AddAssign, Mul};

use crate::quantization::Quantizer4Bit;

#[derive(PartialEq, Debug)]
pub struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

// All operations are unchecked for performance 👻
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

    // todo consider inlining these
    pub fn get_element(&self, row: usize, col: usize) -> T {
        self.data[row * self.cols + col]
    }

    pub fn set_element(&mut self, value: T, row: usize, col: usize) {
        self.data[row * self.cols + col] = value;
    }

    pub fn increment_element(&mut self, increment_by: T, row: usize, col: usize) {
        self.data[row * self.cols + col] += increment_by;
    }

    pub fn naive_multiply(&self, other: &Matrix<T>) -> Matrix<T> {
        let mut result = Matrix::<T>::new(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    let increment_by = self.get_element(i, k) * other.get_element(k, j);
                    result.increment_element(increment_by, i, j);
                }
            }
        }

        result
    }
}

impl Matrix<f32> {
    pub fn quantize(&self, quantizer: &impl Quantizer4Bit) -> Matrix<i8> {
        let quantized = Matrix::<i8>::new(self.rows, (self.cols + 1) / 2);
        // self.data.chunks(2);

        quantized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naive_multiply() {
        let a = Matrix {
            data: vec![1, 2, 3, 4, 5, 6],
            rows: 2,
            cols: 3,
        };
        let b = Matrix {
            data: vec![7, 8, 9, 10, 11, 12],
            rows: 3,
            cols: 2,
        };
        let c = Matrix {
            data: vec![58, 64, 139, 154],
            rows: 2,
            cols: 2,
        };

        assert_eq!(a.naive_multiply(&b), c)
    }
}
