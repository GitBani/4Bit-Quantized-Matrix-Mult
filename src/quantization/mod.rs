pub mod affine_symmetric;
pub mod affine_unsigned;

use affine_symmetric::AffineSymmetric;
use affine_unsigned::AffineUnsigned;

pub enum QuantizationScheme4Bit {
    Unsigned(AffineUnsigned),
    Symmetric(AffineSymmetric),
}

impl QuantizationScheme4Bit {
    pub fn quantize(&self, real_val: f32) -> i8 {
        match self {
            QuantizationScheme4Bit::Unsigned(q) => q.quantize(real_val),
            QuantizationScheme4Bit::Symmetric(q) => q.quantize(real_val),
        }
    }

    pub fn dequantize(&self, q_val: i8) -> f32 {
        match self {
            QuantizationScheme4Bit::Unsigned(q) => q.dequantize(q_val),
            QuantizationScheme4Bit::Symmetric(q) => q.dequantize(q_val),
        }
    }
}
