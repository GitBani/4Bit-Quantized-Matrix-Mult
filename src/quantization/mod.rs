pub mod affine_symmetric;
pub mod affine_unsigned;

pub trait Quantizer4Bit {
    fn quantize(&self, real_val: f32) -> i8;
    fn dequantize(&self, q_val: i8) -> f32;
}
