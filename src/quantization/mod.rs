pub mod affine_symmetric;
pub mod affine_unsigned;

pub trait Quantizer4Bit {
    fn quantize(&self, real_val: f32) -> i8;
    fn dequantize(&self, q_val: i8) -> f32;
}

pub fn quantize_and_pack(quantizer: &impl Quantizer4Bit, v1: f32, v2: f32) -> i8 {
    let q1 = quantizer.quantize(v1);
    let q2 = quantizer.quantize(v2);
    (q2 << 4) + q1
}
