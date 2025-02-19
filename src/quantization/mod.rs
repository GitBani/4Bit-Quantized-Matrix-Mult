pub mod affine_quantizer;
pub mod affine_symmetric;

pub trait Quantizer4Bit {
    fn quantize(&self, real_val: f32) -> u8;
    fn dequantize(&self, q_val: i32) -> f32;
}

pub fn quantize_and_pack(quantizer: &impl Quantizer4Bit, v1: f32, v2: f32) -> u8 {
    let q1 = quantizer.quantize(v1);
    let q2 = quantizer.quantize(v2);
    (q2 << 4) + q1
}
