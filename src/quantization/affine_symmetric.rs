// use super::Quantizer4Bit;

// 4 bit quantization scheme where quantized values range from -7 to 7
// ! Temporarily abandoned, may be permanent
// pub struct AffineSymmetric {
//     scale: f32,
// }

// impl AffineSymmetric {
//     pub fn new(min_val: f32, max_val: f32) -> Self {
//         let scale = (max_val - min_val) / 14.0;
//         AffineSymmetric { scale }
//     }
// }

// impl Quantizer4Bit for AffineSymmetric {
//     fn quantize(&self, real_val: f32) -> i8 {
//         ((real_val / self.scale).round() as i8).clamp(-7, 7)
//     }

//     fn dequantize(&self, q_val: i8) -> f32 {
//         self.scale * q_val as f32
//     }
// }
