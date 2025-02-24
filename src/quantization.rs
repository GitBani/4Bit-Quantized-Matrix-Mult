pub trait Quantizer4Bit {
    fn quantize(&self, real_val: f32) -> u8;
    fn dequantize(&self, q_val: i32) -> f32;
}

pub fn quantize_and_pack(quantizer: &impl Quantizer4Bit, v1: f32, v2: f32) -> u8 {
    let q1 = quantizer.quantize(v1);
    let q2 = quantizer.quantize(v2);
    (q2 << 4) + q1
}

/// 4 bit quantization scheme where quantized values range from 0 to 15
/// todo make zero u8
pub struct AffineQuantizer {
    scale: f32,
    zero: f32,
}

impl AffineQuantizer {
    pub fn new(min_val: f32, max_val: f32) -> Self {
        let scale = (max_val - min_val) / 15.0;
        let zero = -min_val / scale;
        AffineQuantizer { scale, zero }
    }
}

impl Quantizer4Bit for AffineQuantizer {
    fn quantize(&self, real_val: f32) -> u8 {
        (real_val / self.scale + self.zero).clamp(0., 15.).round() as u8
    }

    fn dequantize(&self, q_val: i32) -> f32 {
        self.scale * (q_val as f32 - self.zero)
    }
}
