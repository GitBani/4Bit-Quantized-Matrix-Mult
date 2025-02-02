use super::Quantizer4Bit;

/// 4 bit quantization scheme where quantized values range from 0 to 15
pub struct AffineUnsigned {
    scale: f32,
    zero: f32,
}

impl AffineUnsigned {
    pub fn new(min_val: f32, max_val: f32) -> Self {
        let scale = (max_val - min_val) / 15.0;
        let zero = -min_val / scale;
        AffineUnsigned { scale, zero }
    }
}

impl Quantizer4Bit for AffineUnsigned {
    fn quantize(&self, real_val: f32) -> i8 {
        ((real_val / self.scale + self.zero).round() as i8).clamp(0, 15)
    }

    fn dequantize(&self, q_val: i8) -> f32 {
        self.scale * (q_val as f32 - self.zero)
    }
}
