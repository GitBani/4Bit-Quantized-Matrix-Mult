pub trait Quantizer4Bit {
    fn quantize(&self, real_val: f32) -> u8;
    fn dequantize(&self, q_val: u8) -> f32;
}

pub fn quantize_and_pack(quantizer: &impl Quantizer4Bit, v1: f32, v2: f32) -> u8 {
    let q1 = quantizer.quantize(v1);
    let q2 = quantizer.quantize(v2);
    (q2 << 4) + q1
}

pub fn unpack_and_dequantize(quantizer: &impl Quantizer4Bit, q: u8) -> (f32, f32) {
    let f1 = quantizer.dequantize(q & 0x0F);
    let f2 = quantizer.dequantize(q >> 4);
    (f1, f2)
}

/// Given a real_multiplier in (0, 1), produces a quantize_multiplier (in Q31 format), rshift pair that can be used to to approximate:
/// real_multiplier * x
///
/// by computing:
/// rounding_rshift(FixedPointMultiply(quantized_multiplier, x), rshift)
pub fn quantize_multiplier(mut real_multiplier: f32) -> (i32, i32) {
    assert!(real_multiplier > 0.);
    assert!(real_multiplier < 1.);

    let mut s = 0;
    // Bring the real multiplier into the interval [1/2, 1).
    while real_multiplier < 0.5 {
        real_multiplier *= 2.0;
        s += 1;
    }

    // Now that the real multiplier is in [1/2, 1), we convert it
    // into a fixed-point number.
    let mut q = (real_multiplier * (1_i64 << 31) as f32).round() as i64;
    assert!(q <= (1_i64 << 31));
    // Handle the special case when the real multiplier was so close to 1
    // that its fixed-point approximation was undistinguishable from 1.
    // We handle this by dividing it by two, and remembering to decrement
    // the right shift amount.
    if q == (1_i64 << 31) {
        q /= 2;
        s -= 1;
    }

    assert!(s >= 0);
    assert!(q <= i32::MAX as i64);
    (q as i32, s)
}

/// 4 bit quantization scheme where quantized values range from 0 to 15
pub struct AffineQuantizer {
    pub scale: f32,
    pub zero: u8,
}

impl AffineQuantizer {
    pub fn new(min_val: f32, max_val: f32) -> Self {
        // to avoid edge case of min_val = max_val, pad range
        let min_val = min_val - 1e-6;
        let max_val = max_val + 1e-6;
        let scale = (max_val - min_val) / 15.0; // 4-bit range
        let zero = (-min_val / scale).round().clamp(0., 15.) as u8;
        AffineQuantizer { scale, zero }
    }
}

impl Quantizer4Bit for AffineQuantizer {
    fn quantize(&self, real_val: f32) -> u8 {
        (real_val / self.scale + self.zero as f32)
            .clamp(0., 15.)
            .round() as u8
    }

    fn dequantize(&self, q_val: u8) -> f32 {
        self.scale * (q_val as f32 - self.zero as f32)
    }
}

pub struct LogQuantizer {
    pub scale: f32,
}

impl LogQuantizer {
    pub fn new(min_val: f32, max_val: f32) -> Self {
        let min_val = min_val - 1e-6;
        let max_val = max_val + 1e-6;
        let scale = (max_val.log2() - min_val.log2()) / 15.0; // 4-bit range
        LogQuantizer { scale }
    }
}

impl Quantizer4Bit for LogQuantizer {
    fn quantize(&self, real_val: f32) -> u8 {
        (real_val.log2() / self.scale).clamp(0., 15.).round() as u8
    }

    fn dequantize(&self, q_val: u8) -> f32 {
        2.0f32.powf(q_val as f32 * self.scale)
    }
}
