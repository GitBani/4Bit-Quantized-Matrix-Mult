mod quantization;

use quantization::affine_symmetric::AffineSymmetric;
use quantization::affine_unsigned::AffineUnsigned;
use quantization::Quantizer4Bit;
use rand::Rng;

fn main() {
    // Just some temporary testing
    let mut rng = rand::rng();
    let min_val = -100.0;
    let max_val = 100.0;
    let unsigned_scheme = AffineUnsigned::new(min_val, max_val);
    let symmetric_scheme = AffineSymmetric::new(min_val, max_val);

    for _ in 0..20 {
        let f = rng.random_range(min_val..=max_val);

        let uq = unsigned_scheme.quantize(f);
        let udq = unsigned_scheme.dequantize(uq);
        let symq = symmetric_scheme.quantize(f);
        let symdq = symmetric_scheme.dequantize(symq);

        println!("Float: {f}");
        println!("Unsigned q: {uq}, back: {udq}",);
        println!("Symm q: {symq}, back: {symdq}",);
        println!("")
    }
}
