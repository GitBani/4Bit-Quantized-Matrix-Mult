use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use four_bit_quantized_matrix_mult::{
    matrices::Matrix,
    quantization::{self, quantize_multiplier},
};
use quantization::AffineQuantizer;

fn matrix_mult_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("MatrixMult");
    group.sample_size(10);
    group.warm_up_time(Duration::new(0, 1000));

    let fmin: f32 = -1.0;
    let fmax: f32 = 1.0;

    for i in 8..=20 {
        let f_lhs = Matrix::random_square(i, fmin..fmax);
        let f_rhs = Matrix::random_square(i, fmin..fmax);
        let mut f_result = Matrix::<f32>::new(i, i);

        let size = i as i32;
        group.bench_with_input(BenchmarkId::new("BLAS", i), &size, |b, &size| {
            b.iter(|| unsafe {
                blas::sgemm(
                    b'N',
                    b'N',
                    size,
                    size,
                    size,
                    1.0,
                    &f_rhs.data,
                    size,
                    &f_lhs.data,
                    size,
                    1.0,
                    &mut f_result.data,
                    size,
                );
            });
        });

        let (lhs_min, lhs_max) = f_lhs.min_and_max();
        let (rhs_min, rhs_max) = f_rhs.min_and_max();
        let (result_min, result_max) = f_result.min_and_max();

        let lhs_quantizer = AffineQuantizer::new(lhs_min, lhs_max);
        let rhs_quantizer = AffineQuantizer::new(rhs_min, rhs_max);
        let result_quantizer = AffineQuantizer::new(result_min, result_max);

        let q_lhs = f_lhs.quantize_lhs(&lhs_quantizer);
        let q_rhs = f_rhs.quantize_rhs(&rhs_quantizer);

        let lhs_offset = -(lhs_quantizer.zero as i32);
        let rhs_offset = -(rhs_quantizer.zero as i32);
        let result_offset = result_quantizer.zero as i32;

        let real_multiplier = lhs_quantizer.scale * rhs_quantizer.scale / result_quantizer.scale;
        let (q_multiplier, rshift) = quantize_multiplier(real_multiplier);

        group.bench_function(BenchmarkId::new("Quantized", i), |b| {
            b.iter(|| unsafe {
                q_lhs.qmultiply(
                    &q_rhs,
                    lhs_offset,
                    rhs_offset,
                    result_offset,
                    q_multiplier,
                    rshift,
                )
            });
        });
    }
}

criterion_group!(benches, matrix_mult_benchmark);
criterion_main!(benches);
