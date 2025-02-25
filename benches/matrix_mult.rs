use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use four_bit_quantized_matrix_mult::{matrices::Matrix, quantization};
use quantization::AffineQuantizer;

fn matrix_mult_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("MatrixMult");
    group.sample_size(10);
    group.warm_up_time(Duration::new(0, 1000));

    let fmin: f32 = -10.0;
    let fmax: f32 = 10.0;
    let quantizer = AffineQuantizer::new(fmin, fmax);

    for i in 1..=20 {
        let fmm_lhs = Matrix::random_square(i, fmin..fmax);
        let fmm_rhs = Matrix::random_square(i, fmin..fmax);
        let mut fmm_result = Matrix::<f32>::new(i, i);

        let qmm_lhs = fmm_lhs.quantize_lhs(&quantizer);
        let qmm_rhs = fmm_rhs.quantize_rhs(&quantizer);

        group.bench_with_input(BenchmarkId::new("BLAS", i), &i, |b, &i| {
            b.iter(|| unsafe {
                blas::sgemm(
                    b'N',
                    b'N',
                    i as i32,
                    i as i32,
                    i as i32,
                    1.0,
                    &fmm_rhs.data,
                    i as i32,
                    &fmm_lhs.data,
                    i as i32,
                    1.0,
                    &mut fmm_result.data,
                    i as i32,
                );
            });
        });

        group.bench_function(BenchmarkId::new("Quantized", i), |b| {
            b.iter(|| qmm_lhs.naive_qmultiply(&qmm_rhs));
        });
    }
}

criterion_group!(benches, matrix_mult_benchmark);
criterion_main!(benches);
