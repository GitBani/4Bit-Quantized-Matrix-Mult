// Plot MSE of quantized matrix multiplication
use four_bit_quantized_matrix_mult::matrices::Matrix;
use four_bit_quantized_matrix_mult::quantization::{quantize_multiplier, AffineQuantizer};

use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // As range increases
    let mse = mse_as_range_increases(10);
    dbg!(&mse);
    let root = BitMapBackend::new("mse-range.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let (x_min, x_max) = (
        mse.first().map(|(x, _)| *x).unwrap_or(1.0),
        mse.last().map(|(x, _)| *x).unwrap_or(100.0),
    );
    let y_max = mse.iter().map(|(_, y)| *y).fold(0.0, f32::max) * 1.1; // Add 10% padding

    let mut chart = ChartBuilder::on(&root)
        .caption("MSE vs. Matrix Value Range", ("sans-serif", 40).into_font())
        .margin(30)
        .x_label_area_size(50)
        .y_label_area_size(75)
        .build_cartesian_2d(x_min..x_max, 0f32..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Matrix Value Range")
        .y_desc("Mean Squared Error (MSE)")
        .light_line_style(&WHITE.mix(0.8))
        .draw()?;

    chart
        .draw_series(LineSeries::new(mse.iter().cloned(), &RED))?
        .label("MSE Trend")
        .legend(|(x, y)| PathElement::new(vec![(x - 5, y), (x + 5, y)], &RED));

    chart.draw_series(
        mse.iter()
            .map(|(x, y)| Circle::new((*x, *y), 3, RED.filled())),
    )?;

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    // ==========================================================================================================================================
    // As size increases
    let mse = mse_as_size_increases(50);
    let root = BitMapBackend::new("mse-size.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let (x_min, x_max) = (
        mse.first().map(|(x, _)| *x).unwrap_or(1.0),
        mse.last().map(|(x, _)| *x).unwrap_or(100.0),
    );
    let y_max = mse.iter().map(|(_, y)| *y).fold(0.0, f32::max) * 1.1; // Add 10% padding

    let mut chart = ChartBuilder::on(&root)
        .caption("MSE vs. Matrix Size", ("sans-serif", 40).into_font())
        .margin(30)
        .x_label_area_size(50)
        .y_label_area_size(75)
        .build_cartesian_2d(x_min..x_max, 0f32..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Matrix Size (Square)")
        .y_desc("Mean Squared Error (MSE)")
        .light_line_style(&WHITE.mix(0.8))
        .draw()?;

    chart
        .draw_series(LineSeries::new(mse.iter().cloned(), &RED))?
        .label("MSE Trend")
        .legend(|(x, y)| PathElement::new(vec![(x - 5, y), (x + 5, y)], &RED));

    chart.draw_series(
        mse.iter()
            .map(|(x, y)| Circle::new((*x, *y), 3, RED.filled())),
    )?;

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

fn mean_squared_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Number of values must be equal");
    let n = a.len() as f32;
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f32>()
        / n
}

fn mse_as_range_increases(max_real_value: i32) -> Vec<(f32, f32)> {
    assert!(max_real_value >= 1);

    let mut mses = Vec::new();

    let iterations_per_range = 100;
    let size = 10;

    for range_magnitude in 1..=max_real_value {
        let range_magnitude = range_magnitude as f32;
        let mut sum = 0.0;

        for _ in 0..iterations_per_range {
            let f_lhs = Matrix::random_square(size, -range_magnitude..range_magnitude);
            let f_rhs = Matrix::random_square(size, -range_magnitude..range_magnitude);
            let mut f_result = Matrix::<f32>::new(size, size);

            let size = size as i32;
            unsafe {
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
            }

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

            let real_multiplier =
                lhs_quantizer.scale * rhs_quantizer.scale / result_quantizer.scale;
            let (q_multiplier, rshift) = quantize_multiplier(real_multiplier);

            let q_result = unsafe {
                q_lhs
                    .qmultiply(
                        &q_rhs,
                        lhs_offset,
                        rhs_offset,
                        result_offset,
                        q_multiplier,
                        rshift,
                    )
                    .dequantize(&result_quantizer)
            };

            sum += mean_squared_error(&f_result.data, &q_result.data);
        }

        mses.push((range_magnitude as f32, sum / iterations_per_range as f32));
    }

    mses
}

fn mse_as_size_increases(max_dimension: usize) -> Vec<(f32, f32)> {
    assert!(max_dimension >= 1);

    let mut mses = Vec::new();

    let iterations_per_range = 100;

    for dimension in 1..=max_dimension {
        let mut sum = 0.0;

        for _ in 0..iterations_per_range {
            let f_lhs = Matrix::random_square(dimension, -1f32..1f32);
            let f_rhs = Matrix::random_square(dimension, -1f32..1f32);
            let mut f_result = Matrix::<f32>::new(dimension, dimension);

            let dimension = dimension as i32;
            unsafe {
                blas::sgemm(
                    b'N',
                    b'N',
                    dimension,
                    dimension,
                    dimension,
                    1.0,
                    &f_rhs.data,
                    dimension,
                    &f_lhs.data,
                    dimension,
                    1.0,
                    &mut f_result.data,
                    dimension,
                );
            }

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

            let real_multiplier =
                lhs_quantizer.scale * rhs_quantizer.scale / result_quantizer.scale;
            let (q_multiplier, rshift) = quantize_multiplier(real_multiplier);

            let q_result = unsafe {
                q_lhs
                    .qmultiply(
                        &q_rhs,
                        lhs_offset,
                        rhs_offset,
                        result_offset,
                        q_multiplier,
                        rshift,
                    )
                    .dequantize(&result_quantizer)
            };

            sum += mean_squared_error(&f_result.data, &q_result.data);
        }

        mses.push((dimension as f32, sum / iterations_per_range as f32));
    }

    mses
}
