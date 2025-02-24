use blas::sgemm;

mod matrices;
mod quantization;

fn main() {
    // Testing out BLAS

    let (m, n, k) = (2, 2, 3);
    let a: Vec<f32> = vec![7.0, 10.0, 8.0, 11.0, 9.0, 12.0];
    let b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let mut c: Vec<f32> = vec![0.0; 4];
    unsafe {
        sgemm(b'N', b'N', m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c, m);
    }

    let expected: Vec<f32> = vec![50.0, 68.0, 122.0, 167.0];

    dbg!(&c);
    assert_eq!(&c, &expected);
}
