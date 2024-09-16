use nalgebra::{DMatrix, DVector};

pub fn mean(x: &DMatrix<f64>, wgt: &DVector<f64>) -> DVector<f64> {
    assert_eq!(x.nrows(), wgt.len(), "dimension mismatch of x and wgt in mean");
    assert_eq!(0, wgt.iter().filter(|e| e.is_nan()).count(), "wgt contains NaN in mean");

    let x_transpose = x.transpose();
    let x_transpose_clean : DMatrix<f64> = x_transpose.map(|e| if e.is_nan() { 0.0_f64 } else { e });
    let x_transpose_ind : DMatrix<f64> = x_transpose.map(|e| if e.is_nan() { 0.0_f64 } else { 1.0_f64 });

    let weighted_sums = x_transpose_clean * wgt;
    let sum_of_weights = x_transpose_ind * wgt;

    weighted_sums.component_div(&sum_of_weights)
}

#[cfg(test)]
mod tests {
    use nalgebra::dvector;
    use super::*;

    #[test]
    fn test_mean() {
        let data = DMatrix::from_row_slice(3, 4, &[
            1.0, 4.0, 2.5, -1.0,
            2.5, 1.75, 4.0, -2.5,
            3.0, 3.0, 1.0, -3.5,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5];

        let result = mean(&data, &wgt);
        assert_eq!(result, dvector![2.25, 3.125, 2.0, -2.5]);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch of x and wgt in mean")]
    fn test_mean_panic_dimension_mismatch() {
        let data = DMatrix::from_row_slice(2, 3, &[
            1.0, 4.0, 2.5,
            2.5, 1.75, 4.0,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5];

        mean(&data, &wgt);
    }

    #[test]
    #[should_panic(expected = "wgt contains NaN in mean")]
    fn test_mean_panic_wgt_containing_nan() {
        let data = DMatrix::from_row_slice(3, 3, &[
            1.0, 4.0, 2.5,
            2.5, 1.75, 4.0,
            3.0, 3.0, 1.0,
        ]);

        let wgt = dvector![1.0, 0.5, f64::NAN];

        mean(&data, &wgt);
    }

    #[test]
    fn test_mean_with_nan() {
        let data = DMatrix::from_row_slice(3, 3, &[
            1.0, 4.0, 2.5,
            2.5, 1.75, f64::NAN,
            3.0, 3.0, 1.5,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5];

        let result = mean(&data, &wgt);
        assert_eq!(result, dvector![2.25, 3.125, 1.9]);
    }

    #[test]
    fn test_mean_all_nan() {
        let data = DMatrix::from_row_slice(3, 1, &[
            f64::NAN,
            f64::NAN,
            f64::NAN,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5];

        let result = mean(&data, &wgt);
        assert_eq!(true, result[0].is_nan());
    }
}
