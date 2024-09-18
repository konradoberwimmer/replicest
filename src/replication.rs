use nalgebra::{DMatrix, DVector};

fn calc_samp_var(final_estimates: &DVector<f64>, replicated_estimates: &DMatrix<f64>, factor: f64) -> DVector<f64> {
    assert_eq!(final_estimates.len(), replicated_estimates.nrows(), "dimension mismatch of final_estimates and replicated_estimates in calc_samp_var");

    let final_estimates_repeated = DMatrix::from_fn(final_estimates.len(), replicated_estimates.ncols(), |r, _| final_estimates[r]);
    let deviations = replicated_estimates - final_estimates_repeated;

    DVector::from_fn(deviations.nrows(), |r, _| { deviations.row(r).map(|v| v.powf(2.0_f64)).sum().sqrt() * factor })
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector};
    use super::*;

    #[test]
    fn test_calc_samp_var() {
        let final_estimates = dvector![2.5, 4.0];
        let replicated_estimates = dmatrix![
            2.42, 2.57, 2.49, 2.52;
            4.20, 4.05, 3.80, 3.95;
        ];

        let result = calc_samp_var(&final_estimates, &replicated_estimates, 1.0);
        assert_eq!(result, dvector![0.1086278049120021, 0.2915475947422652]);

        let result = calc_samp_var(&final_estimates, &replicated_estimates, 0.5);
        assert_eq!(result, dvector![0.05431390245600105, 0.1457737973711326]);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch of final_estimates and replicated_estimates in calc_samp_var")]
    fn test_calc_samp_var_dimension_mismatch() {
        let final_estimates = dvector![2.5, 4.0];
        let replicated_estimates = dmatrix![
            2.42, 2.57, 2.49, 2.52;
        ];

        calc_samp_var(&final_estimates, &replicated_estimates, 1.0);
    }

    #[test]
    fn test_calc_samp_var_with_nan() {
        let final_estimates = dvector![2.5, 4.0, f64::NAN];
        let replicated_estimates = dmatrix![
            2.42, f64::NAN, 2.49, 2.52;
            f64::NAN, f64::NAN, f64::NAN, f64::NAN;
            4.20, 4.05, 3.80, 3.95;
        ];

        let result = calc_samp_var(&final_estimates, &replicated_estimates, 1.0);
        assert_eq!(3, result.len());
        assert_eq!(true, result[0].is_nan());
        assert_eq!(true, result[1].is_nan());
        assert_eq!(true, result[2].is_nan());
    }
}