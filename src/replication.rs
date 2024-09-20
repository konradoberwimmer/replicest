use nalgebra::{DMatrix, DVector};

pub struct ReplicatedEstimates {
    estimates: DVector<f64>,
    sampling_variances: DVector<f64>,
}

pub fn replicate_estimates(estimator: fn(&DMatrix<f64>, &DVector<f64>) -> DVector<f64>, x: &Vec<&DMatrix<f64>>, wgt: &DVector<f64>, replicate_wgts: &DMatrix<f64>, factor: f64) -> ReplicatedEstimates {
    let estimates = estimator(&x[0], &wgt);

    let sampling_variances : DVector<f64> = if replicate_wgts.ncols() > 0 {
        let mut replicated_estimates : DMatrix<f64> = DMatrix::<f64>::zeros(x[0].ncols(), replicate_wgts.ncols());
        for c in 0..replicate_wgts.ncols() {
            let estimates0 = estimator(&x[0], &DVector::from(replicate_wgts.column(c)));
            replicated_estimates.set_column(c, &estimates0);
        }

        calc_samp_var(&estimates, &replicated_estimates, factor)
    } else {
        DVector::<f64>::zeros(x[0].ncols())
    };

    ReplicatedEstimates {
        estimates,
        sampling_variances,
    }
}

fn calc_samp_var(final_estimates: &DVector<f64>, replicated_estimates: &DMatrix<f64>, factor: f64) -> DVector<f64> {
    assert_eq!(final_estimates.len(), replicated_estimates.nrows(), "dimension mismatch of final_estimates and replicated_estimates in calc_samp_var");

    let final_estimates_repeated = DMatrix::from_fn(final_estimates.len(), replicated_estimates.ncols(), |r, _| final_estimates[r]);
    let deviations = replicated_estimates - final_estimates_repeated;

    DVector::from_fn(deviations.nrows(), |r, _| { deviations.row(r).map(|v| v.powf(2.0_f64)).sum() * factor })
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector};
    use super::*;

    #[test]
    fn test_replicate_estimate_mean() {
        let mut imp_data: Vec<&DMatrix<f64>> = Vec::new();
        let data0 = DMatrix::from_row_slice(3, 4, &[
            1.0, 4.0, 2.5, -1.0,
            2.5, 1.75, 4.0, -2.5,
            3.0, 3.0, 1.0, -3.5,
        ]);
        imp_data.push(&data0);

        let wgt = dvector![1.0, 0.5, 1.5];
        let rep_wgts = DMatrix::from_row_slice(3, 3, &[
            0.0, 1.0, 1.0,
            0.5, 0.0, 0.5,
            1.5, 1.5, 0.0,
        ]);

        let result = replicate_estimates(crate::estimates::mean, &imp_data, &wgt, &rep_wgts, 2.0_f64/3.0_f64);
        assert_eq!(result.estimates, dvector![2.25, 3.125, 2.0, -2.5]);
        assert_eq!(result.sampling_variances, dvector![0.6370833333333332, 0.18843749999999995, 0.815, 1.0416666666666665]);
    }

    #[test]
    #[should_panic(expected = "wgt contains NaN in mean")]
    fn test_replicate_estimate_mean_nan_in_replicate_weight() {
        let mut imp_data: Vec<&DMatrix<f64>> = Vec::new();
        let data0 = DMatrix::from_row_slice(3, 4, &[
            1.0, 4.0, 2.5, -1.0,
            2.5, 1.75, 4.0, -2.5,
            3.0, 3.0, 1.0, -3.5,
        ]);
        imp_data.push(&data0);

        let wgt = dvector![1.0, 0.5, 1.5];
        let rep_wgts = DMatrix::from_row_slice(3, 3, &[
            0.0, 1.0, 1.0,
            0.5, f64::NAN, 0.5,
            1.5, 1.5, 0.0,
        ]);

        replicate_estimates(crate::estimates::mean, &imp_data, &wgt, &rep_wgts, 2.0_f64/3.0_f64);
    }

    #[test]
    fn test_replicate_estimate_mean_all_nan() {
        let mut imp_data: Vec<&DMatrix<f64>> = Vec::new();
        let data0 = DMatrix::from_row_slice(3, 1, &[
            f64::NAN,
            f64::NAN,
            f64::NAN,
        ]);
        imp_data.push(&data0);

        let wgt = dvector![1.0, 0.5, 1.5];
        let rep_wgts = DMatrix::from_row_slice(3, 3, &[
            0.0, 1.0, 1.0,
            0.5, 0.0, 0.5,
            1.5, 1.5, 0.0,
        ]);

        let result = replicate_estimates(crate::estimates::mean, &imp_data, &wgt, &rep_wgts, 2.0_f64/3.0_f64);
        assert_eq!(1, result.estimates.len());
        assert_eq!(true, result.estimates[0].is_nan());
        assert_eq!(1, result.sampling_variances.len());
        assert_eq!(true, result.sampling_variances[0].is_nan());
    }

    #[test]
    fn test_calc_samp_var() {
        let final_estimates = dvector![2.5, 4.0];
        let replicated_estimates = dmatrix![
            2.42, 2.57, 2.49, 2.52;
            4.20, 4.05, 3.80, 3.95;
        ];

        let result = calc_samp_var(&final_estimates, &replicated_estimates, 1.0);
        assert_eq!(result, dvector![0.011799999999999986, 0.08500000000000012]);

        let result = calc_samp_var(&final_estimates, &replicated_estimates, 0.5);
        assert_eq!(result, dvector![0.005899999999999993, 0.04250000000000006]);
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