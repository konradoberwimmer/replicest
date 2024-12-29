use std::sync::mpsc;
use nalgebra::{DMatrix, DVector};
use std::thread;
use crate::estimates;

#[derive(Debug)]
#[derive(Clone)]
pub struct ReplicatedEstimates {
    parameter_names: Vec<String>,
    final_estimates: DVector<f64>,
    sampling_variances: DVector<f64>,
    imputation_variances: DVector<f64>,
    standard_errors: DVector<f64>,
}

impl ReplicatedEstimates {
    pub fn parameter_names(&self) -> &Vec<String> {
        &self.parameter_names
    }

    pub fn final_estimates(&self) -> &DVector<f64> {
        &self.final_estimates
    }

    pub fn sampling_variances(&self) -> &DVector<f64> {
        &self.sampling_variances
    }

    pub fn imputation_variances(&self) -> &DVector<f64> {
        &self.imputation_variances
    }

    pub fn standard_errors(&self) -> &DVector<f64> {
        &self.standard_errors
    }
}

pub fn replicate_estimates(estimator: fn(&DMatrix<f64>, &DVector<f64>) -> estimates::Estimates, x: &Vec<&DMatrix<f64>>, weights: &Vec<&DVector<f64>>, replicate_wgts: &Vec<&DMatrix<f64>>, factor: f64) -> ReplicatedEstimates {
    assert!(weights.len() == 1 || weights.len() == x.len(), "length mismatch of weights and data in replicate_estimates");
    assert!(replicate_wgts.len() == 1 || replicate_wgts.len() == x.len(), "length mismatch of replicate weights and data in replicate_estimates");

    let mut parameter_names = Vec::<String>::new();
    let mut estimates = DMatrix::<f64>::zeros(0, 0);
    let mut sampling_variances = DVector::<f64>::zeros(0);

    let empty_matrix: DMatrix<f64> = DMatrix::<f64>::zeros(0, 0);

    let (transmitter, receiver) = mpsc::channel();
    thread::scope(|scope| {
        for imputation in 0..x.len() {
            let data = x[imputation];
            let weight = if weights.len() > 1 { weights[imputation] } else { weights[0] };
            let repweights = match replicate_wgts.len() {
                0 => { &empty_matrix },
                1 => { replicate_wgts[0] },
                _ => { replicate_wgts[imputation] },
            };
            let transmitter1 = transmitter.clone();

            scope.spawn(move || {
                let estimates_imputation = estimator(&data, weight);

                let sampling_variances_imputation: DVector<f64> = if repweights.ncols() > 0 {
                    let mut replicated_estimates: DMatrix<f64> = DMatrix::<f64>::zeros(estimates_imputation.estimates().len(), repweights.ncols());
                    for c in 0..repweights.ncols() {
                        let estimates0 = estimator(&data, &DVector::from(repweights.column(c)));
                        replicated_estimates.set_column(c, &estimates0.estimates());
                    }

                    calc_replication_variance(&estimates_imputation.estimates(), &replicated_estimates, factor)
                } else {
                    DVector::<f64>::zeros(estimates_imputation.estimates().len())
                };
                transmitter1.send((estimates_imputation, sampling_variances_imputation)).unwrap();
            });
        }
    });

    drop(transmitter);
    let mut next_column_estimates = 0;
    for received in receiver {
        parameter_names = received.0.parameter_names().clone();
        if next_column_estimates == 0 {
            estimates = DMatrix::<f64>::zeros(received.0.estimates().len(), x.len());
            sampling_variances = DVector::<f64>::zeros(received.0.estimates().len());
        }
        estimates.set_column(next_column_estimates, &received.0.estimates());
        sampling_variances += &received.1;
        next_column_estimates += 1;
    }

    let final_estimates = DVector::from_fn(estimates.nrows(), |r, _| { estimates.row(r).mean() });
    sampling_variances /= x.len() as f64;
    let imputation_variances = if x.len() > 1 {
        calc_replication_variance(&final_estimates, &estimates, 1.0 / (x.len() - 1) as f64)
    } else {
        DVector::<f64>::zeros(sampling_variances.len())
    };
    let standard_errors = calc_standard_errors_from_variances(&sampling_variances, &imputation_variances, x.len());

    ReplicatedEstimates {
        parameter_names,
        final_estimates,
        sampling_variances,
        imputation_variances,
        standard_errors,
    }
}

fn calc_replication_variance(estimates: &DVector<f64>, replicated_estimates: &DMatrix<f64>, factor: f64) -> DVector<f64> {
    assert_eq!(estimates.len(), replicated_estimates.nrows(), "dimension mismatch of estimates and replicated_estimates in calc_replication_variance");

    let final_estimates_repeated = DMatrix::from_fn(estimates.len(), replicated_estimates.ncols(), |r, _| estimates[r]);
    let deviations = replicated_estimates - final_estimates_repeated;

    DVector::from_fn(deviations.nrows(), |r, _| { deviations.row(r).map(|v| v.powf(2.0_f64)).sum() * factor })
}

fn calc_standard_errors_from_variances(sampling_variances: &DVector<f64>, imputation_variances: &DVector<f64>, n_imp: usize) -> DVector<f64> {
    assert_eq!(sampling_variances.len(), imputation_variances.len(), "dimension mismatch of sampling_variances and imputation_variances in calc_standard_error_from_variances");

    (sampling_variances + (imputation_variances * (1.0 + (1.0 / n_imp as f64)))).map(|v| v.sqrt())
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector};
    use crate::assert_approx_eq_iter_f64;
    use crate::estimates::mean;
    use super::*;

    #[test]
    fn test_replicate_estimate_mean_no_imputation() {
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

        let result = replicate_estimates(mean, &imp_data, &vec![&wgt], &vec![&rep_wgts], 2.0/3.0);
        assert_eq!(result.final_estimates, dvector![2.25, 3.125, 2.0, -2.5]);
        assert_eq!(result.sampling_variances, dvector![0.6370833333333332, 0.18843749999999995, 0.815, 1.0416666666666665]);
        assert_eq!(result.imputation_variances, dvector![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(result.standard_errors, dvector![0.7981750016965786, 0.4340938838546334, 0.9027735042633894, 1.0206207261596574]);
    }

    #[test]
    fn test_replicate_estimate_mean_no_resampling() {
        let mut imp_data: Vec<&DMatrix<f64>> = Vec::new();
        let data0 = DMatrix::from_row_slice(3, 4, &[
            1.0, 4.0, 2.5, -1.0,
            2.5, 1.75, 4.0, -2.5,
            3.0, 3.0, 1.0, -3.5,
        ]);
        imp_data.push(&data0);
        let data1 = DMatrix::from_row_slice(3, 4, &[
            1.2, 4.0, 2.5, -1.0,
            2.5, 1.75, 3.9, -2.5,
            2.7, 3.0, 1.0, -3.5,
        ]);
        imp_data.push(&data1);
        let data2 = DMatrix::from_row_slice(3, 4, &[
            0.8, 4.0, 2.5, -1.0,
            2.5, 1.75, 4.1, -2.5,
            3.3, 3.0, 1.0, -3.5,
        ]);
        imp_data.push(&data2);

        let wgt = dvector![1.0, 0.5, 1.5];
        let rep_wgts = DMatrix::from_row_slice(3, 0, &[]);

        let result = replicate_estimates(mean, &imp_data, &vec![&wgt], &vec![&rep_wgts], 1.0);
        assert_approx_eq_iter_f64!(result.final_estimates, dvector![2.25, 3.125, 2.0, -2.5]);
        assert_approx_eq_iter_f64!(result.sampling_variances, dvector![0.0, 0.0, 0.0, 0.0]);
        assert_approx_eq_iter_f64!(result.imputation_variances, dvector![0.0069444444444443955, 0.0, 0.0002777777777777758, 0.0]);
        assert_approx_eq_iter_f64!(result.standard_errors, dvector![0.09622504486493728, 0.0, 0.01924500897298746, 0.0]);
    }

    #[test]
    fn test_replicate_estimate_mean() {
        let mut imp_data: Vec<&DMatrix<f64>> = Vec::new();
        let data0 = DMatrix::from_row_slice(3, 4, &[
            1.0, 4.0, 2.5, -1.0,
            2.5, 1.75, 4.0, -2.5,
            3.0, 3.0, 1.0, -3.5,
        ]);
        imp_data.push(&data0);
        let data1 = DMatrix::from_row_slice(3, 4, &[
            1.2, 4.0, 2.5, -1.0,
            2.5, 1.75, 3.9, -2.5,
            2.7, 3.0, 1.0, -3.5,
        ]);
        imp_data.push(&data1);
        let data2 = DMatrix::from_row_slice(3, 4, &[
            0.8, 4.0, 2.5, -1.0,
            2.5, 1.75, 4.1, -2.5,
            3.3, 3.0, 1.0, -3.5,
        ]);
        imp_data.push(&data2);

        let wgt = dvector![1.0, 0.5, 1.5];
        let rep_wgts = DMatrix::from_row_slice(3, 3, &[
            0.0, 1.0, 1.0,
            0.5, 0.0, 0.5,
            1.5, 1.5, 0.0,
        ]);

        let result = replicate_estimates(mean, &imp_data, &vec![&wgt], &vec![&rep_wgts], 1.0);
        assert_eq!(4, result.parameter_names.len());
        assert_eq!("mean_x2", result.parameter_names[1]);
        assert_approx_eq_iter_f64!(result.final_estimates, dvector![2.25, 3.125, 2.0, -2.5]);
        assert_approx_eq_iter_f64!(result.sampling_variances, dvector![1.000486111111111, 0.28265624999999994, 1.2229166666666667, 1.5625]);
        assert_approx_eq_iter_f64!(result.imputation_variances, dvector![0.0069444444444443955, 0.0, 0.0002777777777777758, 0.0]);
        assert_approx_eq_iter_f64!(result.standard_errors, dvector![1.0048608711510119, 0.5316542579534184, 1.1060230725608924, 1.25]);
    }

    #[test]
    #[should_panic(expected = "a scoped thread panicked")]
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

        replicate_estimates(mean, &imp_data, &vec![&wgt], &vec![&rep_wgts], 2.0_f64/3.0_f64);
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

        let result = replicate_estimates(mean, &imp_data, &vec![&wgt], &vec![&rep_wgts], 2.0_f64/3.0_f64);
        assert_eq!(1, result.parameter_names.len());
        assert_eq!("mean_x1", result.parameter_names[0]);
        assert_eq!(1, result.final_estimates.len());
        assert_eq!(true, result.final_estimates[0].is_nan());
        assert_eq!(1, result.sampling_variances.len());
        assert_eq!(true, result.sampling_variances[0].is_nan());
    }

    #[test]
    fn test_replicate_estimate_mean_neither_imputation_nor_resampling() {
        let mut imp_data: Vec<&DMatrix<f64>> = Vec::new();
        let data0 = DMatrix::from_row_slice(3, 4, &[
            1.0, 4.0, 2.5, -1.0,
            2.5, 1.75, 4.0, -2.5,
            3.0, 3.0, 1.0, -3.5,
        ]);
        imp_data.push(&data0);

        let wgt = dvector![1.0, 0.5, 1.5];
        let rep_wgts = DMatrix::from_row_slice(3, 0, &[]);

        let result = replicate_estimates(mean, &imp_data, &vec![&wgt], &vec![&rep_wgts], 2.0_f64/3.0_f64);
        assert_eq!(result.final_estimates, dvector![2.25, 3.125, 2.0, -2.5]);
        assert_eq!(result.sampling_variances, dvector![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_calc_replication_variance() {
        let final_estimates = dvector![2.5, 4.0];
        let replicated_estimates = dmatrix![
            2.42, 2.57, 2.49, 2.52;
            4.20, 4.05, 3.80, 3.95;
        ];

        let result = calc_replication_variance(&final_estimates, &replicated_estimates, 1.0);
        assert_eq!(result, dvector![0.011799999999999986, 0.08500000000000012]);

        let result = calc_replication_variance(&final_estimates, &replicated_estimates, 0.5);
        assert_eq!(result, dvector![0.005899999999999993, 0.04250000000000006]);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch of estimates and replicated_estimates in calc_replication_variance")]
    fn test_calc_replication_variance_dimension_mismatch() {
        let final_estimates = dvector![2.5, 4.0];
        let replicated_estimates = dmatrix![
            2.42, 2.57, 2.49, 2.52;
        ];

        calc_replication_variance(&final_estimates, &replicated_estimates, 1.0);
    }

    #[test]
    fn test_calc_replication_variance_with_nan() {
        let final_estimates = dvector![2.5, 4.0, f64::NAN];
        let replicated_estimates = dmatrix![
            2.42, f64::NAN, 2.49, 2.52;
            f64::NAN, f64::NAN, f64::NAN, f64::NAN;
            4.20, 4.05, 3.80, 3.95;
        ];

        let result = calc_replication_variance(&final_estimates, &replicated_estimates, 1.0);
        assert_eq!(3, result.len());
        assert_eq!(true, result[0].is_nan());
        assert_eq!(true, result[1].is_nan());
        assert_eq!(true, result[2].is_nan());
    }

    #[test]
    fn test_calc_standard_errors_from_variances_no_imputation() {
        let sampling_variances = dvector![1.0, 4.0, 0.25];
        let imputation_variances = dvector![0.0, 0.0, 0.0];

        let result = calc_standard_errors_from_variances(&sampling_variances, &imputation_variances, 1);

        assert_eq!(dvector![1.0, 2.0, 0.5], result);
    }

    #[test]
    fn test_calc_standard_errors_from_variances_no_resampling() {
        let sampling_variances = dvector![0.0, 0.0, 0.0];
        let imputation_variances = dvector![1.0 / 1.1, 4.0 / 1.1, 0.25 / 1.1];

        let result = calc_standard_errors_from_variances(&sampling_variances, &imputation_variances, 10);

        assert_eq!(dvector![1.0, 2.0, 0.5], result);
    }

    #[test]
    fn test_calc_standard_errors_from_variances() {
        let sampling_variances = dvector![0.5, 2.0, 0.125];
        let imputation_variances = dvector![1.0 / 2.2, 4.0 / 2.2, 0.25 / 2.2];

        let result = calc_standard_errors_from_variances(&sampling_variances, &imputation_variances, 10);

        assert_eq!(dvector![1.0, 2.0, 0.5], result);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch of sampling_variances and imputation_variances in calc_standard_error_from_variances")]
    fn test_calc_standard_errors_from_variances_dimension_mismatch() {
        let sampling_variances = dvector![0.5, 2.0, 0.125];
        let imputation_variances = dvector![1.0 / 2.2, 4.0 / 2.2];

        calc_standard_errors_from_variances(&sampling_variances, &imputation_variances, 10);
    }

    #[test]
    #[should_panic(expected = "length mismatch of weights and data in replicate_estimates")]
    fn test_replicate_estimates_length_mismatch_weights() {
        let mut imp_data: Vec<&DMatrix<f64>> = Vec::new();
        let data0 = DMatrix::from_row_slice(3, 1, &[ 1.0, 2.5, 3.0 ]);
        imp_data.push(&data0);
        let data1 = DMatrix::from_row_slice(3, 1, &[ 1.2, 2.5, 2.7 ]);
        imp_data.push(&data1);
        let data2 = DMatrix::from_row_slice(3, 1, &[ 0.8, 2.5, 3.3 ]);
        imp_data.push(&data2);

        let wgt = dvector![1.0, 0.5, 1.5];

        replicate_estimates(mean, &imp_data, &vec![&wgt, &wgt], &vec![], 1.0);
    }

    #[test]
    #[should_panic(expected = "length mismatch of replicate weights and data in replicate_estimates")]
    fn test_replicate_estimates_length_mismatch_replicate_weights() {
        let mut imp_data: Vec<&DMatrix<f64>> = Vec::new();
        let data0 = DMatrix::from_row_slice(3, 1, &[ 1.0, 2.5, 3.0 ]);
        imp_data.push(&data0);
        let data1 = DMatrix::from_row_slice(3, 1, &[ 1.2, 2.5, 2.7 ]);
        imp_data.push(&data1);
        let data2 = DMatrix::from_row_slice(3, 1, &[ 0.8, 2.5, 3.3 ]);
        imp_data.push(&data2);

        let wgt = dvector![1.0, 0.5, 1.5];
        let rep_wgts = DMatrix::from_row_slice(3, 3, &[
            0.0, 1.0, 1.0,
            0.5, 0.0, 0.5,
            1.5, 1.5, 0.0,
        ]);

        replicate_estimates(mean, &imp_data, &vec![&wgt, &wgt, &wgt], &vec![&rep_wgts, &rep_wgts, &rep_wgts, &rep_wgts], 1.0);
    }

    #[test]
    fn test_replicate_estimates_multiple_weights() {
        let data1 = dmatrix![1.0; 3.0; 5.0; 7.0; 9.0];
        let data2 = dmatrix![1.0; 7.0; 9.0];
        let data3 = dmatrix![1.0; 3.0; 4.0; 5.0; 7.0; 9.0];
        let data4 = dmatrix![1.0; 4.0; 7.0; 9.0];
        let mut imp_data: Vec<&DMatrix<f64>> = Vec::new();
        imp_data.push(&data1);
        imp_data.push(&data2);
        imp_data.push(&data3);
        imp_data.push(&data4);

        let wgt1 = dvector![1.0, 1.25, 1.5, 1.75, 2.0];
        let wgt2 = dvector![1.0, 1.75, 2.0];
        let wgt3 = dvector![1.0, 1.25, 1.25, 1.5, 1.75, 2.0];
        let wgt4 = dvector![1.0, 1.25, 1.75, 2.0];
        let mut imp_wgt: Vec<&DVector<f64>> = Vec::new();
        imp_wgt.push(&wgt1);
        imp_wgt.push(&wgt2);
        imp_wgt.push(&wgt3);
        imp_wgt.push(&wgt4);

        let repwgt1 = dmatrix![
            2.0, 1.0, 1.0, 1.0, 1.0;
            1.25, 2.5, 1.25, 1.25, 1.25;
            1.5, 1.5, 3.0, 1.5, 1.5;
            1.75, 1.75, 1.75, 3.5, 1.75;
            2.0, 2.0, 2.0, 2.0, 4.0;
        ];
        let repwgt2 = dmatrix![
            2.0, 1.0, 1.0, 1.0, 1.0;
            1.75, 1.75, 1.75, 3.5, 1.75;
            2.0, 2.0, 2.0, 2.0, 4.0;
        ];
        let repwgt3 = dmatrix![
            2.0, 1.0, 1.0, 1.0, 1.0;
            1.25, 2.5, 1.25, 1.25, 1.25;
            1.25, 0.0, 1.25, 1.25, 1.25;
            1.5, 1.5, 3.0, 1.5, 1.5;
            1.75, 1.75, 1.75, 3.5, 1.75;
            2.0, 2.0, 2.0, 2.0, 4.0;
        ];
        let repwgt4 = dmatrix![
            2.0, 1.0, 1.0, 1.0, 1.0;
            1.25, 0.0, 1.25, 1.25, 1.25;
            1.75, 1.75, 1.75, 3.5, 1.75;
            2.0, 2.0, 2.0, 2.0, 4.0;
        ];
        let mut imp_repwgt: Vec<&DMatrix<f64>> = Vec::new();
        imp_repwgt.push(&repwgt1);
        imp_repwgt.push(&repwgt2);
        imp_repwgt.push(&repwgt3);
        imp_repwgt.push(&repwgt4);

        let result = replicate_estimates(mean, &imp_data, &imp_wgt, &imp_repwgt, 1.0);
        assert_eq!(1, result.final_estimates.len());
        assert_approx_eq_iter_f64!(result.final_estimates, vec![5.9289630325814535]);
        assert_approx_eq_iter_f64!(result.sampling_variances, vec![1.1564444389077233]);
        assert_approx_eq_iter_f64!(result.imputation_variances, vec![0.25145762896956225]);
        assert_approx_eq_iter_f64!(result.standard_errors, vec![1.2127516131177383]);
    }
}