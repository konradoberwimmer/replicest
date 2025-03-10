use std::collections::HashMap;
use std::sync::Arc;
use nalgebra::{DMatrix, DVector, Dyn, Matrix, U1};
use serde::{Deserialize, Serialize};
use crate::{estimates, replication};

pub enum Estimate {
    Frequencies,
    Quantiles,
    Mean,
    Correlation,
    LinearRegression,
}

#[derive(Serialize, Deserialize)]
pub struct ReplicatedEstimates {
    pub parameter_names: Vec<String>,
    pub final_estimates: Vec<f64>,
    pub sampling_variances: Vec<f64>,
    pub imputation_variances: Vec<f64>,
    pub standard_errors: Vec<f64>,
}

impl ReplicatedEstimates {
    pub fn from_internal(internal_struct: &replication::ReplicatedEstimates) -> ReplicatedEstimates {
        ReplicatedEstimates {
            parameter_names: internal_struct.parameter_names().clone(),
            final_estimates: Vec::from(internal_struct.final_estimates().as_slice()),
            sampling_variances: Vec::from(internal_struct.sampling_variances().as_slice()),
            imputation_variances: Vec::from(internal_struct.imputation_variances().as_slice()),
            standard_errors: Vec::from(internal_struct.standard_errors().as_slice()),
        }
    }
}

pub fn replicate_estimates(estimate: Estimate, options: HashMap<String, String>, x: &Vec<Vec<Vec<f64>>>, wgt: &Vec<Vec<f64>>, replicate_wgts: &Vec<Vec<Vec<f64>>>, factor: f64) -> ReplicatedEstimates {
    let estimate_function : Arc<dyn Fn(&DMatrix<f64>, &DVector<f64>) -> estimates::Estimates + Send + Sync> = match estimate {
        Estimate::Frequencies => { Arc::new(estimates::frequencies) }
        Estimate::Quantiles => {
            let quantiles = if options.contains_key("quantiles") {
                options["quantiles"].split(",").map(|v| v.parse().unwrap()).collect()
            } else {
                vec![0.25, 0.50, 0.75]
            };
            let quantile_type = if options.contains_key("quantile_type") {
                options["quantile_type"].clone().into()
            } else {
                estimates::QuantileType::Interpolation
            };
            Arc::new(move |x, wgt| estimates::quantiles_with_options(x, wgt, quantiles.clone(), quantile_type.clone()))
        }
        Estimate::Mean => { Arc::new(estimates::mean) }
        Estimate::Correlation => { Arc::new(estimates::correlation) }
        Estimate::LinearRegression => {
            let intercept = if options.contains_key("intercept") {
                if options["intercept"] == "true" {
                    true
                } else {
                    false
                }
            } else {
                true
            };
            Arc::new(move |x, wgt| estimates::linreg_with_options(x, wgt, intercept))
        }
    };

    let mut data : Vec<DMatrix<f64>> = Vec::new();
    for imputation in x.iter() {
        let mut imp_matrix : DMatrix<f64> = DMatrix::<f64>::zeros(imputation.len(), imputation[0].len());
        for (r, row )in imputation.into_iter().enumerate() {
            imp_matrix.set_row(r, &Matrix::<f64, U1, Dyn, _>::from_row_slice(row));
        }

        data.push(imp_matrix);
    }
    let ref_data : Vec<&DMatrix<f64>> = Vec::from_iter(data.iter());

    let mut weights : Vec<DVector<f64>> = Vec::new();
    for weight in wgt.iter() {
        weights.push(DVector::<f64>::from_row_slice(&weight))
    }
    let ref_weights : Vec<&DVector<f64>> = Vec::from_iter(weights.iter());

    let mut replicate_weights : Vec<DMatrix<f64>> = Vec::new();
    for replicate_weight in replicate_wgts.iter() {
        let mut rep_wgt_matrix : DMatrix<f64> = DMatrix::<f64>::zeros(replicate_weight.len(), if replicate_weight.len() == 0 { 0 } else { replicate_weight[0].len() });
        for (r, row) in replicate_weight.into_iter().enumerate() {
            rep_wgt_matrix.set_row(r, &Matrix::<f64, U1, Dyn, _>::from_row_slice(row));
        }

        replicate_weights.push(rep_wgt_matrix);
    }
    let ref_replicate_weights : Vec<&DMatrix<f64>> = Vec::from_iter(replicate_weights.iter());

    let result = replication::replicate_estimates(
        estimate_function,
        &ref_data,
        &ref_weights,
        &ref_replicate_weights,
        factor
    );

    ReplicatedEstimates::from_internal(&result)
}

#[cfg(test)]
mod tests {
    use crate::assert_approx_eq_iter_f64;
    use super::*;

    #[test]
    fn test_replicate_estimates() {
        let imp_data = vec![
            vec![
                vec![1.0, 4.0, 2.5, -1.0],
                vec![2.5, 1.75, 4.0, -2.5],
                vec![3.0, 3.0, 1.0, -3.5],
            ],
            vec![
                vec![1.2, 4.0, 2.5, -1.0],
                vec![2.5, 1.75, 3.9, -2.5],
                vec![2.7, 3.0, 1.0, -3.5],
            ],
            vec![
                vec![0.8, 4.0, 2.5, -1.0],
                vec![2.5, 1.75, 4.1, -2.5],
                vec![3.3, 3.0, 1.0, -3.5],
            ]
        ];

        let wgt = vec![1.0, 0.5, 1.5];
        let rep_wgts = vec![
            vec![0.0, 1.0, 1.0],
            vec![0.5, 0.0, 0.5],
            vec![1.5, 1.5, 0.0],
        ];

        let result = replicate_estimates(Estimate::Mean, HashMap::new(), &imp_data, &vec![wgt], &vec![rep_wgts], 1.0);
        assert_eq!(4, result.parameter_names.len());
        assert_eq!("mean_x2", result.parameter_names[1]);

        let expected_final_estimates = vec![2.25, 3.125, 2.0, -2.5];
        let expected_sampling_variances = vec![1.000486111111111, 0.28265624999999994, 1.2229166666666667, 1.5625];
        let expected_imputation_variances = vec![0.0069444444444443955, 0.0, 0.0002777777777777758, 0.0];
        let expected_standard_errors = vec![1.0048608711510119, 0.5316542579534184, 1.1060230725608924, 1.25];

        assert_approx_eq_iter_f64!(result.final_estimates, expected_final_estimates);
        assert_approx_eq_iter_f64!(result.sampling_variances, expected_sampling_variances);
        assert_approx_eq_iter_f64!(result.imputation_variances, expected_imputation_variances);
        assert_approx_eq_iter_f64!(result.standard_errors, expected_standard_errors);
    }

    #[test]
    fn test_replicate_estimates_with_options() {
        let data = vec![
            vec![
                vec![1.0, 4.0],
                vec![2.5, 1.75],
                vec![3.0, 3.0],
            ]
        ];

        let wgt = vec![1.0, 0.5, 1.5];
        let rep_wgts = vec![
            vec![0.0, 1.0, 1.0],
            vec![0.5, 0.0, 0.5],
            vec![1.5, 1.5, 0.0],
        ];

        let options = HashMap::from([("intercept".to_string(), "false".to_string())]);

        let result = replicate_estimates(Estimate::LinearRegression, options, &data, &vec![wgt], &vec![rep_wgts], 1.0);
        assert_eq!(4, result.parameter_names.len());
        assert_eq!(vec!["linreg_b_X1", "linreg_sigma", "linreg_R2", "linreg_beta_X1"], result.parameter_names);

        let expected_final_estimates = vec![0.6344410876132931, 1.6022548311072888, -1.1064373692772485, 0.53516843619415233];
        let expected_imputation_variances = vec![0.0, 0.0, 0.0, 0.0];
        let expected_standard_errors = vec![0.4983981196204999, 1.1609336342898504, 0.85450312343302826, 2.0876620405907396];

        assert_approx_eq_iter_f64!(result.final_estimates, expected_final_estimates);
        assert_approx_eq_iter_f64!(result.imputation_variances, expected_imputation_variances);
        assert_approx_eq_iter_f64!(result.standard_errors, expected_standard_errors);
    }
}