use std::fmt;
use crate::helper::{ExtractValues, OrderedF64Counts};
use nalgebra::{DMatrix, DVector};

pub struct Estimates {
    parameter_names: Vec<String>,
    estimates: DVector<f64>,
}

impl Estimates {
    pub fn parameter_names(&self) -> &Vec<String> {
        &self.parameter_names
    }

    pub fn estimates(&self) -> &DVector<f64> {
        &self.estimates
    }
}

fn weighted_count_values(x: &DMatrix<f64>, wgt: &DVector<f64>) -> Vec<OrderedF64Counts> {
    let mut counts = Vec::new();
    for _ in 0..x.ncols() {
        counts.push(OrderedF64Counts::new());
    }

    for rr in 0..x.nrows() {
        for cc in 0..x.ncols() {
            counts[cc].push(x[(rr, cc)], wgt[rr]);
        }
    }

    counts
}

macro_rules! assert_validity_of_data_and_weights {
    ( $x: expr, $wgt: expr, $estimate_name: expr ) => {
        assert_eq!($x.nrows(), $wgt.len(), "dimension mismatch of x and wgt in {}", $estimate_name);
        assert_eq!(0, $wgt.iter().filter(|e| e.is_nan()).count(), "wgt contains NaN in {}", $estimate_name);
    };
}

pub fn frequencies(x: &DMatrix<f64>, wgt: &DVector<f64>) -> Estimates {
    assert_validity_of_data_and_weights!(x, wgt, "frequencies");

    let counts = weighted_count_values(&x, &wgt);

    let mut parameter_names = Vec::new();
    let mut estimates : Vec<f64> = Vec::new();

    for (cc, count_column) in counts.iter().enumerate() {
        for count in count_column.get_counts().iter() {
            parameter_names.push(format!("ncases_x{}_{}", cc + 1, count.get_key()));
            estimates.push(count.get_count_cases() as f64);
            parameter_names.push(format!("nweighted_x{}_{}", cc + 1, count.get_key()));
            estimates.push(count.get_count_weighted());
            parameter_names.push(format!("perc_x{}_{}", cc + 1, count.get_key()));
            estimates.push(count.get_count_weighted() / count_column.get_sum_of_weights());
        }
    }

    Estimates {
        parameter_names,
        estimates : DVector::from_vec(estimates),
    }
}

pub fn missings(x: &DMatrix<f64>, wgt: &DVector<f64>) -> Estimates {
    assert_validity_of_data_and_weights!(x, wgt, "missings");

    let mut parameter_names = Vec::new();
    let mut estimates : Vec<f64> = Vec::new();

    let sum_of_weights = wgt.sum();

    for (cc, column) in x.column_iter().enumerate() {
        let missingcases = column.iter().filter(|e| e.is_nan()).count();
        let missingweights = column.iter().zip(wgt.iter()).filter(|(e, _)| e.is_nan()).map(|(_, wgt)| wgt).sum();

        parameter_names.push(format!("missingcases_x{}", cc + 1));
        estimates.push(missingcases as f64);
        parameter_names.push(format!("missingweights_x{}", cc + 1));
        estimates.push(missingweights);
        parameter_names.push(format!("percmissing_x{}", cc + 1));
        estimates.push(missingweights / sum_of_weights);

        parameter_names.push(format!("validcases_x{}", cc + 1));
        estimates.push((column.len() - missingcases) as f64);
        parameter_names.push(format!("validweights_x{}", cc + 1));
        estimates.push(sum_of_weights - missingweights);
        parameter_names.push(format!("percvalid_x{}", cc + 1));
        estimates.push((sum_of_weights - missingweights) / sum_of_weights);
    }

    let mut count_missings_listwise = 0;
    let mut weights_missings_listwise = 0.0;

    for (rr, row) in x.row_iter().enumerate() {
        if row.iter().any(|e| e.is_nan()) {
            count_missings_listwise += 1;
            weights_missings_listwise += wgt[rr];
        }
    }

    parameter_names.push("missingcases_listwise".to_string());
    estimates.push(count_missings_listwise as f64);
    parameter_names.push("missingweights_listwise".to_string());
    estimates.push(weights_missings_listwise);
    parameter_names.push("percmissing_listwise".to_string());
    estimates.push(weights_missings_listwise / sum_of_weights);

    parameter_names.push("validcases_listwise".to_string());
    estimates.push((x.nrows() - count_missings_listwise) as f64);
    parameter_names.push("validweights_listwise".to_string());
    estimates.push(sum_of_weights - weights_missings_listwise);
    parameter_names.push("percvalid_listwise".to_string());
    estimates.push((sum_of_weights - weights_missings_listwise) / sum_of_weights);

    Estimates {
        parameter_names,
        estimates: DVector::from_vec(estimates),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum QuantileType {
    Lower,
    Interpolation,
    Upper,
}

impl fmt::Display for QuantileType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<String> for QuantileType {
    fn from(value: String) -> Self {
        match value.as_str() {
            "Lower" => QuantileType::Lower,
            "Interpolation" => QuantileType::Interpolation,
            "Upper" => QuantileType::Upper,
            _ => QuantileType::Interpolation,
        }
    }
}

pub fn quantiles_with_options(x: &DMatrix<f64>, wgt: &DVector<f64>, quantiles: Vec<f64>, quantile_type: QuantileType) -> Estimates {
    assert_validity_of_data_and_weights!(x, wgt, "quantiles");
    assert!(quantiles.len() > 0, "quantiles are empty");
    assert_eq!(0, quantiles.iter().filter(|e| e.is_nan()).count(), "quantiles contain NaNs");

    let counts = weighted_count_values(&x, &wgt);

    let mut ordered_quantiles = quantiles.clone();
    ordered_quantiles.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut parameter_names = Vec::new();
    let mut estimates = DVector::from_element(counts.len() * quantiles.len(), f64::NAN);

    for (cc, count_column) in counts.iter().enumerate() {
        let mut cumulative_weight = 0.0;
        let mut current_quantile = 0;

        for (vv, count) in count_column.get_counts().iter().enumerate() {
            let old_cumulative_weight = cumulative_weight;
            cumulative_weight += count.get_count_weighted();
            let cumulative_percent = cumulative_weight / count_column.get_sum_of_weights();

            while current_quantile < ordered_quantiles.len() && cumulative_percent > ordered_quantiles[current_quantile] {
                parameter_names.push(format!("quantile_x{}_{}", cc + 1, ordered_quantiles[current_quantile]));

                let raised_weight = old_cumulative_weight + count.get_first_weight();
                let raised_percent = raised_weight / count_column.get_sum_of_weights();

                if raised_percent <= ordered_quantiles[current_quantile] {
                    estimates[cc * quantiles.len() + current_quantile] = count.get_key();
                } else {
                    match quantile_type {
                        QuantileType::Lower => {
                            estimates[cc * quantiles.len() + current_quantile] = if vv > 0 { count_column.get_counts()[vv - 1].get_key() } else { count.get_key() };
                        }
                        QuantileType::Interpolation => {
                            let percent_change = count.get_first_weight() / count_column.get_sum_of_weights();
                            estimates[cc * quantiles.len() + current_quantile] = if vv > 0 {
                                let lower = count_column.get_counts()[vv - 1].get_key();
                                lower + (count.get_key() - lower) * (ordered_quantiles[current_quantile] - old_cumulative_weight / count_column.get_sum_of_weights()) / (percent_change + f64::EPSILON)
                            } else { count.get_key() };
                        }
                        QuantileType::Upper => {
                            estimates[cc * quantiles.len() + current_quantile] = count.get_key();
                        }
                    }
                }
                current_quantile += 1;
            }

            if current_quantile == ordered_quantiles.len() {
                break;
            }
        }

        while current_quantile < ordered_quantiles.len() {
            parameter_names.push(format!("quantile_x{}_{}", cc + 1, ordered_quantiles[current_quantile]));
            estimates[cc * quantiles.len() + current_quantile] = count_column.get_counts().last().unwrap().get_key();
            current_quantile += 1;
        }
    }

    Estimates {
        parameter_names,
        estimates,
    }
}

pub fn quantiles(x: &DMatrix<f64>, wgt: &DVector<f64>) -> Estimates {
    quantiles_with_options(x, wgt, vec![0.25, 0.50, 0.75], QuantileType::Interpolation)
}

pub fn mean(x: &DMatrix<f64>, wgt: &DVector<f64>) -> Estimates {
    assert_validity_of_data_and_weights!(x, wgt, "mean");

    let x_transpose = x.transpose();
    let x_transpose_clean : DMatrix<f64> = x_transpose.map(|e| if e.is_nan() { 0.0_f64 } else { e });
    let x_transpose_ind : DMatrix<f64> = x_transpose.map(|e| if e.is_nan() { 0.0_f64 } else { 1.0_f64 });

    let weighted_sums = x_transpose_clean * wgt;
    let sum_of_weights = x_transpose_ind * wgt;

    Estimates {
        parameter_names: (1..=x.ncols()).into_iter().map(|e| format!("mean_x{}", e)).collect(),
        estimates: weighted_sums.component_div(&sum_of_weights),
    }
}

pub fn correlation_with_options(x: &DMatrix<f64>, wgt: &DVector<f64>, pairwise_delete: bool) -> Estimates {
    assert_validity_of_data_and_weights!(x, wgt, "correlation");

    let means = mean(&x, &wgt).estimates;
    let mut x_centered = DMatrix::<f64>::from_columns(
        &Vec::from_iter(x.column_iter().enumerate().map(|(i, c)| c.clone_owned() - DVector::<f64>::from_element(c.nrows(), means[i])))
    );

    let mut weights_by_column: Vec<DVector<f64>> = (0..x_centered.ncols()).map(|_| wgt.clone()).collect();
    if pairwise_delete {
        for i in 0..x_centered.ncols() {
            weights_by_column.push(wgt.clone());
            for j in 0..x_centered.nrows() {
                if x_centered[(j, i)].is_nan() {
                    x_centered[(j, i)] = 0.0;
                    weights_by_column[i][j] = 0.0;
                }
            }
        }
    }
    let weights_by_column_sum: Vec<f64> = weights_by_column.iter().map(|w| w.sum()).collect();

    let x_centered_weighted = DMatrix::<f64>::from_columns(
        &Vec::from_iter(x_centered.column_iter().map(|c| c.component_mul(wgt)))
    );
    let x_centered_transposed = x_centered.transpose();

    let mut covariance_matrix = x_centered_transposed * x_centered_weighted;
    for i in 0..covariance_matrix.nrows() {
        for j in 0..covariance_matrix.ncols() {
            covariance_matrix[(i, j)] /= weights_by_column_sum[i].min(weights_by_column_sum[j]) - 1.0;
        }
    }

    let standard_deviations : Vec<f64> = covariance_matrix.diagonal().iter().map(|v| v.sqrt()).collect();
    let mut standard_deviations_matrix_inverse = DMatrix::<f64>::zeros(standard_deviations.len(), standard_deviations.len());
    for (i, standard_deviation) in standard_deviations.into_iter().enumerate() {
        standard_deviations_matrix_inverse[(i,i)] = standard_deviation;
    }
    standard_deviations_matrix_inverse = standard_deviations_matrix_inverse.try_inverse().unwrap_or_else(|| panic!("standard deviation matrix not invertible"));

    let correlation_matrix = &standard_deviations_matrix_inverse * &covariance_matrix * &standard_deviations_matrix_inverse;

    let mut estimates = covariance_matrix.extract_lower_triangle();
    for correlation in correlation_matrix.extract_lower_triangle().iter() {
         estimates = estimates.clone().insert_row(estimates.nrows(), correlation.clone());
    }

    let mut parameter_names = Vec::<String>::new();
    let mut parameter_names_correlation = Vec::<String>::new();
    for i in 1..=x.ncols() {
        for j in i..=x.ncols() {
            parameter_names.push(format!("covariance_x{}_x{}", i, j));
            parameter_names_correlation.push(format!("correlation_x{}_x{}", i, j));
        }
    }
    parameter_names.append(&mut parameter_names_correlation);

    Estimates {
        parameter_names,
        estimates,
    }
}

pub fn correlation(x: &DMatrix<f64>, wgt: &DVector<f64>) -> Estimates {
    correlation_with_options(x, wgt, true)
}

pub fn linreg_with_options(x: &DMatrix<f64>, wgt: &DVector<f64>, intercept: bool) -> Estimates {
    assert_validity_of_data_and_weights!(x, wgt, "linreg");
    assert!(x.ncols() > 1 || intercept, "linear regression missing a predictor");

    let dep = x.column(0);
    let pre = if x.ncols() > 1 && intercept {
        let mut pre = DMatrix::<f64>::zeros(x.nrows(), x.ncols());
        pre.set_column(0, &DVector::<f64>::from_element(x.nrows(), 1.0));
        for cc in 1..x.ncols() {
            pre.set_column(cc, &x.column(cc));
        }
        pre
    } else if x.ncols() > 1 && !intercept {
        x.columns(1, x.ncols() - 1).clone_owned()
    } else {
        DMatrix::<f64>::from_element(x.nrows(), 1, 1.0)
    };

    let dep_weighted = DMatrix::<f64>::from_columns(
        &Vec::from_iter(dep.column_iter().map(|c| c.component_mul(&wgt)))
    );
    let pre_weighted = DMatrix::<f64>::from_columns(
        &Vec::from_iter(pre.column_iter().map(|c| c.component_mul(&wgt)))
    );

    let pre_transposed = pre.transpose();
    let pre_transposed_weighted = &pre_transposed * pre_weighted;
    let pre_transposed_dep = pre_transposed * dep_weighted;

    let coeffs = pre_transposed_weighted.qr().solve(&pre_transposed_dep).expect("failed to solve linear regression");
    assert_eq!(coeffs.ncols(), 1, "unrecognized coefficient vector");

    let k = x.ncols() - 1 + intercept as usize;
    let mut parameter_names = Vec::new();
    let mut estimates = Vec::<f64>::new();

    if intercept {
        parameter_names.push("linreg_b_intercept".to_string());
    }
    for xx in 1..x.ncols() {
        parameter_names.push(format!("linreg_b_X{}", xx));
    }
    coeffs.iter().for_each(|v| estimates.push(*v));

    let sum_of_weights = wgt.sum();
    let errors = &dep - (pre * &coeffs);
    let sum_of_squared_errors = DVector::<f64>::from_iterator(dep.nrows(), errors.iter().map(|v| v.powf(2.0))).component_mul(&wgt).sum();
    let sigma = (sum_of_squared_errors / (sum_of_weights - k as f64)).sqrt();
    let dep_mean = &dep.component_mul(&wgt).sum() / sum_of_weights;
    let sum_of_squared_total = DVector::<f64>::from_iterator(dep.nrows(), dep.iter().map(|v| (v - dep_mean).powf(2.0))).component_mul(&wgt).sum();
    let r2 = 1.0 - sum_of_squared_errors / sum_of_squared_total;

    parameter_names.push("linreg_sigma".to_string());
    estimates.push(sigma);
    parameter_names.push("linreg_R2".to_string());
    estimates.push(r2);

    let means = mean(&x, &wgt).estimates;
    let x_full_centered = DMatrix::<f64>::from_columns(
        &Vec::from_iter(x.column_iter().enumerate().map(|(i, c)| c.clone_owned() - DVector::<f64>::from_element(c.nrows(), means[i])))
    );
    let x_full_centered_weighted = DMatrix::<f64>::from_columns(
        &Vec::from_iter(x_full_centered.column_iter().map(|c| c.component_mul(&wgt)))
    );
    let covariance_matrix = x_full_centered.transpose() * x_full_centered_weighted;
    let std_devs = DVector::<f64>::from_iterator(covariance_matrix.nrows(), covariance_matrix.diagonal().iter().map(|v| v.sqrt()));

    let std_coeffs = if !intercept {
        coeffs.component_mul(&std_devs.rows(1, x.ncols() - 1)) / std_devs[0]
    } else if intercept && x.ncols() > 1 {
        coeffs.rows(1, coeffs.nrows() - 1).component_mul(&std_devs.rows(1, x.ncols() - 1)) / std_devs[0]
    } else {
        DVector::<f64>::zeros(0)
    };

    for xx in 1..x.ncols() {
        parameter_names.push(format!("linreg_beta_X{}", xx));
    }
    std_coeffs.iter().for_each(|v| estimates.push(*v));

    Estimates {
        parameter_names,
        estimates: DVector::<f64>::from_vec(estimates),
    }
}

pub fn linreg(x: &DMatrix<f64>, wgt: &DVector<f64>) -> Estimates {
    linreg_with_options(x, wgt, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helper::ImmutableF64Count;
    use crate::{assert_approx_eq_iter_f64, assert_f64count_eq};
    use nalgebra::dvector;
    use rand::prelude::*;

    #[test]
    fn test_weighted_count_values() {
        let data = DMatrix::from_row_slice(12, 2, &[
            1.0, 4.0,
            2.0, 1.0,
            1.0, 3.0,
            1.0, 4.0,
            2.0, 2.0,
            1.0, 3.0,
            1.0, 4.0,
            2.0, 1.0,
            1.0, 3.0,
            1.0, 4.0,
            2.0, 1.0,
            1.0, 3.0,
        ]);

        let wgt = dvector![1.0, 0.5, 1.0, 0.5, 1.5, 1.0, 0.5, 1.5, 1.0, 0.5, 1.5, 1.5];

        let result = weighted_count_values(&data, &wgt);
        assert_eq!(2, result.len());

        assert_eq!(2, result[0].get_counts().len());
        assert_f64count_eq!(ImmutableF64Count::init(1.0, 8, 1.0, 7.0), result[0].get_counts()[0]);
        assert_f64count_eq!(ImmutableF64Count::init(2.0, 4, 0.5, 5.0), result[0].get_counts()[1]);
        assert_eq!(12.0, result[0].get_sum_of_weights());

        assert_eq!(4, result[1].get_counts().len());
        assert_f64count_eq!(ImmutableF64Count::init(1.0, 3, 0.5, 3.5), result[1].get_counts()[0]);
        assert_f64count_eq!(ImmutableF64Count::init(2.0, 1, 1.5, 1.5), result[1].get_counts()[1]);
        assert_f64count_eq!(ImmutableF64Count::init(3.0, 4, 1.0, 4.5), result[1].get_counts()[2]);
        assert_f64count_eq!(ImmutableF64Count::init(4.0, 4, 1.0, 2.5), result[1].get_counts()[3]);
        assert_eq!(12.0, result[1].get_sum_of_weights());
    }

    #[test]
    #[should_panic(expected = "dimension mismatch of x and wgt in frequencies")]
    fn test_frequencies_panic_dimension_mismatch() {
        let data = DMatrix::from_row_slice(2, 3, &[
            1.0, 4.0, 2.5,
            2.5, 1.75, 4.0,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5];

        frequencies(&data, &wgt);
    }

    #[test]
    #[should_panic(expected = "wgt contains NaN in frequencies")]
    fn test_frequencies_panic_wgt_containing_nan() {
        let data = DMatrix::from_row_slice(3, 3, &[
            1.0, 4.0, 2.5,
            2.5, 1.75, 4.0,
            3.0, 3.0, 1.0,
        ]);

        let wgt = dvector![1.0, 0.5, f64::NAN];

        frequencies(&data, &wgt);
    }

    #[test]
    fn test_frequencies() {
        let data = DMatrix::from_row_slice(10, 2, &[
            1.0, 4.0,
            2.0, 1.75,
            3.0, 3.0,
            1.0, 4.0,
            2.0, 1.75,
            3.0, 3.0,
            1.0, 4.0,
            2.0, 1.75,
            3.0, 3.0,
            3.0, 3.0,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5, 1.0, 0.5, 1.5, 1.0, 0.5, 1.5, 1.0];

        let result = frequencies(&data, &wgt);
        assert_eq!(result.parameter_names.len(), 18);
        assert_eq!(result.parameter_names[0], "ncases_x1_1");
        assert_eq!(result.parameter_names[4], "nweighted_x1_2");
        assert_eq!(result.parameter_names[8], "perc_x1_3");
        assert_eq!(result.parameter_names[9], "ncases_x2_1.75");
        assert_eq!(result.parameter_names[13], "nweighted_x2_3");
        assert_eq!(result.parameter_names[17], "perc_x2_4");
        assert_eq!(result.estimates, dvector![3.0, 3.0, 0.3, 3.0, 1.5, 0.15, 4.0, 5.5, 0.55, 3.0, 1.5, 0.15, 4.0, 5.5, 0.55, 3.0, 3.0, 0.3]);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch of x and wgt in missings")]
    fn test_missings_panic_dimension_mismatch() {
        let data = DMatrix::from_row_slice(2, 3, &[
            1.0, 4.0, 2.5,
            2.5, 1.75, 4.0,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5];

        missings(&data, &wgt);
    }

    #[test]
    #[should_panic(expected = "wgt contains NaN in missings")]
    fn test_missings_panic_wgt_containing_nan() {
        let data = DMatrix::from_row_slice(3, 3, &[
            1.0, 4.0, 2.5,
            2.5, 1.75, 4.0,
            3.0, 3.0, 1.0,
        ]);

        let wgt = dvector![1.0, 0.5, f64::NAN];

        missings(&data, &wgt);
    }

    #[test]
    fn test_missings() {
        let data = DMatrix::from_row_slice(10, 2, &[
            1.0, 4.0,
            f64::NAN, 1.75,
            3.0, 3.0,
            f64::NAN, f64::NAN,
            f64::NAN, f64::NAN,
            3.0, f64::NAN,
            f64::NAN, 4.0,
            2.0, 1.75,
            f64::NAN, 3.0,
            3.0, 3.0,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5, 1.0, 0.5, 1.5, 1.0, 0.5, 1.5, 1.0];

        let result = missings(&data, &wgt);
        assert_eq!(result.parameter_names.len(), 18);
        assert_eq!(result.parameter_names[0], "missingcases_x1");
        assert_eq!(result.parameter_names[5], "percvalid_x1");
        assert_eq!(result.parameter_names[6], "missingcases_x2");
        assert_eq!(result.parameter_names[11], "percvalid_x2");
        assert_eq!(result.parameter_names[12], "missingcases_listwise");
        assert_eq!(result.parameter_names[17], "percvalid_listwise");
        assert_eq!(result.estimates, dvector![
            5.0, 4.5, 0.45, 5.0, 5.5, 0.55,
            3.0, 3.0, 0.30, 7.0, 7.0, 0.70,
            6.0, 6.0, 0.60, 4.0, 4.0, 0.40,
        ]);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch of x and wgt in quantiles")]
    fn test_quantiles_panic_dimension_mismatch() {
        let data = DMatrix::from_row_slice(2, 3, &[
            1.0, 4.0, 2.5,
            2.5, 1.75, 4.0,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5];

        quantiles(&data, &wgt);
    }

    #[test]
    #[should_panic(expected = "wgt contains NaN in quantiles")]
    fn test_quantiles_panic_wgt_containing_nan() {
        let data = DMatrix::from_row_slice(3, 3, &[
            1.0, 4.0, 2.5,
            2.5, 1.75, 4.0,
            3.0, 3.0, 1.0,
        ]);

        let wgt = dvector![1.0, 0.5, f64::NAN];

        quantiles(&data, &wgt);
    }

    #[test]
    #[should_panic(expected = "quantiles are empty")]
    fn test_quantiles_panic_quantiles_empty() {
        let data = DMatrix::from_row_slice(3, 3, &[
            1.0, 4.0, 2.5,
            2.5, 1.75, 4.0,
            2.5, 1.75, 4.0,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5];

        quantiles_with_options(&data, &wgt, vec![], QuantileType::Interpolation);
    }

    #[test]
    #[should_panic(expected = "quantiles contain NaNs")]
    fn test_quantiles_panic_quantiles_containing_nan() {
        let data = DMatrix::from_row_slice(3, 3, &[
            1.0, 4.0, 2.5,
            2.5, 1.75, 4.0,
            3.0, 3.0, 1.0,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5];

        quantiles_with_options(&data, &wgt, vec![0.25, 0.50, f64::NAN], QuantileType::Interpolation);
    }

    #[test]
    fn test_quantiles_lower() {
        let data = DMatrix::from_row_slice(10, 2, &[
            1.0, 4.0,
            2.0, 1.75,
            3.0, 3.0,
            1.0, 4.0,
            2.0, 1.75,
            3.0, 3.0,
            1.0, 4.0,
            2.0, 1.75,
            3.0, 3.0,
            3.0, 3.0,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5, 1.0, 0.5, 1.5, 1.0, 0.5, 1.5, 1.0];

        let result = quantiles_with_options(&data, &wgt, vec![0.90, 0.25, 0.50, 0.75, 0.10], QuantileType::Lower);
        assert_eq!(result.parameter_names.len(), 10);
        assert_eq!(result.parameter_names[0], "quantile_x1_0.1");
        assert_eq!(result.parameter_names[8], "quantile_x2_0.75");
        assert_eq!(result.estimates, dvector![
            1.0, 1.0, 2.0, 3.0, 3.0, 1.75, 1.75, 3.0, 3.0, 4.0
        ]);
    }

    #[test]
    fn test_quantiles_interpolation() {
        let data = DMatrix::from_row_slice(10, 2, &[
            1.0, 4.0,
            2.0, 1.75,
            3.0, 3.0,
            1.0, 4.0,
            2.0, 1.75,
            3.0, 3.0,
            1.0, 4.0,
            2.0, 1.75,
            3.0, 3.0,
            3.0, 3.0,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5, 1.0, 0.5, 1.5, 1.0, 0.5, 1.5, 1.0];

        let result = quantiles(&data, &wgt);
        assert_eq!(result.parameter_names.len(), 6);
        assert_eq!(result.parameter_names[1], "quantile_x1_0.5");
        assert_eq!(result.parameter_names[5], "quantile_x2_0.75");
        assert_approx_eq_iter_f64!(result.estimates, dvector![
            1.0, 2.3333333333333333, 3.0, 2.5833333333333333, 3.0, 3.5
        ]);
    }

    #[test]
    fn test_quantiles_upper() {
        let data = DMatrix::from_row_slice(10, 2, &[
            1.0, 4.0,
            2.0, 1.75,
            3.0, 3.0,
            1.0, 4.0,
            2.0, 1.75,
            3.0, 3.0,
            1.0, 4.0,
            2.0, 1.75,
            3.0, 3.0,
            3.0, 3.0,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5, 1.0, 0.5, 1.5, 1.0, 0.5, 1.5, 1.0];

        let result = quantiles_with_options(&data, &wgt, vec![0.10, 0.25, 0.50, 0.75, 0.90], QuantileType::Upper);
        assert_eq!(result.parameter_names.len(), 10);
        assert_eq!(result.parameter_names[1], "quantile_x1_0.25");
        assert_eq!(result.parameter_names[8], "quantile_x2_0.75");
        assert_eq!(result.estimates, dvector![
            1.0, 1.0, 3.0, 3.0, 3.0, 1.75, 3.0, 3.0, 4.0, 4.0
        ]);
    }

    #[test]
    fn test_mean() {
        let data = DMatrix::from_row_slice(3, 4, &[
            1.0, 4.0, 2.5, -1.0,
            2.5, 1.75, 4.0, -2.5,
            3.0, 3.0, 1.0, -3.5,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5];

        let result = mean(&data, &wgt);
        assert_eq!(result.parameter_names.len(), 4);
        assert_eq!(result.parameter_names[1], "mean_x2");
        assert_eq!(result.estimates, dvector![2.25, 3.125, 2.0, -2.5]);
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
        assert_eq!(result.parameter_names.len(), 3);
        assert_eq!(result.parameter_names[2], "mean_x3");
        assert_eq!(result.estimates, dvector![2.25, 3.125, 1.9]);
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
        assert_eq!(result.parameter_names.len(), 1);
        assert_eq!(result.parameter_names[0], "mean_x1");
        assert_eq!(true, result.estimates[0].is_nan());
    }

    #[test]
    fn test_correlation() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(123454321);

        let mut data = DMatrix::<f64>::zeros(100,5);
        data.set_column(0, &DVector::from_iterator(100, (0..100).into_iter().map(|_| rng.gen::<f64>())));

        for cc in 1..5 {
            let mut correlated_values = DVector::from(data.column(0));
            correlated_values += DVector::from_iterator(100, (0..100).into_iter().map(|_| rng.gen::<f64>() * cc as f64));
            data.set_column(cc, &correlated_values);
        }

        let mut writer_data = csv::Writer::from_path("./tests/_output/correl_data.csv").unwrap();
        for row in data.row_iter() {
            writer_data.write_record(row.iter().map(|v| format!("{}", v))).unwrap();
        }
        writer_data.flush().unwrap();

        let wgt = DVector::from_fn(100, |_,_| rng.gen_range(1..=10) as f64);

        let mut writer_wgt = csv::Writer::from_path("./tests/_output/correl_wgt.csv").unwrap();
        for row in wgt.row_iter() {
            writer_wgt.write_record(row.iter().map(|v| format!("{}", v))).unwrap();
        }
        writer_wgt.flush().unwrap();

        let result = correlation(&data, &wgt);

        assert_eq!(result.parameter_names.len(), 30);
        assert_eq!(result.parameter_names[3], "covariance_x1_x4");
        assert_eq!(result.parameter_names[29], "correlation_x5_x5");
        assert_approx_eq_iter_f64!(result.estimates, dvector![
            0.0851125127537528, 0.0745170364892616, 0.0999257999354344, 0.0809170707561286, 0.0827147272285658,
            0.1422383888502900, 0.0896882794424707, 0.1052814596478720, 0.0374160358562062, 0.4255072964698915,
            0.0738826454202535, 0.0426837953915197, 0.7256835306983238, 0.1556382159926197, 1.5304209125363695,
            1.0, 0.6772522500178582, 0.5250823516888409, 0.3255890668883828, 0.2291820976004681,
            1.0, 0.3645639949803244, 0.3276950278469914, 0.0801943765003546, 1.0,
            0.1329582523578879, 0.0528937158520245, 1.0, 0.1476852643924908, 1.0,
        ]);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch of x and wgt in correlation")]
    fn test_correlation_panic_dimension_mismatch() {
        let data = DMatrix::from_row_slice(2, 3, &[
            1.0, 4.0, 2.5,
            2.5, 1.75, 4.0,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5];

        correlation(&data, &wgt);
    }

    #[test]
    #[should_panic(expected = "wgt contains NaN in correlation")]
    fn test_correlation_panic_wgt_containing_nan() {
        let data = DMatrix::from_row_slice(3, 3, &[
            1.0, 4.0, 2.5,
            2.5, 1.75, 4.0,
            3.0, 3.0, 1.0,
        ]);

        let wgt = dvector![1.0, 0.5, f64::NAN];

        correlation(&data, &wgt);
    }

    #[test]
    fn test_correlation_with_nan_pairwise_delete() {
        let data = DMatrix::from_row_slice(5, 3, &[
            1.0, 2.0, 3.0,
            2.0, 1.0, 1.0,
            3.0, 3.0, 3.0,
            4.0, 2.0, f64::NAN,
            5.0, 1.0, 3.0,
        ]);

        let wgt = dvector![1.0, 2.0, 1.0, 1.0, 1.5];

        let result = correlation(&data, &wgt);
        assert_eq!(result.parameter_names.len(), 12);
        assert_eq!(result.parameter_names[2], "covariance_x1_x3");
        assert_approx_eq_iter_f64!(result.estimates, dvector![
            2.3636363636363638, -0.18181818181818182, 0.727272727272726, 0.6433566433566433, 0.484848484848484, 1.131313131313130,
            1.0, -0.147441956154897, 0.4447495899966607, 1.0, 0.56831449608436613, 1.0
        ]);
    }

    #[test]
    fn test_correlation_with_nan_no_pairwise_delete() {
        let data = DMatrix::from_row_slice(5, 3, &[
            1.0, 2.0, 3.0,
            2.0, 1.0, 1.0,
            3.0, 3.0, 3.0,
            4.0, 2.0, f64::NAN,
            5.0, 1.0, 3.0,
        ]);

        let wgt = dvector![1.0, 2.0, 1.0, 1.0, 1.5];

        let result = correlation_with_options(&data, &wgt, false);
        assert_eq!(result.parameter_names.len(), 12);
        assert_eq!(result.parameter_names[2], "covariance_x1_x3");
        assert_approx_eq_iter_f64!(result.estimates, dvector![
            2.3636363636363638, -0.18181818181818182, f64::NAN, 0.6433566433566433, f64::NAN, f64::NAN,
            f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN
        ]);
    }

    #[test]
    #[should_panic(expected = "standard deviation matrix not invertible")]
    fn test_correlation_all_nan() {
        let data = DMatrix::from_row_slice(3, 2, &[
            f64::NAN, 1.0,
            f64::NAN, 2.0,
            f64::NAN, 3.0,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5];

        correlation(&data, &wgt);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch of x and wgt in linreg")]
    fn test_linreg_panic_dimension_mismatch() {
        let data = DMatrix::from_row_slice(2, 3, &[
            1.0, 4.0, 2.5,
            2.5, 1.75, 4.0,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5];

        linreg(&data, &wgt);
    }

    #[test]
    #[should_panic(expected = "wgt contains NaN in linreg")]
    fn test_linreg_panic_wgt_containing_nan() {
        let data = DMatrix::from_row_slice(3, 3, &[
            1.0, 4.0, 2.5,
            2.5, 1.75, 4.0,
            3.0, 3.0, 1.0,
        ]);

        let wgt = dvector![1.0, 0.5, f64::NAN];

        linreg(&data, &wgt);
    }

    #[test]
    #[should_panic(expected = "linear regression missing a predictor")]
    fn test_linreg_panic_no_predictor() {
        let data = DMatrix::from_row_slice(3, 1, &[
            1.0, 4.0, 2.5,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5];

        linreg_with_options(&data, &wgt, false);
    }

    #[test]
    fn test_linreg() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(543212345);

        let mut data = DMatrix::<f64>::zeros(100,5);
        data.set_column(0, &DVector::from_iterator(100, (0..100).into_iter().map(|_| rng.gen::<f64>())));

        for cc in 1..5 {
            let mut correlated_values = DVector::from(data.column(0));
            correlated_values += DVector::from_iterator(100, (0..100).into_iter().map(|_| rng.gen::<f64>() * cc as f64));
            data.set_column(cc, &correlated_values);
        }

        let mut writer_data = csv::Writer::from_path("./tests/_output/linreg_data.csv").unwrap();
        for row in data.row_iter() {
            writer_data.write_record(row.iter().map(|v| format!("{}", v))).unwrap();
        }
        writer_data.flush().unwrap();

        let wgt = DVector::from_fn(100, |_,_| rng.gen_range(1..=10) as f64);

        let mut writer_wgt = csv::Writer::from_path("./tests/_output/linreg_wgt.csv").unwrap();
        for row in wgt.row_iter() {
            writer_wgt.write_record(row.iter().map(|v| format!("{}", v))).unwrap();
        }
        writer_wgt.flush().unwrap();

        let result = linreg(&data, &wgt);

        assert_eq!(result.parameter_names, vec![
            "linreg_b_intercept", "linreg_b_X1", "linreg_b_X2", "linreg_b_X3", "linreg_b_X4",
            "linreg_sigma", "linreg_R2",
            "linreg_beta_X1", "linreg_beta_X2", "linreg_beta_X3", "linreg_beta_X4",
        ]);

        assert_approx_eq_iter_f64!(result.estimates, vec![
            -0.23666168, 0.40318749, 0.12941675, 0.06207778, -0.00264859,
            0.18718991, 0.60802085,
            0.54225732, 0.28499101, 0.19328278, -0.01025211,
        ], 1e-8);

        let result_without_intercept = linreg_with_options(&data, &wgt, false);

        assert_eq!(result_without_intercept.parameter_names, vec![
            "linreg_b_X1", "linreg_b_X2", "linreg_b_X3", "linreg_b_X4",
            "linreg_sigma", "linreg_R2",
            "linreg_beta_X1", "linreg_beta_X2", "linreg_beta_X3", "linreg_beta_X4",
        ]);

        assert_approx_eq_iter_f64!(result_without_intercept.estimates, vec![
            0.35835451, 0.08677286, 0.03483418, -0.02913420,
            0.19738016, 0.56340932,
            0.48196029, 0.19108412, 0.10845825, -0.11277205,
        ], 1e-8);
    }

    #[test]
    fn test_linreg_just_intercept() {
        let data = DMatrix::from_row_slice(3, 1, &[
            1.0, 4.0, 2.5,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5];

        let result = linreg(&data, &wgt);

        assert_eq!(result.parameter_names, vec![
            "linreg_b_intercept",
            "linreg_sigma", "linreg_R2",
        ]);

        assert_approx_eq_iter_f64!(result.estimates, vec![
            2.25,
            1.262438117, 0.0,
        ], 1e-8);
    }

    #[test]
    fn test_linreg_with_nan() {
        let data = DMatrix::from_row_slice(10, 3, &[
            0.0, 0.0, 1.0,
            1.0, 1.0, 2.0,
            0.0, 1.0, 2.0,
            1.0, 1.0, f64::NAN,
            0.0, 0.0, 2.0,
            1.0, 1.0, 2.0,
            0.0, 0.0, 2.0,
            1.0, 1.0, 1.0,
            f64::NAN, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ]);

        let wgt = dvector![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let result = linreg(&data, &wgt);

        assert_eq!(result.parameter_names, vec![
            "linreg_b_intercept", "linreg_b_X1", "linreg_b_X2",
            "linreg_sigma", "linreg_R2",
            "linreg_beta_X1", "linreg_beta_X2",
        ]);

        assert!(result.estimates.iter().all(|v| v.is_nan()));
    }
}
