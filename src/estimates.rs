use nalgebra::{DMatrix, DVector};
use crate::helper::ExtractValues;

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

pub fn mean(x: &DMatrix<f64>, wgt: &DVector<f64>) -> Estimates {
    assert_eq!(x.nrows(), wgt.len(), "dimension mismatch of x and wgt in mean");
    assert_eq!(0, wgt.iter().filter(|e| e.is_nan()).count(), "wgt contains NaN in mean");

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

pub fn correlation(x: &DMatrix<f64>, wgt: &DVector<f64>) -> Estimates {
    assert_eq!(x.nrows(), wgt.len(), "dimension mismatch of x and wgt in correlation");
    assert_eq!(0, wgt.iter().filter(|e| e.is_nan()).count(), "wgt contains NaN in correlation");

    let means = mean(&x, &wgt).estimates;
    let x_centered = DMatrix::<f64>::from_columns(
        &Vec::from_iter(x.column_iter().enumerate().map(|(i, c)| c.clone_owned() - DVector::<f64>::from_element(c.nrows(), means[i])))
    );
    let x_centered_weighted = DMatrix::<f64>::from_columns(
        &Vec::from_iter(x_centered.column_iter().map(|c| c.component_mul(wgt)))
    );
    let x_centered_transposed = x_centered.transpose();

    let covariance_matrix = (x_centered_transposed * x_centered_weighted).component_div(&DMatrix::<f64>::from_element(x.ncols(), x.ncols(), wgt.sum() - 1.0));

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

#[cfg(test)]
mod tests {
    use nalgebra::{dvector};
    use rand::prelude::*;
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
        assert_eq!(0, (result.estimates - dvector![
            0.0851125127537528, 0.0745170364892616, 0.0999257999354344, 0.0809170707561286, 0.0827147272285658,
            0.1422383888502900, 0.0896882794424707, 0.1052814596478720, 0.0374160358562062, 0.4255072964698915,
            0.0738826454202535, 0.0426837953915197, 0.7256835306983238, 0.1556382159926197, 1.5304209125363695,
            1.0, 0.6772522500178582, 0.5250823516888409, 0.3255890668883828, 0.2291820976004681,
            1.0, 0.3645639949803244, 0.3276950278469914, 0.0801943765003546, 1.0,
            0.1329582523578879, 0.0528937158520245, 1.0, 0.1476852643924908, 1.0,
        ]).iter().filter(|&&v| v.abs() > 1e-10).count());
    }
}
