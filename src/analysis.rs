use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;
use nalgebra::{DMatrix, DVector};
use crate::errors::{InconsistencyError, MissingElementError};
use crate::estimates;
use crate::helper::Split;
use crate::replication::{replicate_estimates, ReplicatedEstimates};

pub enum Imputation<'a> {
    Yes(&'a Vec<&'a DMatrix<f64>>),
    No(&'a DMatrix<f64>),
}

pub struct Analysis {
    x: Option<Rc<Vec<DMatrix<f64>>>>,
    wgt: Option<Rc<DVector<f64>>>,
    repwgts: Option<Rc<DMatrix<f64>>>,
    variance_adjustment_factor: f64,
    estimate_name: Option<String>,
    estimate: Option<Arc<dyn Fn(&DMatrix<f64>, &DVector<f64>) -> estimates::Estimates + Send + Sync>>,
    groups: Option<Rc<Vec<DMatrix<f64>>>>,
    options: HashMap<String, String>,
}

pub fn analysis() -> Analysis {
    Analysis {
        x: None,
        wgt: None,
        repwgts: None,
        variance_adjustment_factor: 1.0,
        estimate_name: None,
        estimate: None,
        groups: None,
        options: HashMap::new(),
    }
}

impl Analysis {
    pub fn for_data(&mut self, data: Imputation) -> &mut Self {
        let mut new_vec : Vec<DMatrix<f64>> = Vec::new();

        match data {
            Imputation::Yes(&ref vec) => {
                for &mat in vec.iter() {
                    new_vec.push(mat.clone());
                }
            }
            Imputation::No(&ref mat) => {
                new_vec.push(mat.clone());
            }
        }

        self.x = Some(Rc::new(new_vec));
        self
    }

    pub fn set_weights(&mut self, wgt: &DVector<f64>) -> &mut Self {
        self.wgt = Some(Rc::new(wgt.clone()));
        self
    }

    pub fn with_replicate_weights(&mut self, replicate_weights: &DMatrix<f64>) -> &mut Self {
        self.repwgts = Some(Rc::new(replicate_weights.clone()));
        self
    }

    pub fn set_variance_adjustment_factor(&mut self, variance_adjustment_factor: f64) -> &mut Self {
        self.variance_adjustment_factor = variance_adjustment_factor;
        self
    }

    pub fn mean(&mut self) -> &mut Self {
        self.estimate_name = Some("mean".to_string());
        self.estimate = Some(Arc::new(estimates::mean));
        self
    }

    pub fn correlation(&mut self) -> &mut Self {
        self.estimate_name = Some("correlation".to_string());
        self.estimate = Some(Arc::new(estimates::correlation));
        self
    }

    pub fn linreg(&mut self) -> &mut Self {
        self.estimate_name = Some("linreg".to_string());
        let intercept = if self.options.contains_key("intercept") {
            if self.options["intercept"] == "true" {
                true
            } else {
                false
            }
        } else {
            true
        };
        self.estimate = Some(Arc::new(move |x, wgt| estimates::linreg_with_options(x, wgt, intercept.clone())));
        self
    }

    pub fn with_intercept(&mut self, intercept: bool) -> &mut Self {
        self.options.insert("intercept".to_string(), intercept.to_string());
        self.linreg()
    }

    pub fn quantiles(&mut self) -> &mut Self {
        self.estimate_name = Some("quantiles".to_string());
        let quantiles = if self.options.contains_key("quantiles") {
            self.options["quantiles"].split(",").map(|v| v.parse().unwrap()).collect()
        } else {
            vec![0.25, 0.50, 0.75]
        };
        let quantile_type = if self.options.contains_key("quantile_type") {
            self.options["quantile_type"].clone().into()
        } else {
            estimates::QuantileType::Interpolation
        };
        self.estimate = Some(Arc::new(move |x, wgt| estimates::quantiles_with_options(x, wgt, quantiles.clone(), quantile_type.clone())));
        self
    }

    pub fn set_quantiles(&mut self, quantiles: Vec<f64>) -> &mut Self {
        self.options.insert("quantiles".to_string(), quantiles.iter().map(|v| v.to_string()).collect::<Vec<String>>().join(","));
        self.quantiles()
    }

    pub fn set_quantile_type(&mut self, quantile_type: estimates::QuantileType) -> &mut Self {
        self.options.insert("quantile_type".to_string(), quantile_type.to_string());
        self.quantiles()
    }

    pub fn group_by(&mut self, data: Imputation) -> &mut Self {
        let mut new_vec : Vec<DMatrix<f64>> = Vec::new();

        match data {
            Imputation::Yes(&ref vec) => {
                for &mat in vec.iter() {
                    new_vec.push(mat.clone());
                }
            }
            Imputation::No(&ref mat) => {
                new_vec.push(mat.clone());
            }
        }

        self.groups = Some(Rc::new(new_vec));
        self
    }

    fn prepare_missing_weights(&mut self) -> Result<(), Box<dyn Error>> {
        if self.x.is_none() || self.x.as_ref().unwrap().deref().len() == 0 {
            return Err(Box::new(MissingElementError::new("data")))
        }

        let ncases = self.x.as_ref().unwrap().deref()[0].nrows();

        if self.wgt.is_none() {
            self.wgt = Some(Rc::new(DVector::<f64>::from_element(ncases, 1.0)));
        }

        if self.repwgts.is_none() {
            self.repwgts = Some(Rc::new(DMatrix::<f64>::from_row_slice(ncases, 0, &[])));
        }

        Ok(())
    }

    fn prepare_for_calculate_overall(&self)
        -> Result<(HashSet<Vec<String>>, HashMap<Vec<String>, Vec<&DMatrix<f64>>>, HashMap<Vec<String>, Vec<&DVector<f64>>>, HashMap<Vec<String>, Vec<&DMatrix<f64>>>), Box<dyn Error>>
    {
        let mut keys : HashSet<Vec<String>> = HashSet::new();
        let mut x_split : HashMap<Vec<String>, Vec<&DMatrix<f64>>> = HashMap::new();
        let mut wgt_split : HashMap<Vec<String>, Vec<&DVector<f64>>> = HashMap::new();
        let mut repwgt_split : HashMap<Vec<String>, Vec<&DMatrix<f64>>> = HashMap::new();

        keys.insert(vec!["overall".to_string()]);

        let mut x : Vec<&DMatrix<f64>> = Vec::new();
        for mat in self.x.as_ref().unwrap().deref() {
            x.push(mat);
        }
        let ncases = x[0].nrows();

        x_split.insert(vec!["overall".to_string()], x);

        if ncases != self.wgt.as_ref().unwrap().nrows() {
            return Err(Box::new(InconsistencyError::new("unequal number of rows for data and weights")))
        }
        wgt_split.insert(vec!["overall".to_string()], vec![self.wgt.as_ref().unwrap().deref()]);

        if ncases != self.repwgts.as_ref().unwrap().nrows() {
            return Err(Box::new(InconsistencyError::new("unequal number of rows for data and replicate weights")))
        }
        repwgt_split.insert(vec!["overall".to_string()], vec![self.repwgts.as_ref().unwrap().deref()]);

        Ok((keys, x_split, wgt_split, repwgt_split))
    }

    fn prepare_for_calculate_group_by(&self)
        -> Result<(HashSet<Vec<String>>, HashMap<Vec<String>, Vec<DMatrix<f64>>>, HashMap<Vec<String>, Vec<DVector<f64>>>, HashMap<Vec<String>, Vec<DMatrix<f64>>>), Box<dyn Error>>
    {
        let mut keys : HashSet<Vec<String>> = HashSet::new();
        let mut x_split : HashMap<Vec<String>, Vec<DMatrix<f64>>> = HashMap::new();
        let mut wgt_split : HashMap<Vec<String>, Vec<DVector<f64>>> = HashMap::new();
        let mut repwgt_split : HashMap<Vec<String>, Vec<DMatrix<f64>>> = HashMap::new();

        let groups = self.groups.as_ref().unwrap().deref();

        if groups.len() > 1 && groups.len() != self.x.as_ref().unwrap().deref().len() {
            return Err(Box::new(InconsistencyError::new("number of data sets does not match number of sets with grouping columns")))
        }

        let multiple_imputation_groups = groups.len() > 1;

        let unique_combinations = groups.first().unwrap().get_keys();
        for combination in unique_combinations {
            keys.insert(combination);
        }

        for (i, mat) in self.x.as_ref().unwrap().deref().iter().enumerate() {
            let mat_split = mat.split_by(if multiple_imputation_groups { &groups[i] }  else { &groups[0] });

            match i {
                0 => {
                    for (key, mat0) in mat_split {
                        x_split.insert(key, vec![mat0]);
                    }
                }
                _ => {
                    for (key, mat0) in mat_split {
                        x_split.get_mut(&key).unwrap().push(mat0);
                    }
                }
            }
        }

        for (i, groups0) in groups.iter().enumerate() {
            let vec_split = self.wgt.as_ref().unwrap().deref().split_by(groups0);
            let mat_split = self.repwgts.as_ref().unwrap().deref().split_by(groups0);

            match i {
                0 => {
                    for (key, vec0) in vec_split {
                        wgt_split.insert(key, vec![vec0]);
                    }
                    for (key, mat0) in mat_split {
                        repwgt_split.insert(key, vec![mat0]);
                    }
                }
                _ => {
                    for (key, vec0) in vec_split {
                        wgt_split.get_mut(&key).unwrap().push(vec0);
                    }
                    for (key, mat0) in mat_split {
                        repwgt_split.get_mut(&key).unwrap().push(mat0);
                    }
                }
            }
        }

        Ok((keys, x_split, wgt_split, repwgt_split))
    }

    pub fn calculate(&mut self) -> Result<HashMap<Vec<String>, ReplicatedEstimates>, Box<dyn Error>> {
        if self.estimate.is_none() {
            return Err(Box::new(MissingElementError::new("estimate")))
        }

        self.prepare_missing_weights()?;

        let keys : HashSet<Vec<String>>;

        let x_storage : HashMap<Vec<String>, Vec<DMatrix<f64>>>;
        let wgt_storage : HashMap<Vec<String>, Vec<DVector<f64>>>;
        let repwgt_storage : HashMap<Vec<String>, Vec<DMatrix<f64>>>;

        let mut x_split : HashMap<Vec<String>, Vec<&DMatrix<f64>>>;
        let mut wgt_split : HashMap<Vec<String>, Vec<&DVector<f64>>>;
        let mut repwgt_split : HashMap<Vec<String>, Vec<&DMatrix<f64>>>;

        match self.groups {
            Some(ref groups) if groups.deref().len() > 0 => {
                (keys, x_storage, wgt_storage, repwgt_storage) = self.prepare_for_calculate_group_by()?;

                x_split = HashMap::new();
                for (key, data) in x_storage.iter() {
                    let x : Vec<&DMatrix<f64>> = data.iter().map(|mat| mat).collect();
                    x_split.insert(key.clone(), x);
                }

                wgt_split = HashMap::new();
                for (key, data) in wgt_storage.iter() {
                    let wgt : Vec<&DVector<f64>> = data.iter().map(|wgt| wgt).collect();
                    wgt_split.insert(key.clone(), wgt);
                }

                repwgt_split = HashMap::new();
                for (key, data) in repwgt_storage.iter() {
                    let repwgt : Vec<&DMatrix<f64>> = data.iter().map(|repwgt| repwgt).collect();
                    repwgt_split.insert(key.clone(), repwgt);
                }
            }
            _ => {
                (keys, x_split, wgt_split, repwgt_split) = self.prepare_for_calculate_overall()?
            }
        }

        let mut results : HashMap<Vec<String>, ReplicatedEstimates> = HashMap::new();

        for key in keys {
            let result = replicate_estimates(
                self.estimate.as_ref().unwrap().clone(),
                x_split.get(&key).unwrap(),
                wgt_split.get(&key).unwrap(),
                repwgt_split.get(&key).unwrap(),
                self.variance_adjustment_factor,
            );

            results.insert(key, result);
        }

        Ok(results)
    }

    pub fn summary(&self) -> String {
        let estimate_name = self.estimate_name.as_ref().unwrap_or(&"none".to_string()).clone();

        let group_info = match self.groups.as_ref() {
            None => { "".to_string() }
            Some(groups) => {
                let group_data = groups.as_ref();
                format!(" by {} grouping columns", group_data[0].ncols())
            }
        };

        let data_info = if self.x.is_none() {
            "no data".to_string()
        } else {
            let data = self.x.as_ref().unwrap().deref();
            format!("{} datasets with {} cases", data.len(), data[0].nrows())
        };

        let wgt_info = if self.wgt.is_none() {
            "wgt missing".to_string()
        } else {
            let wgts = self.wgt.as_ref().unwrap().deref();
            format!("{} weights of sum {}", wgts.len(), wgts.sum())
        };

        let factor_info = if self.variance_adjustment_factor == 1.0 {
            "".to_string()
        } else {
            format!(", factor {}", self.variance_adjustment_factor)
        };

        let repwgt_info = if self.repwgts.is_none() {
            "no replicate weights".to_string()
        } else {
            let repwgts = self.repwgts.as_ref().unwrap().deref();
            format!("{} replicate weights{}", repwgts.ncols(), factor_info)
        };

        estimate_name + &group_info +  " (" + &data_info + "; " + &wgt_info + "; " + &repwgt_info + ")"
    }

    pub fn copy(&self) -> Analysis {
        Analysis {
            x: self.x.clone(),
            wgt: self.wgt.clone(),
            repwgts: self.repwgts.clone(),
            variance_adjustment_factor: self.variance_adjustment_factor,
            estimate_name: self.estimate_name.clone(),
            estimate: match &self.estimate {
                None => None,
                Some(estimate) => Some(Arc::clone(estimate)),
            },
            groups: self.groups.clone(),
            options: self.options.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector};
    use crate::analysis::*;
    use crate::assert_approx_eq_iter_f64;
    use crate::estimates::QuantileType;

    #[test]
    fn test_for_data() {
        let data1 = dmatrix![
            537.0, 456.2, 501.7;
            499.1, 433.2, 500.6;
            611.0, 501.9, 588.2;
        ];

        let mut analysis1 = analysis();
        analysis1.for_data(Imputation::No(&data1));

        assert_eq!("none (1 datasets with 3 cases; wgt missing; no replicate weights)", analysis1.summary());

        let data2imp1 = dmatrix![
            537.0, 456.2, 501.7;
            499.1, 433.2, 500.6;
            611.0, 501.9, 588.2;
            534.7, 479.9, 512.4;
        ];
        let data2imp2 = dmatrix![
            538.0, 456.2, 501.7;
            490.1, 433.2, 500.6;
            612.0, 501.9, 588.2;
            533.7, 479.9, 512.4;
        ];
        let data2imp3 = dmatrix![
            537.0, 457.2, 501.7;
            499.1, 432.2, 500.6;
            611.0, 503.9, 588.2;
            534.7, 475.9, 512.4;
        ];
        let mut data2 : Vec<&DMatrix<f64>> = Vec::new();
        data2.push(&data2imp1);
        data2.push(&data2imp2);
        data2.push(&data2imp3);

        let mut analysis2 = analysis();
        analysis2.for_data(Imputation::Yes(&data2));

        assert_eq!("none (3 datasets with 4 cases; wgt missing; no replicate weights)", analysis2.summary());
    }

    #[test]
    fn test_group_by() {
        let groups1 = dmatrix![
            1.0, 1.0;
            2.0, 1.0;
            1.0, 2.0;
        ];

        let mut analysis1 = analysis();
        analysis1.group_by(Imputation::No(&groups1));

        assert_eq!("none by 2 grouping columns (no data; wgt missing; no replicate weights)", analysis1.summary());

        let groups2imp1 = dmatrix![ 1.0; 1.0; 2.0; 2.0; ];
        let groups2imp2 = dmatrix![ 1.0; 1.0; 1.0; 2.0; ];
        let groups2imp3 = dmatrix![ 1.0; 1.0; 2.0; 2.0; ];
        let mut groups2: Vec<&DMatrix<f64>> = Vec::new();
        groups2.push(&groups2imp1);
        groups2.push(&groups2imp2);
        groups2.push(&groups2imp3);

        let mut analysis2 = analysis();
        analysis2.group_by(Imputation::Yes(&groups2));

        assert_eq!("none by 1 grouping columns (no data; wgt missing; no replicate weights)", analysis2.summary());
    }

    #[test]
    fn test_calculate_does_not_work_without_data() {
        let mut analysis1 = analysis();
        let result = analysis1.mean().calculate();

        assert!(result.is_err());
        assert_eq!("Analysis is missing some element: data", result.err().unwrap().deref().to_string());

        let mut analysis1 = analysis();
        let result = analysis1.mean().for_data(Imputation::Yes(&Vec::<&DMatrix<f64>>::new())).calculate();

        assert!(result.is_err());
        assert_eq!("Analysis is missing some element: data", result.err().unwrap().deref().to_string());
    }

    #[test]
    fn test_calculate_does_not_work_without_estimate() {
        let data = dmatrix![
            537.0, 456.2, 501.7;
            499.1, 433.2, 500.6;
            611.0, 501.9, 588.2;
        ];

        let mut analysis1 = analysis();
        let result = analysis1.for_data(Imputation::No(&data)).calculate();

        assert!(result.is_err());
        assert_eq!("Analysis is missing some element: estimate", result.err().unwrap().deref().to_string());
    }

    #[test]
    fn test_calculate_does_not_work_with_unequal_rows_between_data_and_weights() {
        let data = dmatrix![
            537.0, 456.2, 501.7;
            499.1, 433.2, 500.6;
            611.0, 501.9, 588.2;
        ];

        let wgt = dvector![1.0, 2.0];

        let mut analysis1 = analysis();
        let result = analysis1.for_data(Imputation::No(&data)).set_weights(&wgt).mean().calculate();

        assert!(result.is_err());
        assert_eq!("Inconsistency in analysis: unequal number of rows for data and weights", result.err().unwrap().deref().to_string());
    }

    #[test]
    fn test_calculate_works_without_weights() {
        let data = dmatrix![
            537.0, 456.2, 501.7;
            499.1, 433.2, 502.9;
            611.0, 501.9, 589.3;
        ];

        let mut analysis1 = analysis();
        let result = analysis1.for_data(Imputation::No(&data)).mean().calculate();

        assert!(result.is_ok());
        let result = result.unwrap();

        assert_eq!(1, result.len());
        assert_eq!(3, result[&vec!["overall".to_string()]].final_estimates().len());
        assert_eq!(531.3, result[&vec!["overall".to_string()]].final_estimates()[2]);
        assert_eq!(0.0, result[&vec!["overall".to_string()]].standard_errors()[1]);
    }

    #[test]
    fn test_calculate_works_for_mean_without_resampling() {
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

        let mut analysis1 = analysis();
        let result = analysis1.for_data(Imputation::Yes(&imp_data)).set_weights(&wgt).mean().calculate();

        assert!(result.is_ok());
        let result = result.unwrap();

        assert_eq!(1, result.len());
        let first_result = result[&vec!["overall".to_string()]].clone();

        assert_eq!(4, first_result.parameter_names().len());
        assert_eq!("mean_x2", first_result.parameter_names()[1]);
        assert_approx_eq_iter_f64!(first_result.final_estimates(), dvector![2.25, 3.125, 2.0, -2.5]);
        assert_approx_eq_iter_f64!(first_result.sampling_variances(), dvector![0.0, 0.0, 0.0, 0.0]);
        assert_approx_eq_iter_f64!(first_result.imputation_variances(), dvector![0.0069444444444443955, 0.0, 0.0002777777777777758, 0.0]);
    }

    #[test]
    fn test_calculate_works_for_mean_with_resampling() {
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
        let rep_wgts = DMatrix::from_row_slice(3, 6, &[
            0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
            0.5, 0.0, 0.5, 0.5, 0.0, 0.5,
            1.5, 1.5, 0.0, 1.5, 1.5, 0.0,
        ]);

        let mut analysis = analysis();
        let result =
            analysis
                .for_data(Imputation::Yes(&imp_data))
                .set_weights(&wgt)
                .with_replicate_weights(&rep_wgts)
                .set_variance_adjustment_factor(0.5)
                .mean()
                .calculate();

        assert!(result.is_ok());
        let result = result.unwrap();

        assert_eq!(1, result.len());
        let first_result = result[&vec!["overall".to_string()]].clone();

        assert_eq!(4, first_result.parameter_names().len());
        assert_eq!("mean_x2", first_result.parameter_names()[1]);
        assert_approx_eq_iter_f64!(first_result.final_estimates(), dvector![2.25, 3.125, 2.0, -2.5]);
        assert_approx_eq_iter_f64!(first_result.sampling_variances(), dvector![1.000486111111111, 0.28265624999999994, 1.2229166666666667, 1.5625]);
        assert_approx_eq_iter_f64!(first_result.imputation_variances(), dvector![0.0069444444444443955, 0.0, 0.0002777777777777758, 0.0]);
        assert_approx_eq_iter_f64!(first_result.standard_errors(), dvector![1.0048608711510119, 0.5316542579534184, 1.1060230725608924, 1.25]);
    }

    #[test]
    fn test_calculate_works_for_mean_with_groups_same() {
        let mut imp_data: Vec<&DMatrix<f64>> = Vec::new();
        let data0 = DMatrix::from_row_slice(6, 4, &[
            1.0, 4.0, 2.5, -1.0,
            2.5, 1.75, 4.0, -2.5,
            3.0, 3.0, 1.0, -3.5,
            1.0, 4.0, 2.5, -1.0,
            2.5, 1.75, 4.0, -2.5,
            3.0, 3.0, 1.0, -3.5,
        ]);
        imp_data.push(&data0);
        let data1 = DMatrix::from_row_slice(6, 4, &[
            1.2, 4.0, 2.5, -1.0,
            2.5, 1.75, 3.9, -2.5,
            2.7, 3.0, 1.0, -3.5,
            1.2, 4.0, 2.5, -1.0,
            2.5, 1.75, 3.9, -2.5,
            2.7, 3.0, 1.0, -3.5,
        ]);
        imp_data.push(&data1);
        let data2 = DMatrix::from_row_slice(6, 4, &[
            0.8, 4.0, 2.5, -1.0,
            2.5, 1.75, 4.1, -2.5,
            3.3, 3.0, 1.0, -3.5,
            0.8, 4.0, 2.5, -1.0,
            2.5, 1.75, 4.1, -2.5,
            3.3, 3.0, 1.0, -3.5,
        ]);
        imp_data.push(&data2);

        let wgt = dvector![1.0, 0.5, 1.5, 1.0, 0.5, 1.5];
        let rep_wgts = DMatrix::from_row_slice(6, 6, &[
            0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
            0.5, 0.0, 0.5, 0.5, 0.0, 0.5,
            1.5, 1.5, 0.0, 1.5, 1.5, 0.0,
            0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
            0.5, 0.0, 0.5, 0.5, 0.0, 0.5,
            1.5, 1.5, 0.0, 1.5, 1.5, 0.0,
        ]);

        let groups = DMatrix::from_row_slice(6, 1, &[
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
        ]);

        let mut analysis = analysis();
        let result =
            analysis
                .for_data(Imputation::Yes(&imp_data))
                .set_weights(&wgt)
                .with_replicate_weights(&rep_wgts)
                .set_variance_adjustment_factor(0.5)
                .mean()
                .group_by(Imputation::No(&groups))
                .calculate();

        assert!(result.is_ok());
        let result = result.unwrap();

        assert_eq!(2, result.len());

        let first_result = result[&vec!["1".to_string()]].clone();
        assert_approx_eq_iter_f64!(first_result.final_estimates(), dvector![2.25, 3.125, 2.0, -2.5]);
        assert_approx_eq_iter_f64!(first_result.sampling_variances(), dvector![1.000486111111111, 0.28265624999999994, 1.2229166666666667, 1.5625]);
        assert_approx_eq_iter_f64!(first_result.imputation_variances(), dvector![0.0069444444444443955, 0.0, 0.0002777777777777758, 0.0]);
        assert_approx_eq_iter_f64!(first_result.standard_errors(), dvector![1.0048608711510119, 0.5316542579534184, 1.1060230725608924, 1.25]);

        let second_result = result[&vec!["2".to_string()]].clone();
        assert_approx_eq_iter_f64!(second_result.final_estimates(), dvector![2.25, 3.125, 2.0, -2.5]);
        assert_approx_eq_iter_f64!(second_result.sampling_variances(), dvector![1.000486111111111, 0.28265624999999994, 1.2229166666666667, 1.5625]);
        assert_approx_eq_iter_f64!(second_result.imputation_variances(), dvector![0.0069444444444443955, 0.0, 0.0002777777777777758, 0.0]);
        assert_approx_eq_iter_f64!(second_result.standard_errors(), dvector![1.0048608711510119, 0.5316542579534184, 1.1060230725608924, 1.25]);
    }

    #[test]
    fn test_calculate_works_for_mean_with_groups_different() {
        let mut imp_data: Vec<&DMatrix<f64>> = Vec::new();
        let data0 = DMatrix::from_row_slice(10, 1, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let data1 = DMatrix::from_row_slice(10, 1, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let data2 = DMatrix::from_row_slice(10, 1, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let data3 = DMatrix::from_row_slice(10, 1, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        imp_data.push(&data0);
        imp_data.push(&data1);
        imp_data.push(&data2);
        imp_data.push(&data3);

        let mut imp_groups: Vec<&DMatrix<f64>> = Vec::new();
        let group0 = DMatrix::from_row_slice(10, 1, &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let group1 = DMatrix::from_row_slice(10, 1, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let group2 = DMatrix::from_row_slice(10, 1, &[1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let group3 = DMatrix::from_row_slice(10, 1, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        imp_groups.push(&group0);
        imp_groups.push(&group1);
        imp_groups.push(&group2);
        imp_groups.push(&group3);

        let wgts = dvector![1.0, 1.0, 1.25, 1.25, 1.5, 1.5, 1.75, 1.75, 2.0, 2.0];

        let rep_wgts = DMatrix::from_row_slice(10, 5, &[
            2.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 1.0, 1.0, 1.0, 1.0,
            1.25, 2.5, 1.25, 1.25, 1.25,
            1.25, 0.0, 1.25, 1.25, 1.25,
            1.5, 1.5, 3.0, 1.5, 1.5,
            1.5, 1.5, 0.0, 1.5, 1.5,
            1.75, 1.75, 1.75, 3.5, 1.75,
            1.75, 1.75, 1.75, 0.0, 1.75,
            2.0, 2.0, 2.0, 2.0, 4.0,
            2.0, 2.0, 2.0, 2.0, 0.0
        ]);

        let mut analysis = analysis();
        analysis
            .mean()
            .for_data(Imputation::Yes(&imp_data))
            .set_weights(&wgts)
            .with_replicate_weights(&rep_wgts)
            .group_by(Imputation::Yes(&imp_groups));

        let result = analysis.calculate();

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(2, result.len());

        let first_result = result[&vec!["0".to_string()]].clone();
        assert_approx_eq_iter_f64!(first_result.final_estimates(), dvector![6.523069], 1e-6);
        assert_approx_eq_iter_f64!(first_result.sampling_variances(), dvector![2.139808], 1e-6);
        assert_approx_eq_iter_f64!(first_result.imputation_variances(), dvector![0.2843781], 1e-6);
        assert_approx_eq_iter_f64!(first_result.standard_errors(), dvector![1.579646], 1e-6);

        let second_result = result[&vec!["1".to_string()]].clone();
        assert_approx_eq_iter_f64!(second_result.final_estimates(), dvector![5.928963], 1e-6);
        assert_approx_eq_iter_f64!(second_result.sampling_variances(), dvector![1.156444], 1e-6);
        assert_approx_eq_iter_f64!(second_result.imputation_variances(), dvector![0.2514576], 1e-6);
        assert_approx_eq_iter_f64!(second_result.standard_errors(), dvector![1.212752], 1e-6);
    }

    #[test]
    fn test_quantiles_setting() {
        let data = DMatrix::from_row_slice(10, 1, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        let mut analysis = analysis();
        analysis
            .for_data(Imputation::No(&data))
            .quantiles();

        let result_default = analysis.calculate().unwrap();
        assert_eq!(result_default[&vec!["overall".to_string()]].parameter_names(), &vec!["quantile_x1_0.25", "quantile_x1_0.5", "quantile_x1_0.75"]);
        assert_approx_eq_iter_f64!(result_default[&vec!["overall".to_string()]].final_estimates(), &dvector![2.5, 5.0, 7.5]);

        analysis
            .set_quantiles(vec![0.2, 0.4, 0.6, 0.8]);

        let result_quintiles = analysis.calculate().unwrap();
        assert_eq!(result_quintiles[&vec!["overall".to_string()]].parameter_names(), &vec!["quantile_x1_0.2", "quantile_x1_0.4", "quantile_x1_0.6", "quantile_x1_0.8"]);
        assert_approx_eq_iter_f64!(result_quintiles[&vec!["overall".to_string()]].final_estimates(), &dvector![2.0, 4.0, 6.0, 8.0]);

        analysis
            .set_quantiles(vec![0.75, 0.25])
            .set_quantile_type(QuantileType::Lower);

        let result_lower = analysis.calculate().unwrap();
        assert_eq!(result_lower[&vec!["overall".to_string()]].parameter_names(), &vec!["quantile_x1_0.25", "quantile_x1_0.75"]);
        assert_approx_eq_iter_f64!(result_lower[&vec!["overall".to_string()]].final_estimates(), &dvector![2.0, 7.0]);
    }

    #[test]
    fn test_copying() {
        let wgts = dvector![1.1, 1.5, 1.3, 1.7, 1.7, 1.0];

        let mut base_analysis = analysis();
        base_analysis.set_weights(&wgts);

        let mut analysis1 = base_analysis.copy();
        analysis1.mean();

        assert_eq!("none (no data; 6 weights of sum 8.3; no replicate weights)", base_analysis.summary());
        assert_eq!("mean (no data; 6 weights of sum 8.3; no replicate weights)", analysis1.summary());
        assert_eq!(2, Rc::strong_count(base_analysis.wgt.as_ref().unwrap()));

        let new_wgts = dvector![2.1, 2.5, 2.3, 2.7, 2.7, 2.0];
        analysis1.set_weights(&new_wgts);

        assert_eq!("none (no data; 6 weights of sum 8.3; no replicate weights)", base_analysis.summary());
        assert_eq!("mean (no data; 6 weights of sum 14.3; no replicate weights)", analysis1.summary());
        assert_eq!(1, Rc::strong_count(base_analysis.wgt.as_ref().unwrap()));

        let mut analysis2 = analysis1.copy();

        assert_eq!("mean (no data; 6 weights of sum 14.3; no replicate weights)", analysis2.summary());
        assert_eq!(2, Rc::strong_count(analysis2.wgt.as_ref().unwrap()));

        let data = dmatrix![
            537.0, 456.2, 501.7;
            499.1, 433.2, 500.6;
            611.0, 501.9, 588.2;
        ];
        analysis2.for_data(Imputation::No(&data));
        let analysis3 = analysis2.copy();

        assert_eq!("mean (1 datasets with 3 cases; 6 weights of sum 14.3; no replicate weights)", analysis3.summary());
        assert_eq!(3, Rc::strong_count(analysis2.wgt.as_ref().unwrap()));

        analysis1.set_weights(&wgts);

        assert_eq!(2, Rc::strong_count(analysis2.wgt.as_ref().unwrap()));
    }

    #[test]
    fn test_copying_allows_reproducing_analysis() {
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

        let mut analysis1 = analysis();
        analysis1.for_data(Imputation::Yes(&imp_data)).set_weights(&wgt).mean();

        let mut analysis2 = analysis1.copy();

        assert_eq!(1, analysis1.calculate().unwrap().len());
        assert_eq!(1, analysis2.calculate().unwrap().len());

        let mut analysis3 = analysis1.copy();

        assert_eq!(1, analysis3.calculate().unwrap().len());
    }
}