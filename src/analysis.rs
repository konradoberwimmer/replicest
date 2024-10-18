use std::collections::HashMap;
use std::ops::Deref;
use std::rc::Rc;
use nalgebra::{DMatrix, DVector};
use crate::estimates;
use crate::replication::{replicate_estimates, ReplicatedEstimates};

pub enum Imputation<'a> {
    Yes(&'a Vec<&'a DMatrix<f64>>),
    No(&'a DMatrix<f64>),
}

pub struct Analysis {
    x: Option<Rc<Vec<DMatrix<f64>>>>,
    wgt: Option<Rc<DVector<f64>>>,
    estimate_name: Option<String>,
    estimate: Option<fn(&DMatrix<f64>, &DVector<f64>) -> estimates::Estimates>,
}

pub fn analysis() -> Analysis {
    Analysis {
        x: None,
        wgt: None,
        estimate_name: None,
        estimate: None,
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

    pub fn set_wgts(&mut self, wgt: &DVector<f64>) -> &mut Self {
        self.wgt = Some(Rc::new(wgt.clone()));
        self
    }

    pub fn mean(&mut self) -> &mut Self {
        self.estimate_name = Some("mean".to_string());
        self.estimate = Some(estimates::mean);
        self
    }

    pub fn calculate(&mut self) -> HashMap<Vec<String>, ReplicatedEstimates> {
        assert!(self.x.is_some(), "no data provided for analysis");
        assert!(self.x.as_ref().unwrap().deref().len() > 0, "no data provided for analysis");
        assert!(self.estimate.is_some(), "no estimate set for analysis");

        let mut x : Vec<&DMatrix<f64>> = Vec::new();
        for mat in self.x.as_ref().unwrap().deref() {
            x.push(mat);
        }

        let ncases = x[0].nrows();

        if self.wgt.is_some() {
            assert_eq!(ncases, self.wgt.as_ref().unwrap().nrows(),
                       "unequal number of rows for data and weights");
        } else {
            self.wgt = Some(Rc::new(DVector::<f64>::from_element(ncases, 1.0)));
        };

        let mut results : HashMap<Vec<String>, ReplicatedEstimates> = HashMap::new();

        let result = replicate_estimates(
            self.estimate.as_ref().unwrap().clone(),
            &x,
            self.wgt.as_ref().unwrap().deref(),
            &DMatrix::from_row_slice(ncases, 0, &[]),
            1.0
        );

        let mut key : Vec<String> = Vec::new();
        key.push("overall".to_string());

        results.insert(key, result);

        results
    }

    pub fn summary(&self) -> String {
        let estimate_name = self.estimate_name.as_ref().unwrap_or(&"none".to_string()).clone();

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

        estimate_name + " (" + &data_info + "; " + &wgt_info + ")"
    }

    pub fn copy(&self) -> Analysis {
        Analysis {
            x: self.x.clone(),
            wgt: self.wgt.clone(),
            estimate_name: self.estimate_name.clone(),
            estimate: self.estimate.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector};
    use crate::analysis::*;

    #[test]
    fn test_for_data() {
        let data1 = dmatrix![
            537.0, 456.2, 501.7;
            499.1, 433.2, 500.6;
            611.0, 501.9, 588.2;
        ];

        let mut analysis1 = analysis();
        analysis1.for_data(Imputation::No(&data1));

        assert_eq!("none (1 datasets with 3 cases; wgt missing)", analysis1.summary());

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

        assert_eq!("none (3 datasets with 4 cases; wgt missing)", analysis2.summary());
        assert_eq!(3, data1.nrows());
        assert_eq!(3, data2.len());
        assert_eq!(4, data2[0].nrows());
    }

    #[test]
    #[should_panic(expected = "no data provided for analysis")]
    fn test_calculate_does_not_work_without_data() {
        let mut analysis1 = analysis();
        analysis1.calculate();
    }

    #[test]
    #[should_panic(expected = "no data provided for analysis")]
    fn test_calculate_does_not_work_with_zero_data() {
        let mut analysis1 = analysis();
        analysis1.for_data(Imputation::Yes(&Vec::<&DMatrix<f64>>::new())).calculate();
    }

    #[test]
    #[should_panic(expected = "no estimate set for analysis")]
    fn test_calculate_does_not_work_without_estimate() {
        let data = dmatrix![
            537.0, 456.2, 501.7;
            499.1, 433.2, 500.6;
            611.0, 501.9, 588.2;
        ];

        let mut analysis1 = analysis();
        analysis1.for_data(Imputation::No(&data)).calculate();
    }

    #[test]
    #[should_panic(expected = "unequal number of rows for data and weights")]
    fn test_calculate_does_not_work_with_unequal_rows_between_data_and_weights() {
        let data = dmatrix![
            537.0, 456.2, 501.7;
            499.1, 433.2, 500.6;
            611.0, 501.9, 588.2;
        ];

        let wgt = dvector![1.0, 2.0];

        let mut analysis1 = analysis();
        analysis1.for_data(Imputation::No(&data)).set_wgts(&wgt).mean().calculate();
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

        assert_eq!(1, result.len());
        assert_eq!(3, result[&vec!["overall".to_string()]].final_estimates().len());
        assert_eq!(531.3, result[&vec!["overall".to_string()]].final_estimates()[2]);
        assert_eq!(0.0, result[&vec!["overall".to_string()]].standard_errors()[1]);
    }

    #[test]
    fn test_calculate_works_for_mean() {
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
        let result = analysis1.for_data(Imputation::Yes(&imp_data)).set_wgts(&wgt).mean().calculate();

        assert_eq!(1, result.len());
        let first_result = result[&vec!["overall".to_string()]].clone();

        assert_eq!(4, first_result.parameter_names().len());
        assert_eq!("mean_x2", first_result.parameter_names()[1]);
        assert_eq!(0, (first_result.final_estimates() - dvector![2.25, 3.125, 2.0, -2.5]).iter().filter(|&&v| v.abs() > 1e-10).count());
        assert_eq!(0, (first_result.sampling_variances() - dvector![0.0, 0.0, 0.0, 0.0]).iter().filter(|&&v| v.abs() > 1e-10).count());
        assert_eq!(0, (first_result.imputation_variances() - dvector![0.0069444444444443955, 0.0, 0.0002777777777777758, 0.0]).iter().filter(|&&v| v.abs() > 1e-10).count());
    }

    #[test]
    fn test_copying() {
        let wgts = dvector![1.1, 1.5, 1.3, 1.7, 1.7, 1.0];

        let mut base_analysis = analysis();
        base_analysis.set_wgts(&wgts);

        let mut analysis1 = base_analysis.copy();
        analysis1.mean();

        assert_eq!("none (no data; 6 weights of sum 8.3)", base_analysis.summary());
        assert_eq!("mean (no data; 6 weights of sum 8.3)", analysis1.summary());
        assert_eq!(2, Rc::strong_count(base_analysis.wgt.as_ref().unwrap()));

        let new_wgts = dvector![2.1, 2.5, 2.3, 2.7, 2.7, 2.0];
        analysis1.set_wgts(&new_wgts);

        assert_eq!("none (no data; 6 weights of sum 8.3)", base_analysis.summary());
        assert_eq!("mean (no data; 6 weights of sum 14.3)", analysis1.summary());
        assert_eq!(1, Rc::strong_count(base_analysis.wgt.as_ref().unwrap()));

        let mut analysis2 = analysis1.copy();

        assert_eq!("mean (no data; 6 weights of sum 14.3)", analysis2.summary());
        assert_eq!(2, Rc::strong_count(analysis2.wgt.as_ref().unwrap()));

        let data = dmatrix![
            537.0, 456.2, 501.7;
            499.1, 433.2, 500.6;
            611.0, 501.9, 588.2;
        ];
        analysis2.for_data(Imputation::No(&data));
        let analysis3 = analysis2.copy();

        assert_eq!("mean (1 datasets with 3 cases; 6 weights of sum 14.3)", analysis3.summary());
        assert_eq!(3, Rc::strong_count(analysis2.wgt.as_ref().unwrap()));

        analysis1.set_wgts(&wgts);

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
        analysis1.for_data(Imputation::Yes(&imp_data)).set_wgts(&wgt).mean();

        let mut analysis2 = analysis1.copy();

        assert_eq!(1, analysis1.calculate().len());
        assert_eq!(1, analysis2.calculate().len());

        let mut analysis3 = analysis1.copy();

        assert_eq!(1, analysis3.calculate().len());
    }
}