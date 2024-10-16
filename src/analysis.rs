use std::ops::Deref;
use std::rc::Rc;
use nalgebra::{DMatrix, DVector};
use crate::estimates;

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
    pub fn set_wgts(&mut self, wgt: &DVector<f64>) -> &mut Self {
        self.wgt = Some(Rc::new(wgt.clone()));
        self
    }

    pub fn mean(&mut self) -> &mut Self {
        self.estimate_name = Some("mean".to_string());
        self.estimate = Some(estimates::mean);
        self
    }

    pub fn summary(&self) -> String {
        let estimate_name = self.estimate_name.as_ref().unwrap_or(&"none".to_string()).clone();
        let wgt_info = if self.wgt.is_none() {
            "wgt missing".to_string()
        } else {
            let wgts = self.wgt.as_ref().unwrap().deref();
            format!("{} weights of sum {}", wgts.len(), wgts.sum())
        };

        estimate_name + " (" + &wgt_info + ")"
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
    use nalgebra::dvector;
    use crate::analysis::*;

    #[test]
    fn test_copying() {
        let wgts = dvector![1.1, 1.5, 1.3, 1.7, 1.7, 1.0];

        let mut base_analysis = analysis();
        base_analysis.set_wgts(&wgts);

        let mut analysis1 = base_analysis.copy();
        analysis1.mean();

        assert_eq!("none (6 weights of sum 8.3)", base_analysis.summary());
        assert_eq!("mean (6 weights of sum 8.3)", analysis1.summary());
        assert_eq!(2, Rc::strong_count(base_analysis.wgt.as_ref().unwrap()));

        let new_wgts = dvector![2.1, 2.5, 2.3, 2.7, 2.7, 2.0];
        analysis1.set_wgts(&new_wgts);

        assert_eq!("none (6 weights of sum 8.3)", base_analysis.summary());
        assert_eq!("mean (6 weights of sum 14.3)", analysis1.summary());
        assert_eq!(1, Rc::strong_count(base_analysis.wgt.as_ref().unwrap()));
    }
}