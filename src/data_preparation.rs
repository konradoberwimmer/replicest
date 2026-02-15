use nalgebra::{DMatrix, DVector};

pub fn listwise_delete(x: &mut DMatrix<f64>, weight: &mut DVector<f64>, repweights: &mut DMatrix<f64>) {
    assert_eq!(x.nrows(), weight.nrows());

    let has_replicate_weights = repweights.nrows() > 0;

    if has_replicate_weights {
        assert_eq!(x.nrows(), repweights.nrows());
    }

    for rr in 0..x.nrows() {
        if x.row(rr).iter().any(|v| v.is_nan()) {
            x.row_mut(rr).fill(0.0);
            weight[rr] = 0.0;

            if has_replicate_weights {
                repweights.row_mut(rr).fill(0.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_listwise_delete_nothing_to_do() {
        let mut x = DMatrix::<f64>::from_element(10, 10, 1.0);
        let mut weight = DVector::<f64>::from_element(10, 1.0);
        let mut repweights = DMatrix::<f64>::from_element(10, 10, 1.0);

        listwise_delete(&mut x, &mut weight, &mut repweights);

        assert!(x.iter().all(|v| *v == 1.0));
        assert!(weight.iter().all(|v| *v == 1.0));
        assert!(repweights.iter().all(|v| *v == 1.0));
    }

    #[test]
    fn test_listwise_delete() {
        let mut x = DMatrix::<f64>::from_element(10, 10, 1.0);
        x.row_mut(2)[3] = f64::NAN;
        let mut weight = DVector::<f64>::from_element(10, 1.0);
        let mut repweights = DMatrix::<f64>::from_element(10, 10, 1.0);

        listwise_delete(&mut x, &mut weight, &mut repweights);

        assert!(!x.iter().all(|v| *v == 1.0));
        assert!(x.row(2).iter().all(|v| *v == 0.0));
        assert!(!weight.iter().all(|v| *v == 1.0));
        assert_eq!(weight[2], 0.0);
        assert!(!repweights.iter().all(|v| *v == 1.0));
        assert!(repweights.row(2).iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_listwise_delete_without_replicate_weights() {
        let mut x = DMatrix::<f64>::from_element(10, 10, 1.0);
        x.row_mut(2)[3] = f64::NAN;
        let mut weight = DVector::<f64>::from_element(10, 1.0);
        let mut repweights = DMatrix::<f64>::zeros(0, 0);

        listwise_delete(&mut x, &mut weight, &mut repweights);

        assert!(!x.iter().all(|v| *v == 1.0));
        assert!(x.row(2).iter().all(|v| *v == 0.0));
        assert!(!weight.iter().all(|v| *v == 1.0));
        assert_eq!(weight[2], 0.0);
    }
}