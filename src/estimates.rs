use nalgebra::{DMatrix, DVector};

pub fn mean(x: DMatrix<f64>, wgt: DVector<f64>) -> DVector<f64> {
    let wgt_sums = x * &wgt;
    wgt_sums / wgt.len() as f64
}

#[cfg(test)]
mod tests {
    use nalgebra::dvector;
    use super::*;

    #[test]
    fn test_mean() {
        let data = DMatrix::from_row_slice(3, 3, &[
            1.0, 4.0, 2.5,
            2.5, 1.75, 4.0,
            3.0, 3.0, 1.0,
        ]);

        let wgt = dvector![1.0, 0.5, 1.5];

        let result = mean(data, wgt);
        assert_eq!(result, dvector![2.25, 3.125, 2.0]);
    }
}
