use std::collections::HashMap;
use nalgebra::{DMatrix, DVector, Dim, Matrix, RawStorage};

pub trait ExtractValues {
    fn extract_lower_triangle(&self) -> DVector<f64>;
}

impl<R: Dim, C: Dim, S: RawStorage<f64, R, C>> ExtractValues for Matrix<f64, R, C, S> {
    fn extract_lower_triangle(&self) -> DVector<f64> {
        assert_eq!(self.nrows(), self.ncols(), "non-square matrix for extract_lower_triangle");

        DVector::<f64>::from_iterator(
            self.nrows() * (self.nrows() + 1) / 2,
            self.iter().enumerate()
                .filter(|(i, _)| i/self.nrows() <= i%self.nrows())
                .map(|(_, v)| v.clone())
        )
    }
}

pub trait Split {
    fn split_by(&self, other: &DMatrix<f64>) -> HashMap<Vec<String>, DMatrix<f64>>;
}

impl Split for DMatrix<f64> {
    fn split_by(&self, other: &DMatrix<f64>) -> HashMap<Vec<String>, DMatrix<f64>> {
        assert_eq!(self.nrows(), other.nrows(), "unequal number of rows in split_by");

        let mut index_map : HashMap<Vec<String>, Vec<usize>> = HashMap::new();

        for (r, row) in other.row_iter().enumerate() {
            let key : Vec<String> = row.iter().map(|v| v.to_string()).collect();

            let mut index_vector = if index_map.contains_key(&key) {
                index_map[&key].clone()
            } else {
                Vec::<usize>::new()
            };

            index_vector.push(r);
            index_map.insert(key, index_vector);
        }

        let mut hash_map : HashMap<Vec<String>, DMatrix<f64>> = HashMap::new();

        for entry in index_map.into_iter() {
            let mut matrix = DMatrix::<f64>::zeros(entry.1.len(), self.ncols());

            for (r_new, r_old) in entry.1.into_iter().enumerate() {
                matrix.set_row(r_new, &self.row(r_old));
            }

            hash_map.insert(entry.0.clone(), matrix);
        }

        hash_map
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector};
    use super::*;

    #[test]
    fn test_extract_lower_triangle() {
        let matrix = dmatrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0;
        ];

        assert_eq!(matrix.extract_lower_triangle(), dvector![1.0, 4.0, 7.0, 5.0, 8.0, 9.0])
    }

    #[test]
    #[should_panic(expected = "non-square matrix for extract_lower_triangle")]
    fn test_extract_lower_triangle_panic_non_square() {
        let matrix = dmatrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
        ];

        matrix.extract_lower_triangle();
    }

    #[test]
    fn test_split_by_single_column() {
        let data = dmatrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0;
            10.0, 11.0, 12.0;
            13.0, 14.0, 15.0;
        ];

        let split_vector = dmatrix![1.0; 1.0; 2.0; 2.0; 1.0];

        let result = data.split_by(&split_vector);
        assert_eq!(2, result.len());
        assert_eq!(3, result[&vec!["1".to_string()]].nrows());
        assert_eq!(5.0, result[&vec!["1".to_string()]][(1,1)]);
        assert_eq!(15.0, result[&vec!["1".to_string()]][(2,2)]);
        assert_eq!(2, result[&vec!["2".to_string()]].nrows());
        assert_eq!(9.0, result[&vec!["2".to_string()]][(0,2)]);
    }

    #[test]
    fn test_split_by_two_columns() {
        let data = dmatrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0;
            10.0, 11.0, 12.0;
            13.0, 14.0, 15.0;
        ];

        let split_vector = dmatrix![
            1.0, 1.0;
            1.0, 2.0;
            2.0, 1.0;
            2.0, 2.0;
            1.0, 1.0;
        ];

        let result = data.split_by(&split_vector);
        assert_eq!(4, result.len());
        assert_eq!(2, result[&vec!["1".to_string(), "1".to_string()]].nrows());
        assert_eq!(14.0, result[&vec!["1".to_string(), "1".to_string()]][(1,1)]);
        assert_eq!(1, result[&vec!["2".to_string(), "2".to_string()]].nrows());
        assert_eq!(12.0, result[&vec!["2".to_string(), "2".to_string()]][(0,2)]);
    }

    #[test]
    #[should_panic(expected = "unequal number of rows in split_by")]
    fn test_split_by_unequal_number_of_rows() {
        let data = dmatrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0;
        ];

        let split_vector = dmatrix![
            1.0, 1.0;
            1.0, 2.0;
        ];

        data.split_by(&split_vector);
    }
}