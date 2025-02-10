use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};
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

pub trait Split<T> {
    fn get_keys(&self) -> HashSet<Vec<String>>;

    fn split_by(&self, other: &DMatrix<f64>) -> HashMap<Vec<String>, T>;
}

impl Split<DMatrix<f64>> for DMatrix<f64> {
    fn get_keys(&self) -> HashSet<Vec<String>> {
        let mut keys = HashSet::new();

        for row in self.row_iter() {
            let key : Vec<String> = row.iter().map(|s| s.to_string()).collect();
            keys.insert(key);
        }

        keys
    }

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

impl Split<DVector<f64>> for DVector<f64> {
    fn get_keys(&self) -> HashSet<Vec<String>> {
        let mut keys = HashSet::new();

        for value in self.iter() {
            let key : Vec<String> = vec![value.to_string()];
            keys.insert(key);
        }

        keys
    }

    fn split_by(&self, other: &DMatrix<f64>) -> HashMap<Vec<String>, DVector<f64>> {
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

        let mut hash_map : HashMap<Vec<String>, DVector<f64>> = HashMap::new();

        for entry in index_map.into_iter() {
            let mut vector = DVector::<f64>::zeros(entry.1.len());

            for (r_new, r_old) in entry.1.into_iter().enumerate() {
                vector.set_row(r_new, &self.row(r_old));
            }

            hash_map.insert(entry.0.clone(), vector);
        }

        hash_map
    }
}

struct MutableF64Count {
    key: f64,
    count_cases: RefCell<usize>,
    first_weight: RefCell<f64>,
    count_weighted: RefCell<f64>,
}

impl MutableF64Count {
    pub fn init(key: f64, count_cases: usize, first_weight: f64, count_weighted: f64) -> MutableF64Count {
        MutableF64Count {
            key,
            count_cases: RefCell::new(count_cases),
            first_weight: RefCell::new(first_weight),
            count_weighted: RefCell::new(count_weighted),
        }
    }
}

impl Ord for MutableF64Count {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.key > other.key {
            Ordering::Greater
        } else if self.key < other.key {
            Ordering::Less
        } else {
            Ordering::Equal
        }
    }
}

impl PartialOrd for MutableF64Count {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for MutableF64Count {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for MutableF64Count {}

impl Clone for MutableF64Count {
    fn clone(&self) -> Self {
        MutableF64Count {
            key: self.key,
            count_cases: RefCell::new(0),
            first_weight: RefCell::new(0.0),
            count_weighted: RefCell::new(0.0),
        }
    }
}

pub struct ImmutableF64Count {
    key: f64,
    count_cases: usize,
    first_weight: f64,
    count_weighted: f64,
}

impl ImmutableF64Count {
    pub fn init(key: f64, count_cases: usize, first_weight: f64, count_weighted: f64) -> Self {
        ImmutableF64Count { key, count_cases, first_weight, count_weighted }
    }

    pub fn get_key(&self) -> f64 {
        self.key
    }

    pub fn get_count_cases(&self) -> usize {
        self.count_cases
    }

    pub fn get_first_weight(&self) -> f64 {
        self.first_weight
    }

    pub fn get_count_weighted(&self) -> f64 {
        self.count_weighted
    }
}

#[macro_export]
macro_rules! assert_f64count_eq {
    ( $x: expr, $y: expr ) => {
        assert_eq!($x.get_key(), $y.get_key(), "unequal key: left {:?}, right {:?}", $x.get_key(), $y.get_key());
        assert_eq!($x.get_count_cases(), $y.get_count_cases(), "unequal count cases: left {:?}, right {:?}", $x.get_count_cases(), $y.get_count_cases());
        assert_eq!($x.get_first_weight(), $y.get_first_weight(), "unequal first weight: left {:?}, right {:?}", $x.get_first_weight(), $y.get_first_weight());
        assert_eq!($x.get_count_weighted(), $y.get_count_weighted(), "unequal key: left {:?}, right {:?}", $x.get_count_weighted(), $y.get_count_weighted());
    };
}


pub struct OrderedF64Counts {
    counts: BTreeSet<MutableF64Count>,
    sum_of_cases: usize,
    sum_of_weights: f64,
}

impl OrderedF64Counts {
    pub fn new() -> OrderedF64Counts {
        OrderedF64Counts {
            counts: BTreeSet::new(),
            sum_of_cases: 0,
            sum_of_weights: 0.0,
        }
    }

    pub fn push(&mut self, key: f64, weight: f64) {
        if key.is_nan() {
            return;
        }

        self.sum_of_cases += 1;
        self.sum_of_weights += weight;

        let dummy = MutableF64Count::init(key, 1, weight, weight);

        if self.counts.contains(&dummy) {
            *self.counts.get(&dummy).unwrap().count_cases.borrow_mut() += 1;
            *self.counts.get(&dummy).unwrap().count_weighted.borrow_mut() += weight;
        } else {
            self.counts.insert(dummy);
        }
    }

    pub fn get_counts(&self) -> Vec<ImmutableF64Count> {
        Vec::from_iter(self.counts.iter().map(|count| ImmutableF64Count::init(count.key, count.count_cases.borrow().to_owned(), count.first_weight.borrow().to_owned(), count.count_weighted.borrow().to_owned())))
    }

    pub fn get_sum_of_cases(&self) -> usize {
        self.sum_of_cases
    }

    pub fn get_sum_of_weights(&self) -> f64 {
        self.sum_of_weights
    }
}

#[macro_export]
macro_rules! assert_approx_eq_iter_f64 {
    ( $x: expr, $y: expr, $eps: literal ) => {
        assert_eq!($x.len(), $y.len(), "unequal length");
        for (i, value) in $x.iter().enumerate() {
            assert!(f64::abs(value - $y.get(i).unwrap()) < $eps, "unequal value (epsilon {}) at index {}", $eps, i);
        }
    };
    ( $x: expr, $y: expr ) => {
        assert_eq!($x.len(), $y.len(), "unequal length");
        for (i, value) in $x.iter().enumerate() {
            assert!(f64::abs(value - $y.get(i).unwrap()) < 1e-10, "unequal value at index {}", i);
        }
    };
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
    fn test_get_keys() {
        let split_matrix = dmatrix![
            1.0, 1.0;
            1.0, 2.0;
            2.0, 1.0;
            2.0, 2.0;
            1.0, 1.0;
            f64::NAN, 1.0;
            1.0, f64::NAN;
            1.0, 2.0;
            2.0, 1.0;
        ];

        let result = split_matrix.get_keys();

        assert_eq!(result.len(), 6);
        assert!(result.contains(&vec!["1".to_string(), "2".to_string()]));
        assert!(result.contains(&vec!["1".to_string(), "NaN".to_string()]));
        assert!(!result.contains(&vec!["2".to_string(), "NaN".to_string()]));
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

    #[test]
    fn test_get_keys_dvector() {
        let split_vector = dvector![
            1.0,
            1.0,
            2.0,
            2.0,
            1.0,
            f64::NAN,
            1.0,
            1.0,
            2.0,
        ];

        let result = split_vector.get_keys();

        assert_eq!(result.len(), 3);
        assert!(result.contains(&vec!["1".to_string()]));
        assert!(result.contains(&vec!["2".to_string()]));
        assert!(result.contains(&vec!["NaN".to_string()]));
    }

    #[test]
    fn test_split_dvector() {
        let data = dvector![
            1.0,
            4.0,
            7.0,
            10.0,
            13.0,
        ];

        let split_data = dmatrix![
            1.0, 1.0;
            1.0, 2.0;
            2.0, 1.0;
            2.0, 2.0;
            1.0, 1.0;
        ];

        let result = data.split_by(&split_data);
        assert_eq!(4, result.len());
        assert_eq!(2, result[&vec!["1".to_string(), "1".to_string()]].nrows());
        assert_eq!(13.0, result[&vec!["1".to_string(), "1".to_string()]][1]);
        assert_eq!(1, result[&vec!["2".to_string(), "2".to_string()]].nrows());
        assert_eq!(10.0, result[&vec!["2".to_string(), "2".to_string()]][0]);
    }

    #[test]
    fn test_f64count() {
        let count1 = MutableF64Count::init(2.3, 4, 17.99, 138.2);
        let count2 = MutableF64Count::init(1.8, 2, 17.99, 99.6);
        let count3 = MutableF64Count::init(2.3, 0, 0.0, 0.0);

        assert!(count1 != count2);
        assert!(count1 == count3);

        assert_eq!(count1.cmp(&count2), Ordering::Greater);
        assert_eq!(count1.cmp(&count3), Ordering::Equal);
    }

    #[test]
    fn test_ordered_f64_counts() {
        let mut order_weighted_f64_counts = OrderedF64Counts::new();

        order_weighted_f64_counts.push(2.0, 1.0);
        order_weighted_f64_counts.push(1.0, 1.0);

        assert_eq!(2, order_weighted_f64_counts.get_counts().len());
        assert!((order_weighted_f64_counts.get_counts()[0].key - 1.0).abs() < f64::EPSILON);
        assert_eq!(order_weighted_f64_counts.get_counts()[0].count_cases, 1);
        assert!((order_weighted_f64_counts.get_counts()[0].count_weighted - 1.0).abs() < f64::EPSILON);
        assert!((order_weighted_f64_counts.get_counts()[0].first_weight - 1.0).abs() < f64::EPSILON);
        assert_eq!(order_weighted_f64_counts.get_sum_of_cases(), 2);
        assert!((order_weighted_f64_counts.get_sum_of_weights() - 2.0).abs() < f64::EPSILON);

        order_weighted_f64_counts.push(1.0, 1.5);
        order_weighted_f64_counts.push(1.5, 0.75);
        order_weighted_f64_counts.push(1.0, 1.5);

        assert_eq!(3, order_weighted_f64_counts.get_counts().len());
        assert!((order_weighted_f64_counts.get_counts()[0].key - 1.0).abs() < f64::EPSILON);
        assert_eq!(order_weighted_f64_counts.get_counts()[0].count_cases, 3);
        assert!((order_weighted_f64_counts.get_counts()[0].first_weight - 1.0).abs() < f64::EPSILON);
        assert!((order_weighted_f64_counts.get_counts()[0].count_weighted - 4.0).abs() < f64::EPSILON);
        assert!((order_weighted_f64_counts.get_counts()[1].key - 1.5).abs() < f64::EPSILON);
        assert_eq!(order_weighted_f64_counts.get_counts()[1].count_cases, 1);
        assert!((order_weighted_f64_counts.get_counts()[1].first_weight - 0.75).abs() < f64::EPSILON);
        assert!((order_weighted_f64_counts.get_counts()[1].count_weighted - 0.75).abs() < f64::EPSILON);
        assert!((order_weighted_f64_counts.get_counts()[2].key - 2.0).abs() < f64::EPSILON);
        assert_eq!(order_weighted_f64_counts.get_counts()[2].count_cases, 1);
        assert!((order_weighted_f64_counts.get_counts()[2].first_weight - 1.0).abs() < f64::EPSILON);
        assert!((order_weighted_f64_counts.get_counts()[2].count_weighted - 1.0).abs() < f64::EPSILON);
        assert!((order_weighted_f64_counts.get_sum_of_weights() - 5.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ordered_f64_counts_with_nan() {
        let mut order_weighted_f64_counts = OrderedF64Counts::new();

        order_weighted_f64_counts.push(2.0, 1.0);
        order_weighted_f64_counts.push(f64::NAN, 1.0);

        assert_eq!(1, order_weighted_f64_counts.get_counts().len());
        assert!((order_weighted_f64_counts.get_counts()[0].key - 2.0).abs() < f64::EPSILON);
        assert!((order_weighted_f64_counts.get_counts()[0].count_weighted - 1.0).abs() < f64::EPSILON);
        assert!((order_weighted_f64_counts.get_sum_of_weights() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_assert_approx_eq_iter_f64() {
        assert_approx_eq_iter_f64!(vec![1.0, -5.0], vec![1.0000000000001, -5.0]);
        assert_approx_eq_iter_f64!(vec![1.0, -5.0], vec![1.0000000000001, -5.0], 1e-5);
        assert_approx_eq_iter_f64!(dvector![1.0, -5.0], dvector![1.0000000000001, -5.0], 1e-5);
        assert_approx_eq_iter_f64!(vec![1.0, -5.0], dvector![1.0000000000001, -5.0], 1e-5);
    }

    #[test]
    #[should_panic(expected = "unequal length")]
    fn test_assert_approx_eq_iter_f64_fails_unequal_length() {
        assert_approx_eq_iter_f64!(vec![1.0, -5.0], vec![1.0000000000001]);
    }

    #[test]
    #[should_panic(expected = "unequal value at index 0")]
    fn test_assert_approx_eq_iter_f64_fails_different_values() {
        assert_approx_eq_iter_f64!(vec![1.0, -5.0], vec![1.1, -5.0]);
    }

    #[test]
    #[should_panic(expected = "unequal value (epsilon 0.000000000000001) at index 0")]
    fn test_assert_approx_eq_iter_f64_fails_epsilon() {
        assert_approx_eq_iter_f64!(vec![1.0, -5.0], vec![1.0000000000001, -5.0], 1e-15);
    }
}