use criterion::{criterion_group, criterion_main};

pub mod benchmark {
    pub mod helper;
    pub mod replication;

    use nalgebra::{DMatrix, DVector};

    pub struct TestData {
        pub data: Vec<DMatrix<f64>>,
        pub cat: DMatrix<f64>,
        pub wgt: DVector<f64>,
        pub repwgt: DMatrix<f64>,
    }

    pub fn fetch_test_dataset() -> TestData {
        let mut reader_builder = csv::ReaderBuilder::new();
        reader_builder.has_headers(false);

        let mut data: Vec<DMatrix<f64>> = Vec::new();

        for imputation in 1..=5 {
            let mut reader = reader_builder.from_path(format!("./tests/_data/imp{}.csv", imputation)).unwrap();
            let mut nrows = 0;
            let mut values = Vec::new();

            for record in reader.records() {
                for field in &record.unwrap() {
                    values.push(field.parse::<f64>().unwrap());
                }
                nrows += 1;
            }

            let ncols = values.len() / nrows;

            let data_imputation = DMatrix::from_row_slice(nrows, ncols, &values);
            data.push(data_imputation);
        }

        let mut x: Vec<&DMatrix<f64>> = Vec::new();
        for data_entry in &data {
            x.push(&data_entry);
        }

        let mut reader = reader_builder.from_path("./tests/_data/cat.csv").unwrap();
        let mut nrows = 0;
        let mut values = Vec::new();

        for record in reader.records() {
            for field in &record.unwrap() {
                values.push(field.parse::<f64>().unwrap());
            }
            nrows += 1;
        }

        let ncols = values.len() / nrows;

        let cat = DMatrix::from_row_slice(nrows, ncols, &values);

        let mut reader = reader_builder.from_path("./tests/_data/wgt.csv").unwrap();
        let mut values = Vec::new();

        for record in reader.records() {
            for field in &record.unwrap() {
                values.push(field.parse::<f64>().unwrap());
            }
        }

        let wgt = DVector::from(values);

        let mut reader = reader_builder.from_path("./tests/_data/repwgt.csv").unwrap();
        let mut nrows = 0;
        let mut values = Vec::new();

        for record in reader.records() {
            for field in &record.unwrap() {
                values.push(field.parse::<f64>().unwrap());
            }
            nrows += 1;
        }

        let ncols = values.len() / nrows;

        let repwgt = DMatrix::from_row_slice(nrows, ncols, &values);

        TestData {
            data,
            cat,
            wgt,
            repwgt,
        }
    }
}

criterion_group!(benches,
    benchmark::helper::large_benchmark_get_keys,
    benchmark::helper::large_benchmark_split_by,
    benchmark::replication::small_benchmark_mean,
    benchmark::replication::large_benchmark_mean,
    benchmark::replication::large_benchmark_correlation
);
criterion_main!(benches);