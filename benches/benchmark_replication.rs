use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{dvector, DMatrix, DVector};
use replicest::replication;
use replicest::estimates;

fn fetch_test_dataset() -> (Vec<DMatrix<f64>>, DVector<f64>, DMatrix<f64>) {
    let mut reader_builder = csv::ReaderBuilder::new();
    reader_builder.has_headers(false);

    let mut data : Vec<DMatrix<f64>> = Vec::new();

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

    let mut x : Vec<&DMatrix<f64>> = Vec::new();
    for data_entry in &data {
        x.push(&data_entry);
    }

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

    (data, wgt, repwgt)
}
pub fn small_benchmark_mean(c: &mut Criterion) {
    c.bench_function("mean n3 c4 i3 wgt3", |b| b.iter(|| {
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
        let rep_wgts = DMatrix::from_row_slice(3, 3, &[
            0.0, 1.0, 1.0,
            0.5, 0.0, 0.5,
            1.5, 1.5, 0.0,
        ]);

        let _ = replication::replicate_estimates(
            black_box(estimates::mean),
            black_box(&imp_data),
            black_box(&wgt),
            black_box(&rep_wgts),
            black_box(1.0));
    }));
}

pub fn large_benchmark_mean(c: &mut Criterion) {
    let (data, wgt, repwgt) = fetch_test_dataset();
    let mut x = Vec::new();
    for data1 in data.iter() {
        x.push(data1);
    }

    c.bench_function("mean n10000 c5 i5 wgt50", |b| b.iter(|| {
        replication::replicate_estimates(
            black_box(estimates::mean),
            black_box(&x),
            black_box(&wgt),
            black_box(&repwgt),
            black_box(1.0)
        );
    }));
}

pub fn large_benchmark_correlation(c: &mut Criterion) {
    let (data, wgt, repwgt) = fetch_test_dataset();
    let mut x = Vec::new();
    for data1 in data.iter() {
        x.push(data1);
    }

    c.bench_function("mean n10000 c5 i5 wgt50", |b| b.iter(|| {
        replication::replicate_estimates(
            black_box(estimates::correlation),
            black_box(&x),
            black_box(&wgt),
            black_box(&repwgt),
            black_box(1.0)
        );
    }));
}

criterion_group!(benches, small_benchmark_mean, large_benchmark_mean, large_benchmark_correlation);
criterion_main!(benches);
