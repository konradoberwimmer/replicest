use std::sync::Arc;
use criterion::{black_box, Criterion};
use nalgebra::{dvector, DMatrix};
use replicest::{data_preparation, replication};
use replicest::estimates;

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
            black_box(Arc::new(estimates::mean)),
            black_box(None),
            black_box(&imp_data),
            black_box(&vec![&wgt]),
            black_box(&vec![&rep_wgts]),
            black_box(1.0));
    }));
}

pub fn large_benchmark_mean(c: &mut Criterion) {
    let test_data = super::fetch_test_dataset();
    let mut x = Vec::new();
    for data1 in test_data.data.iter() {
        x.push(data1);
    }

    c.bench_function("mean n10000 c5 i5 wgt50", |b| b.iter(|| {
        replication::replicate_estimates(
            black_box(Arc::new(estimates::mean)),
            black_box(None),
            black_box(&x),
            black_box(&vec![&test_data.wgt]),
            black_box(&vec![&test_data.repwgt]),
            black_box(1.0)
        );
    }));
}

pub fn large_benchmark_correlation(c: &mut Criterion) {
    let test_data = super::fetch_test_dataset();
    let mut x = Vec::new();
    for data1 in test_data.data.iter() {
        x.push(data1);
    }

    c.bench_function("correlation n10000 c5 i5 wgt50", |b| b.iter(|| {
        replication::replicate_estimates(
            black_box(Arc::new(estimates::correlation)),
            black_box(None),
            black_box(&x),
            black_box(&vec![&test_data.wgt]),
            black_box(&vec![&test_data.repwgt]),
            black_box(1.0)
        );
    }));
}

pub fn large_benchmark_correlation_without_pairwise_delete(c: &mut Criterion) {
    let test_data = super::fetch_test_dataset();
    let mut x = Vec::new();
    for data1 in test_data.data.iter() {
        x.push(data1);
    }

    c.bench_function("correlation without pairwise deletion n10000 c5 i5 wgt50", |b| b.iter(|| {
        replication::replicate_estimates(
            black_box(Arc::new(move |x, wgt| estimates::correlation_with_options(&x, &wgt, false))),
            black_box(None),
            black_box(&x),
            black_box(&vec![&test_data.wgt]),
            black_box(&vec![&test_data.repwgt]),
            black_box(1.0)
        );
    }));
}

pub fn large_benchmark_correlation_with_listwise_delete(c: &mut Criterion) {
    let test_data = super::fetch_test_dataset();
    let mut x = Vec::new();
    for data1 in test_data.data.iter() {
        x.push(data1);
    }

    c.bench_function("correlation with listwise deletion n10000 c5 i5 wgt50", |b| b.iter(|| {
        replication::replicate_estimates(
            black_box(Arc::new(move |x, wgt| estimates::correlation_with_options(&x, &wgt, false))),
            black_box(Some(Arc::new(data_preparation::listwise_delete))),
            black_box(&x),
            black_box(&vec![&test_data.wgt]),
            black_box(&vec![&test_data.repwgt]),
            black_box(1.0)
        );
    }));
}

pub fn large_benchmark_linear_regression(c: &mut Criterion) {
    let test_data = super::fetch_test_dataset();
    let mut x = Vec::new();
    for data1 in test_data.data.iter() {
        x.push(data1);
    }

    c.bench_function("linreg n10000 c5 i5 wgt50", |b| b.iter(|| {
        replication::replicate_estimates(
            black_box(Arc::new(estimates::linreg)),
            black_box(Some(Arc::new(data_preparation::listwise_delete))),
            black_box(&x),
            black_box(&vec![&test_data.wgt]),
            black_box(&vec![&test_data.repwgt]),
            black_box(1.0)
        );
    }));
}

pub fn large_benchmark_linear_regression_without_listwise_deletion(c: &mut Criterion) {
    let test_data = super::fetch_test_dataset();
    let mut x = Vec::new();
    for data1 in test_data.data.iter() {
        x.push(data1);
    }

    c.bench_function("linreg without listwise deletion n10000 c5 i5 wgt50", |b| b.iter(|| {
        replication::replicate_estimates(
            black_box(Arc::new(estimates::linreg)),
            black_box(None),
            black_box(&x),
            black_box(&vec![&test_data.wgt]),
            black_box(&vec![&test_data.repwgt]),
            black_box(1.0)
        );
    }));
}