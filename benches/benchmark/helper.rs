use criterion::{black_box, Criterion};
use replicest::helper;

pub fn large_benchmark_get_keys(c: &mut Criterion) {
    use helper::Split;

    let test_data = super::fetch_test_dataset();

    c.bench_function("get_keys n10000 cat2", |b| b.iter(|| {
        black_box(&test_data.cat).get_keys()
    }));
}

pub fn large_benchmark_split_by(c: &mut Criterion) {
    use helper::Split;

    let test_data = super::fetch_test_dataset();

    c.bench_function("split_by n10000 c5 cat2", |b| b.iter(|| {
        black_box(&test_data.data[0]).split_by(black_box(&test_data.cat))
    }));
}