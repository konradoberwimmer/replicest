use nalgebra::{dvector, DMatrix, DVector};
use replicest::{estimates, replication};

#[test]
fn test_replicate_estimates_for_a_mean() {
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

    let result = replication::replicate_estimates(estimates::mean, &x, &wgt, &repwgt, 1.0);
    assert_eq!(5, result.parameter_names().len());
    assert_eq!("mean_x5", result.parameter_names()[4]);
    assert_eq!(0, (result.final_estimates() - dvector![0.03599087180982961, 0.054048403991529756, 0.054505197688378325, 0.04159357573970399, 0.042906564801087246]).iter().filter(|&&v| v > 1e-10).count());
    assert_eq!(0, (result.standard_errors() - dvector![0.01107463624697274, 0.009856163557127698, 0.00938364895138013, 0.010089172012288873, 0.011135325191130828]).iter().filter(|&&v| v > 1e-10).count());

    let result_again = replication::replicate_estimates(estimates::mean, &x, &wgt, &repwgt, 1.0);
    assert_eq!(5, result_again.parameter_names().len());
    assert_eq!("mean_x5", result_again.parameter_names()[4]);
    assert_eq!(0, (result_again.final_estimates() - dvector![0.03599087180982961, 0.054048403991529756, 0.054505197688378325, 0.04159357573970399, 0.042906564801087246]).iter().filter(|&&v| v > 1e-10).count());
    assert_eq!(0, (result_again.standard_errors() - dvector![0.01107463624697274, 0.009856163557127698, 0.00938364895138013, 0.010089172012288873, 0.011135325191130828]).iter().filter(|&&v| v > 1e-10).count());
}