use nalgebra::{dvector, DMatrix, DVector};
use replicest::{estimates, replication};

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

#[test]
fn test_replicate_estimates_for_a_mean() {
    let (data, wgt, repwgt) = fetch_test_dataset();
    let mut x = Vec::new();
    for data1 in data.iter() {
        x.push(data1);
    }

    let result = replication::replicate_estimates(estimates::mean, &x, &vec![&wgt], &vec![&repwgt], 1.0);
    assert_eq!(5, result.parameter_names().len());
    assert_eq!("mean_x5", result.parameter_names()[4]);
    assert_eq!(0, (result.final_estimates() - dvector![0.03599087180982961, 0.054048403991529756, 0.054505197688378325, 0.04159357573970399, 0.042906564801087246]).iter().filter(|&&v| v.abs() > 1e-10).count());
    assert_eq!(0, (result.standard_errors() - dvector![0.01107463624697274, 0.009856163557127698, 0.00938364895138013, 0.010089172012288873, 0.011135325191130828]).iter().filter(|&&v| v.abs() > 1e-10).count());

    let result_again = replication::replicate_estimates(estimates::mean, &x, &vec![&wgt], &vec![&repwgt], 1.0);
    assert_eq!(5, result_again.parameter_names().len());
    assert_eq!("mean_x5", result_again.parameter_names()[4]);
    assert_eq!(0, (result_again.final_estimates() - dvector![0.03599087180982961, 0.054048403991529756, 0.054505197688378325, 0.04159357573970399, 0.042906564801087246]).iter().filter(|&&v| v.abs() > 1e-10).count());
    assert_eq!(0, (result_again.standard_errors() - dvector![0.01107463624697274, 0.009856163557127698, 0.00938364895138013, 0.010089172012288873, 0.011135325191130828]).iter().filter(|&&v| v.abs() > 1e-10).count());
}

#[test]
fn test_replicate_estimates_for_a_correlation() {
    let (data, wgt, repwgt) = fetch_test_dataset();
    let mut x = Vec::new();
    for data1 in data.iter() {
        x.push(data1);
    }

    let result = replication::replicate_estimates(estimates::correlation, &x, &vec![&wgt], &vec![&repwgt], 1.0);
    assert_eq!(30, result.parameter_names().len());
    assert_eq!("covariance_x2_x2", result.parameter_names()[5]);
    assert_eq!(0, (result.final_estimates() - dvector![
        0.913435058513876852, 0.021764158354910846, 0.005842810809959046, -0.000829821665348307, 0.004634505439142842,
        0.913500734645705692, -0.008081119326223522, -0.005526240863576300, 0.018290330542602797, 0.923770660723064241,
        0.010778846167137196, -0.002115275130500467, 0.954726079390342486, -0.007500670088574543, 0.962499943557485138,
        1.0, 0.023826006245670710, 0.006360705314737171, -0.000888269668222516, 0.004942613093088714,
        1.0, -0.008796981635998208, -0.005917824969554078, 0.019505239512784368,
        1.0, 0.011477586111653577, -0.002243519625001784, 1.0, -0.007824127579969456,
        1.0,
    ]).iter().filter(|&&v| v.abs() > 1e-10).count());
    assert_eq!(0, (result.standard_errors() - dvector![
        0.01398990090945300, 0.01027745837922445, 0.01080888273649592, 0.01089186383797431, 0.00971497239023331,
        0.01319901673751378, 0.00925712972975630, 0.00971361144331722, 0.01143866292881792, 0.01416185391972646,
        0.01046492433520780, 0.01145257971705268, 0.01168454296456890, 0.01070477386945857, 0.01525351358517860,
        0.0, 0.0111654759926981, 0.0117671209501314, 0.0116575646337245, 0.0103674152513720,
        0.0, 0.0100767873824531, 0.0103900069320061, 0.0121846579457844,
        0.0, 0.0111549500870232, 0.0121424628555753,
        0.0, 0.0111466487848017,
        0.0,
    ]).iter().filter(|&&v| v.abs() > 1e-10).count());

    let result_again = replication::replicate_estimates(estimates::correlation, &x, &vec![&wgt], &vec![&repwgt], 1.0);
    assert_eq!(30, result_again.parameter_names().len());
    assert_eq!("correlation_x4_x5", result_again.parameter_names()[28]);
    assert_eq!(0, (result_again.final_estimates() - dvector![
        0.913435058513876852, 0.021764158354910846, 0.005842810809959046, -0.000829821665348307, 0.004634505439142842,
        0.913500734645705692, -0.008081119326223522, -0.005526240863576300, 0.018290330542602797, 0.923770660723064241,
        0.010778846167137196, -0.002115275130500467, 0.954726079390342486, -0.007500670088574543, 0.962499943557485138,
        1.0, 0.023826006245670710, 0.006360705314737171, -0.000888269668222516, 0.004942613093088714,
        1.0, -0.008796981635998208, -0.005917824969554078, 0.019505239512784368,
        1.0, 0.011477586111653577, -0.002243519625001784, 1.0, -0.007824127579969456,
        1.0,
    ]).iter().filter(|&&v| v.abs() > 1e-10).count());
    assert_eq!(0, (result_again.standard_errors() - dvector![
        0.01398990090945300, 0.01027745837922445, 0.01080888273649592, 0.01089186383797431, 0.00971497239023331,
        0.01319901673751378, 0.00925712972975630, 0.00971361144331722, 0.01143866292881792, 0.01416185391972646,
        0.01046492433520780, 0.01145257971705268, 0.01168454296456890, 0.01070477386945857, 0.01525351358517860,
        0.0, 0.0111654759926981, 0.0117671209501314, 0.0116575646337245, 0.0103674152513720,
        0.0, 0.0100767873824531, 0.0103900069320061, 0.0121846579457844,
        0.0, 0.0111549500870232, 0.0121424628555753,
        0.0, 0.0111466487848017,
        0.0,
    ]).iter().filter(|&&v| v.abs() > 1e-10).count());}
