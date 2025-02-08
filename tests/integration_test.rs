use std::collections::HashMap;
use std::sync::Arc;
use nalgebra::{dvector, DMatrix, DVector};
use replicest::{assert_approx_eq_iter_f64, estimates, replication};
use replicest::analysis::{analysis, Imputation};

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

fn fetch_pirls_2021_aut_dataset(vars: &[&str], group_vars: &[&str]) -> (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>, DVector<f64>, DMatrix<f64>) {
    let mut reader_builder = csv::ReaderBuilder::new();
    reader_builder.has_headers(true).delimiter(b';');

    let mut pre_reader = reader_builder.from_path("./misc/BIFIEsurveyComparison/data/asgautr5.csv").unwrap();
    let n_cases = pre_reader.records().count();

    let mut wgt : DVector<f64> = DVector::zeros(n_cases);
    let mut repwgt : DMatrix<f64> = DMatrix::zeros(n_cases, 160);
    let mut data : Vec<DMatrix<f64>> = Vec::new();
    for _ in 0..5 {
        data.push(DMatrix::zeros(n_cases, vars.len()));
    }
    let mut groups : Vec<DMatrix<f64>> = Vec::new();
    for _ in 0..5 {
        groups.push(DMatrix::zeros(n_cases, group_vars.len()));
    }

    let mut reader = reader_builder.from_path("./misc/BIFIEsurveyComparison/data/asgautr5.csv").unwrap();
    for (rr, result) in reader.deserialize().enumerate() {
        let record: HashMap<String, String> = result.unwrap();

        wgt[rr] = record["TOTWGT"].parse::<f64>().unwrap();

        let jk_zone = record["JKZONE"].parse::<usize>().unwrap();
        let jk_rep = record["JKREP"].parse::<usize>().unwrap();
        for jj in 0..80 {
            repwgt[(rr, jj)] =
                wgt[rr] * (jk_zone != (jj + 1)) as usize as f64 +
                wgt[rr] * (jk_zone == (jj + 1)) as usize as f64 * jk_rep as f64 * 2.0;
        }
        for jj in 80..160 {
            repwgt[(rr, jj)] =
                wgt[rr] * (jk_zone + 80 != (jj + 1)) as usize as f64 +
                wgt[rr] * (jk_zone + 80 == (jj + 1)) as usize as f64 * (1 - jk_rep) as f64 * 2.0;
        }

        for (vv, var) in vars.iter().enumerate() {
            if record.contains_key(*var) {
                for ii in 0..5 {
                    data[ii][(rr, vv)] = record[*var].parse::<f64>().unwrap_or(f64::NAN);
                }
            } else {
                for ii in 0..5 {
                    let mut varname = var.to_string();
                    varname.push_str(format!("{:02}", ii + 1).as_str());
                    data[ii][(rr, vv)] = record[varname.as_str()].parse::<f64>().unwrap_or(f64::NAN);
                }
            }
        }

        for (vv, group_var) in group_vars.iter().enumerate() {
            if record.contains_key(*group_var) {
                for ii in 0..5 {
                    groups[ii][(rr, vv)] = record[*group_var].parse::<f64>().unwrap_or(f64::NAN);
                }
            } else {
                for ii in 0..5 {
                    let mut groupvarname = group_var.to_string();
                    groupvarname.push_str(format!("{:02}", ii + 1).as_str());
                    groups[ii][(rr, vv)] = record[groupvarname.as_str()].parse::<f64>().unwrap_or(f64::NAN);
                }
            }
        }
    }

    (data, groups, wgt, repwgt)
}

#[test]
fn test_replicate_estimates_for_a_mean() {
    let (data, wgt, repwgt) = fetch_test_dataset();
    let mut x = Vec::new();
    for data1 in data.iter() {
        x.push(data1);
    }

    let result = replication::replicate_estimates(Arc::new(estimates::mean), &x, &vec![&wgt], &vec![&repwgt], 1.0);
    assert_eq!(5, result.parameter_names().len());
    assert_eq!("mean_x5", result.parameter_names()[4]);
    assert_approx_eq_iter_f64!(result.final_estimates(), dvector![0.03599087180982961, 0.054048403991529756, 0.054505197688378325, 0.04159357573970399, 0.042906564801087246]);
    assert_approx_eq_iter_f64!(result.standard_errors(), dvector![0.01107463624697274, 0.009856163557127698, 0.00938364895138013, 0.010089172012288873, 0.011135325191130828]);

    let result_again = replication::replicate_estimates(Arc::new(estimates::mean), &x, &vec![&wgt], &vec![&repwgt], 1.0);
    assert_eq!(5, result_again.parameter_names().len());
    assert_eq!("mean_x5", result_again.parameter_names()[4]);
    assert_approx_eq_iter_f64!(result_again.final_estimates(), dvector![0.03599087180982961, 0.054048403991529756, 0.054505197688378325, 0.04159357573970399, 0.042906564801087246]);
    assert_approx_eq_iter_f64!(result_again.standard_errors(), dvector![0.01107463624697274, 0.009856163557127698, 0.00938364895138013, 0.010089172012288873, 0.011135325191130828]);
}

#[test]
fn test_replicate_estimates_for_a_correlation() {
    let (data, wgt, repwgt) = fetch_test_dataset();
    let mut x = Vec::new();
    for data1 in data.iter() {
        x.push(data1);
    }

    let result = replication::replicate_estimates(Arc::new(estimates::correlation), &x, &vec![&wgt], &vec![&repwgt], 1.0);
    assert_eq!(30, result.parameter_names().len());
    assert_eq!("covariance_x2_x2", result.parameter_names()[5]);
    assert_approx_eq_iter_f64!(result.final_estimates(), dvector![
        0.913435058513876852, 0.021764158354910846, 0.005842810809959046, -0.000829821665348307, 0.004634505439142842,
        0.913500734645705692, -0.008081119326223522, -0.005526240863576300, 0.018290330542602797, 0.923770660723064241,
        0.010778846167137196, -0.002115275130500467, 0.954726079390342486, -0.007500670088574543, 0.962499943557485138,
        1.0, 0.023826006245670710, 0.006360705314737171, -0.000888269668222516, 0.004942613093088714,
        1.0, -0.008796981635998208, -0.005917824969554078, 0.019505239512784368,
        1.0, 0.011477586111653577, -0.002243519625001784, 1.0, -0.007824127579969456,
        1.0,
    ]);
    assert_approx_eq_iter_f64!(result.standard_errors(), dvector![
        0.01398990090945300, 0.01027745837922445, 0.01080888273649592, 0.01089186383797431, 0.00971497239023331,
        0.01319901673751378, 0.00925712972975630, 0.00971361144331722, 0.01143866292881792, 0.01416185391972646,
        0.01046492433520780, 0.01145257971705268, 0.01168454296456890, 0.01070477386945857, 0.01525351358517860,
        0.0, 0.0111654759926981, 0.0117671209501314, 0.0116575646337245, 0.0103674152513720,
        0.0, 0.0100767873824531, 0.0103900069320061, 0.0121846579457844,
        0.0, 0.0111549500870232, 0.0121424628555753,
        0.0, 0.0111466487848017,
        0.0,
    ]);

    let result_again = replication::replicate_estimates(Arc::new(estimates::correlation), &x, &vec![&wgt], &vec![&repwgt], 1.0);
    assert_eq!(30, result_again.parameter_names().len());
    assert_eq!("correlation_x4_x5", result_again.parameter_names()[28]);
    assert_approx_eq_iter_f64!(result_again.final_estimates(), dvector![
        0.913435058513876852, 0.021764158354910846, 0.005842810809959046, -0.000829821665348307, 0.004634505439142842,
        0.913500734645705692, -0.008081119326223522, -0.005526240863576300, 0.018290330542602797, 0.923770660723064241,
        0.010778846167137196, -0.002115275130500467, 0.954726079390342486, -0.007500670088574543, 0.962499943557485138,
        1.0, 0.023826006245670710, 0.006360705314737171, -0.000888269668222516, 0.004942613093088714,
        1.0, -0.008796981635998208, -0.005917824969554078, 0.019505239512784368,
        1.0, 0.011477586111653577, -0.002243519625001784, 1.0, -0.007824127579969456,
        1.0,
    ]);
    assert_approx_eq_iter_f64!(result_again.standard_errors(), dvector![
        0.01398990090945300, 0.01027745837922445, 0.01080888273649592, 0.01089186383797431, 0.00971497239023331,
        0.01319901673751378, 0.00925712972975630, 0.00971361144331722, 0.01143866292881792, 0.01416185391972646,
        0.01046492433520780, 0.01145257971705268, 0.01168454296456890, 0.01070477386945857, 0.01525351358517860,
        0.0, 0.0111654759926981, 0.0117671209501314, 0.0116575646337245, 0.0103674152513720,
        0.0, 0.0100767873824531, 0.0103900069320061, 0.0121846579457844,
        0.0, 0.0111549500870232, 0.0121424628555753,
        0.0, 0.0111466487848017,
        0.0,
    ]);
}

#[test]
fn test_analysis_correlation_for_pirls_2021_aut() {
    let (data, groups, wgt, repwgt) = fetch_pirls_2021_aut_dataset(&["ITSEX", "ASRREA"], &["ASRIBM"]);

    let mut analysis = analysis();
    analysis
        .correlation()
        .set_weights(&wgt)
        .with_replicate_weights(&repwgt)
        .set_variance_adjustment_factor(0.5);

    analysis
        .for_data(Imputation::Yes(&Vec::from_iter(data.iter())));

    let result_nogroups = analysis.calculate();
    assert!(result_nogroups.is_ok());

    let result_nogroups = result_nogroups.unwrap();
    assert_eq!(result_nogroups.len(), 1);

    assert_approx_eq_iter_f64!(result_nogroups[&vec!["overall".to_string()]].final_estimates(), vec![0.2499422, -3.5932410, 4711.2810086, 1.0, -0.1047216, 1.0], 1e-7);
    assert_approx_eq_iter_f64!(result_nogroups[&vec!["overall".to_string()]].standard_errors(), vec![0.0001352901, 0.6812778247, 157.0373642758, 0.0, 0.01967591, 0.0], 1e-7);

    analysis
        .group_by(Imputation::Yes(&Vec::from_iter(groups.iter())));

    let result = analysis.calculate();
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(5, result.len());

    assert_approx_eq_iter_f64!(result[&vec!["1".to_string()]].final_estimates(), vec![0.24397963, -0.05789637, 654.07200246, 1.0, -0.002617465, 1.0], 1e-8);
    assert_approx_eq_iter_f64!(result[&vec!["2".to_string()]].final_estimates(), vec![0.24547254,  0.12003083, 434.22210923, 1.0,  0.012008409, 1.0], 1e-8);
    assert_approx_eq_iter_f64!(result[&vec!["3".to_string()]].final_estimates(), vec![0.24889393, -0.40876264, 448.60889779, 1.0, -0.038705965, 1.0], 1e-8);
    assert_approx_eq_iter_f64!(result[&vec!["4".to_string()]].final_estimates(), vec![0.24898385, -0.52654363, 426.28373453, 1.0, -0.050899235, 1.0], 1e-8);
    assert_approx_eq_iter_f64!(result[&vec!["5".to_string()]].final_estimates(), vec![0.23506757, -0.98851340, 464.08738951, 1.0, -0.093617334, 1.0], 1e-8);
    assert_approx_eq_iter_f64!(result[&vec!["1".to_string()]].standard_errors(), vec![0.007082429, 0.976744098, 167.661312585, 0.0, 0.07785359, 0.0], 1e-8);
    assert_approx_eq_iter_f64!(result[&vec!["2".to_string()]].standard_errors(), vec![0.003005384, 0.677529944,  24.166076778, 0.0, 0.06555160, 0.0], 1e-8);
    assert_approx_eq_iter_f64!(result[&vec!["3".to_string()]].standard_errors(), vec![0.001063050, 0.410383983,  13.879319117, 0.0, 0.03882064, 0.0], 1e-8);
    assert_approx_eq_iter_f64!(result[&vec!["4".to_string()]].standard_errors(), vec![0.001165612, 0.486794874,  18.459875205, 0.0, 0.04666934, 0.0], 1e-8);
    assert_approx_eq_iter_f64!(result[&vec!["5".to_string()]].standard_errors(), vec![0.010483638, 0.878510479,  59.264497664, 0.0, 0.08172591, 0.0], 1e-8);
}
