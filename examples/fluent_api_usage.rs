extern crate replicest;

use nalgebra::{dmatrix, dvector};
use replicest::analysis;
use replicest::analysis::Imputation;

pub fn main() {
    let data = dmatrix![
        2.1, 1.7;
        2.9, 1.9;
        1.9, 1.7;
        1.6, 1.3;
        1.4, 1.6;
        2.7, 2.3;
    ];

    let wgts = dvector![1.1, 1.5, 1.3, 1.7, 1.7, 1.0];

    let replicate_weights = dmatrix![
        0.0, 2.2, 1.1, 1.1, 1.1, 1.1;
        3.0, 0.0, 1.5, 1.5, 1.5, 1.5;
        1.3, 1.3, 0.0, 2.6, 1.3, 1.3;
        1.7, 1.7, 3.4, 0.0, 1.7, 1.7;
        1.7, 1.7, 1.7, 1.7, 0.0, 3.4;
        1.0, 1.0, 1.0, 1.0, 2.0, 0.0;
    ];

    let mut analysis = analysis::analysis();

    analysis
        .set_weights(&wgts)
        .with_replicate_weights(&replicate_weights)
        .set_variance_adjustment_factor(0.5);
    println!("Analysis setup: {}", analysis.summary());

    let result_mean = analysis
        .for_data(Imputation::No(&data))
        .mean()
        .calculate()
        .expect("unable to calculate means");
    println!("Analysis 1: {}", analysis.summary());
    let result_mean_overall = &result_mean[&vec!["overall".to_string()]];
    println!("means are {:?} with standard errors of {:?}",
             result_mean_overall.final_estimates().iter().map(|v| v.clone()).collect::<Vec<f64>>(),
             result_mean_overall.standard_errors().iter().map(|v| v.clone()).collect::<Vec<f64>>());

    let mut analysis2 = analysis.copy();
    let result_correlation = analysis2
        .correlation()
        .calculate()
        .expect("unable to calculate correlations");
    println!("Analysis 2: {}", analysis2.summary());
    let result_correlation_overall = &result_correlation[&vec!["overall".to_string()]];
    println!("correlation is {:?} with standard error of {:?}",
             result_correlation_overall.final_estimates()[4],
             result_correlation_overall.standard_errors()[4]);
}