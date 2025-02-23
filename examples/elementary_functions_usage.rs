extern crate replicest;

use std::sync::Arc;
use nalgebra::{dmatrix, dvector};
use replicest::{estimates, replication};

fn main() {
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

    let result = replication::replicate_estimates(
        Arc::new(estimates::correlation),
        &vec![&data],
        &vec![&wgts],
        &vec![&replicate_weights],
        1.0
    );

    println!("correlation {:?} with standard error of {:?}", result.final_estimates()[4], result.standard_errors()[4]);
}