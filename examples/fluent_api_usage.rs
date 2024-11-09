extern crate replicest;

use nalgebra::dvector;
use replicest::analysis;

pub fn main() {
    let wgts = dvector![1.1, 1.5, 1.3, 1.7, 1.7, 1.0];

    let mut analysis = analysis::analysis();

    analysis.set_weights(&wgts).mean();
    println!("{}", analysis.summary());

    analysis.mean().set_weights(&wgts);
    println!("{}", analysis.summary());

    let analysis2 = analysis.copy();
    println!("{}", analysis2.summary());

    let new_wgts = dvector![2.1, 2.5, 2.3, 2.7, 2.7, 2.0];

    analysis.set_weights(&new_wgts);
    println!("{}", analysis.summary());
    println!("{}", analysis2.summary());
}