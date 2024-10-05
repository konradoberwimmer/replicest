use std::process::Command;

use uniffi_bindgen::{generate_bindings};
use uniffi_bindgen::bindings::TargetLanguage;

fn main() {
    let config_file = "./bindings/uniffi.toml";
    let udl_file = "./src/replicest.udl";
    let out_dir = "./bindings/";

    uniffi::generate_scaffolding(udl_file).unwrap();

    generate_bindings(
        udl_file.into(),
        None,
        vec![TargetLanguage::Python],
        Some(out_dir.into()),
        None,
        Some("replicest"),
        true
    ).unwrap();

    Command::new("uniffi-bindgen-cs")
        .arg("--out-dir").arg(out_dir)
        .arg(udl_file)
        .arg("--config")
        .arg(config_file)
        .output()
        .expect("Failed when generating C# bindings");
}