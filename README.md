# replicest
Crate replicest offers calculation of statistical coefficients and their standard errors common in Large Scale Assessment (LSA).

## Usage from within Rust

### Elementary functions
You can of course just use the elementary (pure) functions exposed by the crate. 
See [example elementary_functions_usage.rs](examples/elementary_functions_usage.rs).

API doc: [https://konradoberwimmer.github.io/replicest/](https://konradoberwimmer.github.io/replicest/)

### Fluent API
A more convenient way to use this crate for calculation is via the fluent API provided by the Analysis struct.
See [example fluent_api_usage.rs](examples/fluent_api_usage.rs).

Most importantly, when an Analysis struct instance is cloned, a shallow copy with references to already provided data, 
weights or replicate weights is created. This allows for memory efficient calculation of multiple estimates of multiple 
data vectors.

## Usage from other languages

### Foreign function interface bindings

### replicest server (via Unix Domain Socket)