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
When building the library, bindings for C# and Python are created automatically via [UniFFI](https://mozilla.github.io/uniffi-rs/latest/) (see [build.rs](build.rs)) into folder [/bindings](./bindings).

You can use those bindings to call directly into the dynamic system library (libreplicest.so or libreplicest.dll).
Just make sure your C# or Python project references the library and the bindings correctly.

#### C#.NET example
A) Reference and use the bindings.
```
using uniffi.replicest;
```

B) Load or produce some data.
```
double[][] rawData = [ [1.0, 2.0], [2.5, 1.5], [3.0, 3.5], [3.5, double.NaN], [4.0, 3.0], [5.0, 5.0] ];
List<double> weights = [1.0, 0.5, 1.5, 1.0, 0.5, 1.5];

Random rng = new(12345);
var x = Enumerable.Range(1, 5).Select(_ => {
    var impData = rawData.Select(row => row.ToList()).ToList();
    impData[3][1] = rng.NextDouble() * 4.0 + 1.0;
    return impData;
}).ToList();

var repWeights = Enumerable.Range(0, weights.Count).Select(rr => {
    var repWeightsRow = Enumerable.Repeat(weights[rr], weights.Count).ToList();
    repWeightsRow[rr] = 0.0;
    repWeightsRow[rr + (rr % 2 == 0 ? 1 : -1)] *= 2.0;
    return repWeightsRow;
}).ToList();
```
Note that when it comes to data and replicate weights matrices, those are represented as `List<List<double>>` in row-major order.

C) Calculate some results.
```
var result = ReplicestMethods.ReplicateEstimates(
    Estimate.LinearRegression,
    new Dictionary<string, string> { {"intercept", "false"} },
    x,
    [ weights ],
    [ repWeights ],
    0.5
);

Console.WriteLine($"R-squared is {result.finalEstimates[2]} with standard error of {result.standardErrors[2]}");
```

### replicest server (via Unix Domain Socket)