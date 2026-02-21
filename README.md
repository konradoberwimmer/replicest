# replicest
Crate replicest offers calculations of statistical coefficients and their standard errors common in Large Scale Assessment (LSA).

## Usage from within Rust

### Elementary functions
You can, of course, use the elementary (pure) functions exposed by the crate. 
See [example elementary_functions_usage.rs](examples/elementary_functions_usage.rs).

API doc: [https://konradoberwimmer.github.io/replicest/](https://konradoberwimmer.github.io/replicest/)

### Fluent API
A more convenient way to use this crate for calculation is via the fluent API provided by the Analysis struct.
See [example fluent_api_usage.rs](examples/fluent_api_usage.rs).

Most importantly, when an Analysis struct instance is cloned, a shallow copy with references to already provided data, 
weights or replicate weights is created. This allows for memory-efficient calculation of multiple estimates of multiple 
data vectors.

## Usage from other languages

### Foreign function interface bindings
When building the library, bindings for C# and Python are created automatically via [UniFFI](https://mozilla.github.io/uniffi-rs/latest/) (see [build.rs](build.rs)) into folder [/bindings](./bindings).

You can use those bindings to call directly into the dynamic system library (libreplicest.so or libreplicest.dll).
Make sure your C# or Python project references the library and the bindings correctly.

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

### replicest_server (via TCP)
Another way to use the capacities of replicest from outside Rust is to build and start up the binary [replicest_server](./src/bin/replicest_server.rs) and communicate with that process over TCP.

#### C#.NET example
A) Prepare a class for ReplicatedEstimates that can be (de-)serialized using MessagePack.
```
using MessagePack;

[MessagePackObject]
public class ReplicatedEstimates
{
    [Key(0)]
    public string[] parameterNames { get; set; }
    [Key(1)]
    public double[] finalEstimates { get; set; }
    [Key(2)]
    public double[] samplingVariances { get; set; }
    [Key(3)]
    public double[] imputationVariances { get; set; }
    [Key(4)]
    public double[] standardErrors { get; set; }

    public override string ToString()
    {
        return string.Join("\n", parameterNames.Select((parameterName, index) => parameterName + ": " + finalEstimates[index] + " (" + standardErrors[index] + ")").ToArray());
    }
}
```
Note: replicest_server uses [MessagePack](https://msgpack.org/) to serialize calculation results and send them over TCP. To keep message sizes small, no meta-information is included, so make sure that a class or struct in a foreign language matches struct ReplicatedEstimates from [replication.rs](./src/replication.rs).

B) Start replicest_server as a child process and set up TCP ports.

Note: The following is just a most simple example. In productive code, choosing ports and waiting for the server to listen has to be done with more care.
```
using System.Diagnostics;
using System.Net.Sockets;

Process.Start("/path/to/replicest_server", "-s 8098 -d 8099");
Thread.Sleep(1000);

TcpClient client = new("127.0.0.1", 8098);
var stream = client.GetStream();
StreamWriter writer = new(stream);
StreamReader reader = new(stream);
```
Note: replicest_server uses two ports, one for commands and one for incoming data, weights and replicate weights vectors.

C) Load or produce some data.
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

D) For ease of use, prepare some functions that wrap single communication steps, e.g.,
```
void SendAndPrintResponse(string message, List<byte[]> data) {
    writer.WriteLine(message);
    writer.Flush();

    foreach (var data0 in data)
    {
        TcpClient dataClient = new("127.0.0.1", 8099);
        dataClient.GetStream().Write(data0);
        dataClient.Close();
    }

    var response = reader.ReadLine();

    Console.WriteLine(response ?? "No response");
}

void ReceiveAndPrintResult()
{
    var buffer = new byte[65536];
    stream.Read(buffer, 0, buffer.Length);

    var result = MessagePackSerializer.Deserialize<Dictionary<string[], ReplicatedEstimates>>(buffer);
    foreach (var key in result.Keys)
    {
        Console.WriteLine($"{string.Join(", ", key)}:\n{result[key]}");
    }
    
    Array.Clear(buffer, 0, buffer.Length);
}
```

E) Communicate with replicest_server.
```
SendAndPrintResponse("mean", []);

List<byte[]> dataStream = x.
    Select(imp => imp.
        Select(row => row.
            Select(value => BitConverter.GetBytes(value)).
            SelectMany(a => a).ToArray())
        .SelectMany(a => a).ToArray())
    .ToList();
SendAndPrintResponse("data 5 2", dataStream);

List<byte[]> weightsStream = [ weights.
    Select(value => BitConverter.GetBytes(value)).
    SelectMany(w => w).ToArray() ];
SendAndPrintResponse("weights", weightsStream);

List<byte[]> repWeightsStream = [ repWeights.
    Select(row => row.
        Select(value => BitConverter.GetBytes(value)).
        SelectMany(w => w).ToArray())
    .SelectMany(w => w).ToArray() ];
SendAndPrintResponse("replicate weights 6", repWeightsStream);

SendAndPrintResponse("calculate", []);
ReceiveAndPrintResult();

SendAndPrintResponse("shutdown", []);

client.Close();
```
Note:

- Matrices (data, groups and replicate weights) are transmitted as byte-streams in row-major order.
- Imputations (data and groups) are transmitted separately but within one command of "data" or "groups".
- For a full list of commands, see [replicest_server.md](./docs/replicest_server.md).