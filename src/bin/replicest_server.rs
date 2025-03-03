use clap::{CommandFactory, FromArgMatches, Parser};
use std::collections::HashMap;
use std::error::Error;
use std::fs::remove_file;
use std::io::Read;
use std::os::unix::net::{UnixDatagram, UnixListener};
use std::path::PathBuf;
use nalgebra::{DMatrix, DVector};
use users::get_current_uid;
use replicest::analysis::*;
use replicest::errors::DataLengthError;
use replicest::ReplicatedEstimates;

/// Replicest server
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct CliArguments {
    /// Path for the UDS server socket (optional, defaults vary by OS)
    #[arg(long, short)]
    server_socket: Option<PathBuf>,

    /// Path for the UDS data socket  (optional, defaults vary by OS)
    #[arg(long, short)]
    data_socket: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli_args = CliArguments::from_arg_matches(&mut CliArguments::command().ignore_errors(true).get_matches())?;

    let (message_socket, data_socket) = setup_sockets(cli_args.server_socket, cli_args.data_socket)?;

    let mut current_analysis = analysis();

    loop {
        let mut buffer = [0; 1024];

        break match message_socket.recv_from(&mut buffer) {
            Ok((_, client_addr)) => {
                let message = trim_buffer(&buffer);

                println!("Received: {}", message);

                if message == "shutdown" {
                    message_socket.send_to_addr(b"shutting down", &client_addr)?;
                } else if message == "clear" {
                    current_analysis = analysis();
                    message_socket.send_to_addr(b"cleared", &client_addr)?;
                    continue;
                } else {
                    let response = handle_message(message, &mut current_analysis, &data_socket);
                    match response {
                        Ok(responses) => {
                            for response_data in responses {
                                message_socket.send_to_addr(&response_data, &client_addr)?;
                            }
                        }
                        Err(err) => {
                            message_socket.send_to_addr(format!("error: {}", err).as_bytes(), &client_addr)?;
                        }
                    }
                    continue;
                }
            }
            Err(_) => { }
        }
    }

    Ok(())
}

fn setup_sockets(server_socket_addr: Option<PathBuf>, data_socket_addr: Option<PathBuf>) -> Result<(UnixDatagram, UnixListener), Box<dyn Error>> {
    let user_id = get_current_uid();

    let message_socket_addr = match server_socket_addr {
        Some(server_socket_addr) => server_socket_addr,
        None => format!("/run/user/{}/replicest_server", user_id).parse().unwrap()
    };
    let _ = remove_file(&message_socket_addr);
    let message_socket = UnixDatagram::bind(&message_socket_addr)?;

    let data_socket_addr = match data_socket_addr {
        Some(data_socket_addr) => data_socket_addr,
        None => format!("/run/user/{}/replicest_server_data", user_id).parse().unwrap()
    };
    let _ = remove_file(&data_socket_addr);
    let data_socket = UnixListener::bind(&data_socket_addr)?;

    Ok((message_socket, data_socket))
}

fn trim_buffer(buffer: &[u8]) -> String {
    let message = String::from_utf8(buffer.to_vec()).unwrap_or("".to_string());
    let message = message.trim_end_matches(char::from(0));
    message.trim_end().to_string()
}

fn handle_message(message: String, analysis: &mut Analysis, data_socket: &UnixListener) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    match message.as_str() {
        str if str.starts_with("data") => handle_data_message(str, analysis, data_socket),
        "weights" => handle_weights_message(analysis, data_socket),
        str if str.starts_with("replicate weights") => handle_replicate_weights_message(str, analysis, data_socket),
        str if str.starts_with("set variance adjustment factor") => handle_set_variance_adjustment_factor_message(str, analysis),
        str @ ("frequencies" | "quantiles"  | "mean" | "correlation" | "linear regression") => handle_estimate_message(str, analysis),
        "calculate" => handle_calculate_message(analysis),
        _ => {
            Ok(vec!(b"unknown".into()))
        }
    }
}

fn handle_data_message(message: &str, analysis: &mut Analysis, data_socket: &UnixListener) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    let message_arguments = parse_data_message(&message);

    match message_arguments {
        None => {
            Ok(vec!(b"bad request - usage: data <number_imputations> <number_columns>".into()))
        }
        Some((number_imputations, number_columns)) => {
            let mut data : Vec<DMatrix<f64>> = Vec::new();

            for _ in 0..number_imputations {
                data.push(listen_for_data(data_socket, number_columns)?);
            }

            match number_imputations {
                1 => {
                    analysis.for_data(Imputation::No(&data[0]));
                }
                _ => {
                    let imp_data : Vec<&DMatrix<f64>> = Vec::from_iter(data.iter().map(|v| v));
                    analysis.for_data(Imputation::Yes(&imp_data));
                }
            }

            Ok(vec!(b"received data".into()))
        }
    }
}

fn parse_data_message(message: &str) -> Option<(usize, usize)> {
    let message_components : Vec<&str> = message.split(" ").collect();

    match message_components.as_slice() {
        [_, number_imputations, number_columns] if number_imputations.parse::<usize>().is_ok() && number_columns.parse::<usize>().is_ok() => {
            Some((number_imputations.parse::<usize>().unwrap(), number_columns.parse::<usize>().unwrap()))
        }
        _ => {
            None
        }
    }
}

fn handle_weights_message(analysis: &mut Analysis, data_socket: &UnixListener) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    let data = listen_for_data(data_socket, 1)?;
    let weight_vector : DVector<f64> = DVector::<f64>::from_iterator(data.nrows(), data.iter().map(|v| v.clone()));
    analysis.set_weights(&weight_vector);
    Ok(vec!(b"received weights".into()))
}

fn handle_replicate_weights_message(message: &str, analysis: &mut Analysis, data_socket: &UnixListener) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    let message_arguments = parse_replicate_weights_message(&message);

    match message_arguments {
        None => {
            Ok(vec!(b"bad request - usage: replicate weights <number_columns>".into()))
        }
        Some(number_columns) => {
            let replicate_weights = listen_for_data(data_socket, number_columns)?;
            analysis.with_replicate_weights(&replicate_weights);
            Ok(vec!(b"received replicate weights".into()))
        }
    }
}

fn parse_replicate_weights_message(message: &str) -> Option<usize> {
    let message_components : Vec<&str> = message.split(" ").collect();

    match message_components.as_slice() {
        [_, _, number_columns] if number_columns.parse::<usize>().is_ok() => {
            Some(number_columns.parse::<usize>().unwrap())
        }
        _ => {
            None
        }
    }
}

fn handle_set_variance_adjustment_factor_message(message: &str, analysis: &mut Analysis) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    let message_arguments = parse_set_variance_adjustment_factor_message(&message);

    match message_arguments {
        None => {
            Ok(vec!(b"bad request - usage: set variance adjustment factor <factor>".into()))
        }
        Some(factor) => {
            analysis.set_variance_adjustment_factor(factor);
            Ok(vec!(b"set variance adjustment factor".into()))
        }
    }
}

fn parse_set_variance_adjustment_factor_message(message: &str) -> Option<f64> {
    let message_components : Vec<&str> = message.split(" ").collect();

    match message_components.as_slice() {
        [_, _, _, _, factor] if factor.parse::<f64>().is_ok() => {
            Some(factor.parse::<f64>().unwrap())
        }
        _ => {
            None
        }
    }
}

fn handle_estimate_message(estimate: &str, analysis: &mut Analysis) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    match estimate {
        "frequencies" => { analysis.frequencies(); }
        "quantiles" => { analysis.quantiles(); }
        "mean" => { analysis.mean(); }
        "correlation" => { analysis.correlation(); }
        "linear regression" => { analysis.linreg(); }
        _ => { }
    }
    let mut return_message : Vec<u8> = b"set analysis to ".into();
    return_message.append(estimate.to_string().into_bytes().as_mut());
    Ok(vec!(return_message))
}

fn handle_calculate_message(analysis: &mut Analysis) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    let result = analysis.calculate();
    match result {
        Ok(result_data) => {
            let mut result_data_external : HashMap<Vec<String>, ReplicatedEstimates> = HashMap::new();
            for (key, value) in result_data.iter() {
                result_data_external.insert(key.clone(), ReplicatedEstimates::from_internal(value));
            }
            let serialization = rmp_serde::to_vec(&result_data_external);

            match serialization {
                Ok(serialized_data) => {
                    Ok(vec!(b"calculation complete".try_into().unwrap(), serialized_data))
                }
                Err(err) => {
                    Ok(vec!([b"error serializing calculation result: ", err.to_string().as_bytes()].concat().into()))
                }
            }
        }
        Err(err) => {
            Ok(vec!([b"error calculating: ", err.to_string().as_bytes()].concat().into()))
        }
    }
}

fn listen_for_data(data_socket: &UnixListener, columns: usize) -> Result<DMatrix<f64>, Box<dyn Error>> {
    match data_socket.accept() {
        Ok((mut socket, _)) => {
            let mut buffer = Vec::new();
            let _ = socket.read_to_end(&mut buffer)?;

            let data = u8_to_f64_vec(buffer, columns)?;
            let rows = data.len() / columns;

            Ok(DMatrix::from_vec(rows, columns, data))
        }
        Err(err) => {
            Err(Box::new(err))
        }
    }
}

fn u8_to_f64_vec(u8_data: Vec<u8>, columns: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    if u8_data.len() % (8 * columns) != 0 {
        return Err(Box::new(DataLengthError::new()));
    }
    let rows = u8_data.len() / (8 * columns);

    let mut data = Vec::new();

    for i in 0..columns * rows {
        let bytes : [u8; 8] = u8_data[i*8..(i + 1) * 8].try_into().unwrap();

        data.push(if cfg!(target_endian = "big") {
            f64::from_be_bytes(bytes)
        } else {
            f64::from_le_bytes(bytes)
        })
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use serial_test::serial;
    use std::fs::exists;
    use std::io::Write;
    use std::ops::Deref;
    use std::os::unix::net::UnixStream;
    use super::*;
    use std::thread;
    use std::time::Duration;
    use nalgebra::{dmatrix, dvector};

    #[test]
    #[serial]
    fn test_setup_default_sockets() {
        let user_id = get_current_uid();

        assert!(setup_sockets(None, None).is_ok());
        assert!(exists(format!("/run/user/{}/replicest_server", user_id)).unwrap_or(false));
        assert!(exists(format!("/run/user/{}/replicest_server_data", user_id)).unwrap_or(false));

        assert!(setup_sockets(None, None).is_ok());
    }

    #[test]
    #[serial]
    fn test_setup_custom_sockets() {
        let user_id = get_current_uid();

        assert!(setup_sockets(Some(format!("/run/user/{}/replicest_server_test", user_id).parse().unwrap()), Some(format!("/run/user/{}/replicest_server_data_test", user_id).parse().unwrap())).is_ok());
        assert!(exists(format!("/run/user/{}/replicest_server_test", user_id)).unwrap_or(false));
        assert!(exists(format!("/run/user/{}/replicest_server_data_test", user_id)).unwrap_or(false));
    }

    #[test]
    #[serial]
    fn test_message_socket_general_commands() {
        let client_addr = "/tmp/replicest_server_test_message_socket_general_commands".to_string();
        let _ = remove_file(&client_addr);
        let client = UnixDatagram::bind(&client_addr).unwrap();

        let handle = thread::spawn(|| {
            let return_value = main();
            assert!(return_value.is_ok());
        });

        thread::sleep(Duration::from_secs(1));

        let user_id = get_current_uid();
        let socket_addr = format!("/run/user/{}/replicest_server", user_id);
        client.connect(&socket_addr).unwrap();

        client.send(b"clear").unwrap();

        let mut buffer = [0; 1024];
        let _ = client.recv(&mut buffer);
        let message = trim_buffer(&buffer);

        assert_eq!("cleared", message);

        client.send(b"shutdown").unwrap();

        let mut buffer = [0; 1024];
        let _ = client.recv(&mut buffer);
        let message = trim_buffer(&buffer);

        assert_eq!("shutting down", message);

        handle.join().unwrap();
        let _ = remove_file(&client_addr);
    }

    #[test]
    fn test_u8_to_vec() {
        let result = u8_to_f64_vec(b"abcabcabcabcabcabcabcabc".try_into().unwrap(), 3);
        assert!(result.is_ok());

        let floats = vec![1.5, 2.0, -3.2, 14.44, -7.1, f64::NAN];

        let bytes = Vec::from_iter(floats.iter().map(|&v| f64::to_ne_bytes(v)));
        let bytes = Vec::from(bytes.as_flattened());

        let result = u8_to_f64_vec(bytes, 2).unwrap();

        for (i, &v) in floats.iter().enumerate() {
            if v.is_nan() {
                assert!(result[i].is_nan());
            } else {
                assert_eq!(v, result[i]);
            }
        }
    }

    #[test]
    fn test_u8_to_f64_vec_wrong_length() {
        let result = u8_to_f64_vec(b"abcdeabcdeabcdeabcdeabcde".try_into().unwrap(), 3);
        assert!(result.is_err());
        assert_eq!("Length of data was not a multiple of 8 * columns", result.err().unwrap().deref().to_string())
    }

    #[test]
    fn test_trim_buffer() {
        let mut buf = [0; 1024];
        buf[0] = 0x61;
        buf[1] = 0x62;
        buf[2] = 0x63;
        buf[3] = 0x20;
        let result = trim_buffer(&buf);

        assert_eq!("abc", result);
    }

    #[test]
    fn test_listen_for_data() {
        let data_socket_addr = "/tmp/replicest_server_test_listen_for_data".to_string();
        let _ = remove_file(&data_socket_addr);
        let data_socket = UnixListener::bind(&data_socket_addr).unwrap();

        let handle = thread::spawn(move || {
            let return_value = listen_for_data(&data_socket, 2);
            assert!(return_value.is_ok());

            let expected = dmatrix![
                1.5, 14.44;
                2.0, -7.1;
                -3.2, f64::NAN;
            ];

            let result = return_value.unwrap();

            assert_eq!(0,result.iter().enumerate().filter(|(i, &v)| (expected[(i % 3, i / 3)] - v).abs() > 1e-10).count())
        });

        thread::sleep(Duration::from_millis(200));

        let mut client = UnixStream::connect("/tmp/replicest_server_test_listen_for_data").unwrap();

        let floats = vec![1.5, 2.0, -3.2, 14.44, -7.1, f64::NAN];
        let bytes = Vec::from_iter(floats.iter().map(|&v| f64::to_ne_bytes(v)));
        let bytes = Vec::from(bytes.as_flattened());

        let _ = client.write_all(&bytes);

        drop(client);
        handle.join().unwrap();
    }

    #[test]
    fn test_listen_for_data_wrong_length() {
        let data_socket_addr = "/tmp/replicest_server_test_listen_for_data_wrong_length".to_string();
        let _ = remove_file(&data_socket_addr);
        let data_socket = UnixListener::bind(&data_socket_addr).unwrap();

        let handle = thread::spawn(move || {
            let return_value = listen_for_data(&data_socket, 10);
            assert!(return_value.is_err());
            assert_eq!("Length of data was not a multiple of 8 * columns", return_value.err().unwrap().deref().to_string());
        });

        thread::sleep(Duration::from_millis(200));

        let mut client = UnixStream::connect("/tmp/replicest_server_test_listen_for_data_wrong_length").unwrap();

        let floats = vec![1.5, 2.0, -3.2, 14.44, -7.1, f64::NAN];
        let bytes = Vec::from_iter(floats.iter().map(|&v| f64::to_ne_bytes(v)));
        let bytes = Vec::from(bytes.as_flattened());

        let _ = client.write_all(&bytes);

        drop(client);
        handle.join().unwrap();
    }

    #[test]
    fn test_handle_message_weights() {
        let data_socket_addr = "/tmp/replicest_server_test_handle_message_weights".to_string();
        let _ = remove_file(&data_socket_addr);
        let data_socket = UnixListener::bind(&data_socket_addr).unwrap();

        let handle = thread::spawn(move || {
            let mut current_analysis = analysis();
            let return_value = handle_message("weights".to_string(), &mut current_analysis, &data_socket);
            assert!(return_value.is_ok());
            assert_eq!(Vec::from(b"received weights"), return_value.unwrap()[0]);
            assert_eq!("none (no data; 6 weights of sum 30.540000000000003; no replicate weights)", current_analysis.summary());
        });

        thread::sleep(Duration::from_millis(200));

        let mut client = UnixStream::connect("/tmp/replicest_server_test_handle_message_weights").unwrap();

        let floats = vec![1.5, 2.0, 3.2, 14.44, 7.1, 2.3];
        let bytes = Vec::from_iter(floats.iter().map(|&v| f64::to_ne_bytes(v)));
        let bytes = Vec::from(bytes.as_flattened());

        let _ = client.write_all(&bytes);

        drop(client);
        handle.join().unwrap();
    }

    #[test]
    fn test_parse_data_message() {
        let wrong_message = "data";
        assert!(parse_data_message(wrong_message).is_none());

        let wrong_message = "data a 1";
        assert!(parse_data_message(wrong_message).is_none());

        let message = "data 5 15";
        let result = parse_data_message(message);

        assert!(result.is_some());
        assert_eq!((5, 15), result.unwrap());
    }

    #[test]
    fn test_parse_replicate_weights_message() {
        let wrong_message = "replicate weights";
        assert!(parse_replicate_weights_message(wrong_message).is_none());

        let wrong_message = "replicate weights abc";
        assert!(parse_replicate_weights_message(wrong_message).is_none());

        let message = "replicate weights 80";
        let result = parse_replicate_weights_message(message);

        assert!(result.is_some());
        assert_eq!(80, result.unwrap());
    }

    #[test]
    fn test_parse_set_variance_adjustment_factor_message() {
        let wrong_message = "set variance adjustment factor";
        assert!(parse_set_variance_adjustment_factor_message(wrong_message).is_none());

        let wrong_message = "set variance adjustment factor abc";
        assert!(parse_set_variance_adjustment_factor_message(wrong_message).is_none());

        let message = "set variance adjustment factor 0.25";
        let result = parse_set_variance_adjustment_factor_message(message);

        assert!(result.is_some());
        assert_eq!(0.25, result.unwrap());
    }

    #[test]
    fn test_handle_message_data_without_imputation() {
        let data_socket_addr = "/tmp/replicest_server_test_handle_message_data_without_imputation".to_string();
        let _ = remove_file(&data_socket_addr);
        let data_socket = UnixListener::bind(&data_socket_addr).unwrap();

        let handle = thread::spawn(move || {
            let mut current_analysis = analysis();
            let return_value = handle_message("data 1 3".to_string(), &mut current_analysis, &data_socket);
            assert!(return_value.is_ok());
            assert_eq!(Vec::from(b"received data"), return_value.unwrap()[0]);
            assert_eq!("none (1 datasets with 2 cases; wgt missing; no replicate weights)", current_analysis.summary());
        });

        thread::sleep(Duration::from_millis(200));

        let mut client = UnixStream::connect("/tmp/replicest_server_test_handle_message_data_without_imputation").unwrap();

        let floats = vec![1.5, 2.0, 3.2, 14.44, 7.1, 2.3];
        let bytes = Vec::from_iter(floats.iter().map(|&v| f64::to_ne_bytes(v)));
        let bytes = Vec::from(bytes.as_flattened());

        let _ = client.write_all(&bytes);

        drop(client);
        handle.join().unwrap();
    }

    #[test]
    fn test_handle_message_data_with_imputation() {
        let data_socket_addr = "/tmp/replicest_server_test_handle_message_data_with_imputation".to_string();
        let _ = remove_file(&data_socket_addr);
        let data_socket = UnixListener::bind(&data_socket_addr).unwrap();

        let handle = thread::spawn(move || {
            let mut current_analysis = analysis();
            let return_value = handle_message("data 2 3".to_string(), &mut current_analysis, &data_socket);
            assert!(return_value.is_ok());
            assert_eq!(Vec::from(b"received data"), return_value.unwrap()[0]);
            assert_eq!("none (2 datasets with 2 cases; wgt missing; no replicate weights)", current_analysis.summary());
        });

        thread::sleep(Duration::from_millis(200));

        for _ in 0..2 {
            let mut client = UnixStream::connect("/tmp/replicest_server_test_handle_message_data_with_imputation").unwrap();

            let floats = vec![1.5, 2.0, 3.2, 14.44, 7.1, 2.3];
            let bytes = Vec::from_iter(floats.iter().map(|&v| f64::to_ne_bytes(v)));
            let bytes = Vec::from(bytes.as_flattened());

            let _ = client.write_all(&bytes);

            drop(client);
        }

        handle.join().unwrap();
    }

    #[test]
    fn test_handle_message_replicate_weights_with_error() {
        let data_socket_addr = "/tmp/replicest_server_test_handle_message_replicate_weights_with_error".to_string();
        let _ = remove_file(&data_socket_addr);
        let data_socket = UnixListener::bind(&data_socket_addr).unwrap();

        let mut current_analysis = analysis();

        let return_value = handle_message("replicate weights x".to_string(), &mut current_analysis, &data_socket);

        assert!(return_value.is_ok());
        assert_eq!(Vec::from(b"bad request - usage: replicate weights <number_columns>"), return_value.unwrap()[0]);
    }

    #[test]
    fn test_handle_message_replicate_weights() {
        let data_socket_addr = "/tmp/replicest_server_test_handle_message_replicate_weights".to_string();
        let _ = remove_file(&data_socket_addr);
        let data_socket = UnixListener::bind(&data_socket_addr).unwrap();

        let handle = thread::spawn(move || {
            let mut current_analysis = analysis();
            let return_value = handle_message("replicate weights 3".to_string(), &mut current_analysis, &data_socket);
            assert!(return_value.is_ok());
            assert_eq!(Vec::from(b"received replicate weights"), return_value.unwrap()[0]);
            assert_eq!("none (no data; wgt missing; 3 replicate weights)", current_analysis.summary());
        });

        thread::sleep(Duration::from_millis(200));

        let mut client = UnixStream::connect("/tmp/replicest_server_test_handle_message_replicate_weights").unwrap();

        let floats = vec![1.5, 2.0, 3.2, 14.44, 7.1, 2.3];
        let bytes = Vec::from_iter(floats.iter().map(|&v| f64::to_ne_bytes(v)));
        let bytes = Vec::from(bytes.as_flattened());

        let _ = client.write_all(&bytes);

        drop(client);
        handle.join().unwrap();
    }

    #[test]
    fn test_handle_message_set_variance_adjustment_factor_with_error() {
        let data_socket_addr = "/tmp/replicest_server_test_handle_message_set_variance_adjustment_factor_with_error".to_string();
        let _ = remove_file(&data_socket_addr);
        let data_socket = UnixListener::bind(&data_socket_addr).unwrap();

        let mut current_analysis = analysis();

        let return_value = handle_message("set variance adjustment factor".to_string(), &mut current_analysis, &data_socket);

        assert!(return_value.is_ok());
        assert_eq!(Vec::from(b"bad request - usage: set variance adjustment factor <factor>"), return_value.unwrap()[0]);
    }

    #[test]
    fn test_handle_message_set_variance_adjustment_factor() {
        let data_socket_addr = "/tmp/replicest_server_test_handle_message_set_variance_adjustment_factor".to_string();
        let _ = remove_file(&data_socket_addr);
        let data_socket = UnixListener::bind(&data_socket_addr).unwrap();

        let mut current_analysis = analysis();
        current_analysis.with_replicate_weights(&dmatrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0;
        ]);

        let return_value = handle_message("set variance adjustment factor 0.5000".to_string(), &mut current_analysis, &data_socket);

        assert!(return_value.is_ok());
        assert_eq!(Vec::from(b"set variance adjustment factor"), return_value.unwrap()[0]);
        assert_eq!("none (no data; wgt missing; 3 replicate weights, factor 0.5)", current_analysis.summary());
    }

    #[test]
    fn test_handle_message_estimate() {
        let data_socket_addr = "/tmp/replicest_server_test_handle_message_sestimate".to_string();
        let _ = remove_file(&data_socket_addr);
        let data_socket = UnixListener::bind(&data_socket_addr).unwrap();

        let mut current_analysis = analysis();

        let return_value = handle_message("mean".to_string(), &mut current_analysis, &data_socket);

        assert!(return_value.is_ok());
        assert_eq!(Vec::from(b"set analysis to mean"), return_value.unwrap()[0]);
        assert_eq!("mean (no data; wgt missing; no replicate weights)", current_analysis.summary());

        let return_value = handle_message("linear regression".to_string(), &mut current_analysis, &data_socket);

        assert!(return_value.is_ok());
        assert_eq!(Vec::from(b"set analysis to linear regression"), return_value.unwrap()[0]);
        assert_eq!("linreg (no data; wgt missing; no replicate weights)", current_analysis.summary());
    }

    #[test]
    fn test_handle_message_calculate_with_error() {
        let data_socket_addr = "/tmp/replicest_server_test_handle_message_calculate_with_error".to_string();
        let _ = remove_file(&data_socket_addr);
        let data_socket = UnixListener::bind(&data_socket_addr).unwrap();

        let mut current_analysis = analysis();
        current_analysis.mean();

        let return_value = handle_message("calculate".to_string(), &mut current_analysis, &data_socket);

        assert!(return_value.is_ok());
        assert_eq!(Vec::from(b"error calculating: Analysis is missing some element: data"), return_value.unwrap()[0]);
    }

    #[test]
    fn test_handle_message_calculate() {
        let data_socket_addr = "/tmp/replicest_server_test_handle_message_calculate".to_string();
        let _ = remove_file(&data_socket_addr);
        let data_socket = UnixListener::bind(&data_socket_addr).unwrap();

        let mut imp_data: Vec<&DMatrix<f64>> = Vec::new();
        let data0 = DMatrix::from_row_slice(3, 4, &[
            1.0, 4.0, 2.5, -1.0,
            2.5, 1.75, 4.0, -2.5,
            3.0, 3.0, 1.0, -3.5,
        ]);
        imp_data.push(&data0);
        let data1 = DMatrix::from_row_slice(3, 4, &[
            1.2, 4.0, 2.5, -1.0,
            2.5, 1.75, 3.9, -2.5,
            2.7, 3.0, 1.0, -3.5,
        ]);
        imp_data.push(&data1);
        let data2 = DMatrix::from_row_slice(3, 4, &[
            0.8, 4.0, 2.5, -1.0,
            2.5, 1.75, 4.1, -2.5,
            3.3, 3.0, 1.0, -3.5,
        ]);
        imp_data.push(&data2);

        let wgt = dvector![1.0, 0.5, 1.5];

        let mut current_analysis = analysis();
        current_analysis.for_data(Imputation::Yes(&imp_data)).set_weights(&wgt).mean();

        let return_value = handle_message("calculate".to_string(), &mut current_analysis, &data_socket);

        assert!(return_value.is_ok());

        let responses = return_value.unwrap();
        assert_eq!(2, responses.len());
        assert_eq!(Vec::from(b"calculation complete"), responses[0]);

        let result_data = &responses[1];
        let result = rmp_serde::from_slice::<HashMap<Vec<String>, ReplicatedEstimates>>(result_data.as_slice());
        assert!(result.is_ok());

        let replicated_estimates = result.unwrap();
        assert_eq!(1, replicated_estimates.len());
        assert_eq!(&vec!("overall".to_string()), replicated_estimates.keys().next().unwrap());

        let overall_estimates = replicated_estimates.get(&vec!("overall".to_string())).unwrap();
        assert_eq!(4, overall_estimates.parameter_names.len());
        assert_eq!("mean_x2", overall_estimates.parameter_names[1]);

        let expected_final_estimates = vec![2.25, 3.125, 2.0, -2.5];
        let expected_imputation_variances = vec![0.0069444444444443955, 0.0, 0.0002777777777777758, 0.0];

        for (i, value) in expected_final_estimates.iter().enumerate() {
            assert!(overall_estimates.final_estimates[i] - value < 1e-10);
        }
        for (i, value) in expected_imputation_variances.iter().enumerate() {
            assert!(overall_estimates.imputation_variances[i] - value < 1e-10);
        }
    }
}