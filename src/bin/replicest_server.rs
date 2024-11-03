use std::error::Error;
use std::fs::remove_file;
use std::io::Read;
use std::os::unix::net::{UnixDatagram, UnixListener};
use nalgebra::{DMatrix, DVector};
use users::get_current_uid;
use replicest::analysis::*;
use replicest::errors::DataLengthError;

fn main() -> Result<(), Box<dyn Error>> {
    let (message_socket, data_socket) = setup_sockets()?;

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
                        Ok(response_data) => {
                            message_socket.send_to_addr(&response_data, &client_addr)?;
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

fn setup_sockets() -> Result<(UnixDatagram, UnixListener), Box<dyn Error>> {
    let user_id = get_current_uid();

    let message_socket_addr = format!("/run/user/{}/replicest_server", user_id);
    let _ = remove_file(&message_socket_addr);
    let message_socket = UnixDatagram::bind(&message_socket_addr)?;

    let data_socket_addr = format!("/run/user/{}/replicest_server_data", user_id);
    let _ = remove_file(&data_socket_addr);
    let data_socket = UnixListener::bind(&data_socket_addr)?;

    Ok((message_socket, data_socket))
}

fn trim_buffer(buffer: &[u8]) -> String {
    let message = String::from_utf8(buffer.to_vec()).unwrap_or("".to_string());
    let message = message.trim_end_matches(char::from(0));
    message.trim_end().to_string()
}

fn handle_message(message: String, analysis: &mut Analysis, data_socket: &UnixListener) -> Result<Vec<u8>, Box<dyn Error>> {
    match message.as_str() {
        str if str.starts_with("data") => {
            todo!();
            Ok(b"received data".try_into().unwrap())
        }
        "weights" => {
            let data = listen_for_data(data_socket, 1)?;
            let weight_vector : DVector<f64> = DVector::<f64>::from_iterator(data.nrows(), data.iter().map(|v| v.clone()));
            analysis.set_wgts(&weight_vector);
            Ok(b"received weights".try_into().unwrap())
        }
        "mean" => {
            analysis.mean();
            Ok(b"set analysis to mean".try_into().unwrap())
        }
        "calculate" => {
            todo!();
            Ok(b"".try_into().unwrap())
        }
        _ => {
            Ok(b"unknown".try_into().unwrap())
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
    use nalgebra::{dmatrix};

    #[test]
    #[serial]
    fn test_setup_sockets() {
        let user_id = get_current_uid();

        assert!(setup_sockets().is_ok());
        assert!(exists(format!("/run/user/{}/replicest_server", user_id)).unwrap_or(false));
        assert!(exists(format!("/run/user/{}/replicest_server_data", user_id)).unwrap_or(false));

        assert!(setup_sockets().is_ok());
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
            assert_eq!(Vec::from(b"received weights"), return_value.unwrap());
            assert_eq!("none (no data; 6 weights of sum 30.540000000000003)", current_analysis.summary());
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
}