use std::error::Error;
use std::fs::remove_file;
use std::os::unix::net::{UnixDatagram};
use users::get_current_uid;
use replicest::analysis::*;

struct Command {
    command: String,
    data: Vec<f64>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let user_id = get_current_uid();
    let socket_addr = format!("/run/user/{}/replicest_server", user_id);
    let _ = remove_file(&socket_addr);

    let socket = UnixDatagram::bind(&socket_addr)?;

    let mut command = Command {
        command: "".to_string(),
        data: Vec::<f64>::new(),
    };
    let mut current_analysis = analysis();

    loop {
        let mut buffer = [0; 1024];

        break match socket.recv_from(&mut buffer) {
            Ok((_, client_addr)) => {
                let message = trim_buffer(&buffer);

                println!("Received: {}", message);

                if message == "shutdown" {
                    socket.send_to_addr(b"shutting down", &client_addr)?;
                } else if message == "clear" {
                    current_analysis = analysis();
                    command = Command {
                        command: "".to_string(),
                        data: Vec::<f64>::new(),
                    };
                    socket.send_to_addr(b"cleared", &client_addr)?;
                    continue;
                } else {
                    handle_message(&mut command, &mut current_analysis, message);
                    continue;
                }
            }
            Err(_) => { }
        }
    }

    Ok(())
}

fn trim_buffer(buffer: &[u8]) -> String {
    let message = String::from_utf8(buffer.to_vec()).unwrap_or("".to_string());
    let message = message.trim_end_matches(char::from(0));
    message.trim_end().to_string()
}

fn handle_message(command: &mut Command, analysis: &mut Analysis, message: String) {
    
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_unix_domain_socket() {
        let client_addr = "/tmp/replicest_server_test".to_string();
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
}