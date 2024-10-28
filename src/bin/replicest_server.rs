use std::error::Error;
use std::fs::remove_file;
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener};
use users::get_current_uid;

fn main() -> Result<(), Box<dyn Error>> {
    let user_id = get_current_uid();
    let socket_addr = format!("/run/user/{}/replicest_server", user_id);
    let _ = remove_file(&socket_addr);

    let listener = UnixListener::bind(&socket_addr)?;
    let (mut client, _) = listener.accept()?;

    loop {
        let mut buffer = [0; 1024];

        break match client.read(&mut buffer) {
            Ok(_) => {
                let message = String::from_utf8(buffer.to_vec()).unwrap_or("".to_string());
                let message = message.trim_end_matches(char::from(0));
                let message = message.trim_end();

                println!("Received: {}", message);
                if message == "shutdown" {
                    client.write_all(b"shutting down")?;
                } else {
                    continue;
                }
            }
            Err(_) => { }
        }
    }

    remove_file(&socket_addr)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixStream;
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_unix_domain_socket() {
        let handle = thread::spawn(|| {
            let return_value = main();
            assert!(return_value.is_ok());
        });

        thread::sleep(Duration::from_secs(1));

        let user_id = get_current_uid();
        let socket_addr = format!("/run/user/{}/replicest_server", user_id);
        let mut client = UnixStream::connect(&socket_addr).unwrap();

        let shutdown_message = b"shutdown                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ";
        client.write(shutdown_message).unwrap();

        let mut buffer = [0; 1024];
        let _ = client.read(&mut buffer);
        let message = String::from_utf8(buffer.to_vec()).unwrap();
        let message = message.trim_end_matches(char::from(0));
        let message = message.trim_end();

        assert_eq!("shutting down", message);

        handle.join().unwrap();
    }
}