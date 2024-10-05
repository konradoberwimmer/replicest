pub fn to_upper_case(text: &str) -> String {
    text.to_uppercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_upper_case() {
        let result = to_upper_case("Welcome to Avalonia!");

        assert_eq!("WELCOME TO AVALONIA!", result);
    }
}