// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use clap::Parser;
use lance_tools::cli::LanceToolsArgs;
use lance_tools::util::install_panic_handler;

#[tokio::main]
pub async fn main() -> Result<(), lance_core::Error> {
    // Install global panic handler
    install_panic_handler();

    // Parse arguments from command line
    let args = LanceToolsArgs::parse();

    // Run with the parsed arguments
    lance_result_to_std_result(args.run(&mut std::io::stdout()).await)
}

fn lance_result_to_std_result<T>(
    lance_result: lance_core::Result<T>,
) -> Result<T, lance_core::Error> {
    match lance_result {
        Ok(t) => Result::Ok(t),
        Err(e) => Result::Err(e),
    }
}

#[cfg(test)]
mod tests {
    use crate::lance_result_to_std_result;
    use snafu::location;

    #[test]
    fn test_ok_lance_result_to_ok_std_result() {
        assert!(lance_result_to_std_result(Ok(())).is_ok());
    }

    #[test]
    fn test_error_lance_result_to_error_std_result() {
        assert!(
            lance_result_to_std_result::<()>(Err(lance_core::Error::invalid_input(
                "bad input",
                location!()
            )))
            .is_err()
        );
    }
}
