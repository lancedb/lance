// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use clap::Parser;
use lance_tools::cli::LanceToolsArgs;
use std::io::{Error, ErrorKind};

#[tokio::main]
pub async fn main() -> Result<(), std::io::Error> {
    // Install global panic handler
    install_panic_handler();

    // Parse arguments from command line
    let args = LanceToolsArgs::parse();

    // Run with the parsed arguments
    return lance_result_to_std_result(args.run(&mut std::io::stdout()).await);
}

fn lance_result_to_std_result<T>(lance_result: lance_core::Result<T>) -> Result<T, std::io::Error> {
    return match lance_result {
        Ok(t) => Result::Ok(t),
        Err(e) => Result::Err(Error::new(ErrorKind::Other, e.to_string())),
    };
}

/// Install custom panic handler for better error reporting
fn install_panic_handler() {
    std::panic::set_hook(Box::new(|panic_info| {
        let msg = match panic_info.payload().downcast_ref::<&str>() {
            Some(s) => *s,
            None => match panic_info.payload().downcast_ref::<String>() {
                Some(s) => s,
                None => "Unknown panic",
            },
        };

        let location = if let Some(location) = panic_info.location() {
            format!(
                " at {}:{}:{}",
                location.file(),
                location.line(),
                location.column()
            )
        } else {
            String::new()
        };

        eprintln!("\n\x1b[31mPANIC{}: {}\x1b[0m", location, msg);

        // Print backtrace if available
        if let Ok(var) = std::env::var("RUST_BACKTRACE") {
            if var != "0" {
                eprintln!(
                    "\nBacktrace:\n{:?}",
                    std::backtrace::Backtrace::force_capture()
                );
            }
        }
    }));
}
