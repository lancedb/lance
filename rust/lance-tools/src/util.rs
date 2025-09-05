// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::cli::LanceToolsArgs;
use clap::Parser;
use lance_core::{Error, Result};
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use object_store::path::Path;
use snafu::location;
use std::sync::Arc;
use url::Url;

pub async fn run_cli(args: Vec<String>) {
    // Install global panic handler
    install_panic_handler();

    // Parse arguments from command line
    let args = match LanceToolsArgs::try_parse_from(&args) {
        Ok(args) => args,
        Err(err) => {
            eprint!("{}", err);
            return;
        }
    };

    // Run with the parsed arguments
    match args.run(&mut std::io::stdout()).await {
        Ok(_) => {}
        Err(e) => {
            eprint!("{}", e);
        }
    }
}

/// Install custom panic handler for better error reporting
pub fn install_panic_handler() {
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

fn path_to_parent(path: &Path) -> Result<(Path, String)> {
    let mut parts = path.parts().collect::<Vec<_>>();
    if parts.is_empty() {
        return Err(Error::invalid_input(
            format!("Path {} is not a valid path to a file", path),
            location!(),
        ));
    }
    let filename = parts.pop().unwrap().as_ref().to_owned();
    Ok((Path::from_iter(parts), filename))
}

/// Get an object store and a path from a source string.
pub(crate) async fn get_object_store_and_path(source: &String) -> Result<(Arc<ObjectStore>, Path)> {
    if let Ok(mut url) = Url::parse(source) {
        if url.scheme().len() > 1 {
            let path = object_store::path::Path::parse(url.path()).map_err(Error::from)?;
            let (parent_path, filename) = path_to_parent(&path)?;
            url.set_path(parent_path.as_ref());
            let object_store_registry = Arc::new(ObjectStoreRegistry::default());
            let object_store_params = ObjectStoreParams::default();
            let (object_store, dir_path) = ObjectStore::from_uri_and_params(
                object_store_registry,
                url.as_str(),
                &object_store_params,
            )
            .await?;
            let child_path = dir_path.child(filename);
            return Ok((object_store, child_path));
        }
    }
    let path = Path::from_filesystem_path(source)?;
    let object_store = Arc::new(ObjectStore::local());
    Ok((object_store, path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_to_parent() {
        let (parent_path, filename) =
            path_to_parent(&object_store::path::Path::parse("/a/b/c").unwrap()).unwrap();
        assert_eq!("c", filename);
        let parts: Vec<_> = parent_path.parts().collect();
        assert_eq!(2, parts.len());
        assert_eq!("a", parts.first().unwrap().as_ref());
        assert_eq!("b", parts.get(1).unwrap().as_ref());
    }
}
