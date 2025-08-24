// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use clap::Args;
use lance_core::Result as LanceResult;
use lance_file::v2::reader::CachedFileMetadata;
use lance_file::v2::reader::FileReader;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::ScanScheduler;
use lance_io::scheduler::SchedulerConfig;
use lance_io::utils::CachedFileSize;
use object_store::path::Path;
use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

#[derive(Args, Debug)]
pub struct MetaArgs {
    // The source file to examine.
    #[arg(short = 's', long, value_name = "source")]
    source: String,
}

    //#[arg(long)]
pub struct LanceToolFileMetadata {
    file_metadata: CachedFileMetadata,
}

impl fmt::Display for LanceToolFileMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.file_metadata.file_schema)?;
        writeln!(f, "{}", self.file_metadata.column_metadatas
        return Ok(());
    }
}

async fn get_object_store_and_path(source: &String) -> LanceResult<(Arc<ObjectStore>, Path)> {
    let path = Path::parse(source)?;
    return Ok((Arc::new(ObjectStore::local()),  path));
}

impl LanceToolFileMetadata {
    async fn open(
        source: &String,
    ) -> LanceResult<Self> {
        let (object_store, path) = get_object_store_and_path(source).await?;
        let scan_scheduler = ScanScheduler::new(
            object_store,
            SchedulerConfig {
                io_buffer_size_bytes: 2 * 1024 * 1024 * 1024,
            },
        );
        let file_scheduler = scan_scheduler.open_file(&path,  &CachedFileSize::unknown()).await?;
        let file_metadata = FileReader::read_all_metadata(&file_scheduler).await?;
        let lance_tool_file_metadata = LanceToolFileMetadata {
            file_metadata,
         };
        return Ok(lance_tool_file_metadata);
    }
}

pub async fn run(args: MetaArgs) -> Result<(), std::io::Error> {
    let metadata =
        crate::cli::lance_result_to_std_result(LanceToolFileMetadata::open(&args.source).await)?;
    print!("{}", metadata.to_string());
    return Ok(());
}
