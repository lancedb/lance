// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use arrow_schema::DataType;
use futures::future::BoxFuture;
use futures::FutureExt;
use lance_core::{
    datatypes::{Field, Schema},
    Result,
};
use lance_encoding::{
    decoder::{ColumnInfo, DecoderMiddlewareChainCursor, FieldDecoderStrategy, FieldScheduler},
    encoder::{ColumnIndexSequence, CoreFieldEncodingStrategy, FieldEncodingStrategy},
    encodings::physical::FileBuffers,
};
use zone::{extract_zone_info, UnloadedPushdown, ZoneMapsFieldEncoder, ZoneMapsFieldScheduler};

pub mod format;
pub mod substrait;
pub mod zone;

#[derive(Debug)]
struct LanceDfFieldDecoderState {
    /// We assume that all columns have the same number of rows per map
    rows_per_map: Option<u32>,
    /// As we visit the decoding tree we populate this with the pushdown
    /// information that is available.
    zone_map_buffers: Vec<UnloadedPushdown>,
}

/// This strategy is responsible for creating the field scheduler
/// that handles the pushdown filtering.  It is a top-level scheduler
/// that uses column info from various leaf schedulers.
///
/// The current implementation is a bit of a hack.  It assumes that
/// the decoder strategy will only be used once.  The very first time
/// that create_field_scheduler is called, we assume we are at the root.
///
/// Field decoding strategies are supposed to be stateless but this one
/// is not.  As a result, we use a mutex to gather the state even though
/// we aren't technically doing any concurrency.
#[derive(Debug)]
pub struct LanceDfFieldDecoderStrategy {
    state: Arc<Mutex<Option<LanceDfFieldDecoderState>>>,
    schema: Arc<Schema>,
}

impl LanceDfFieldDecoderStrategy {
    pub fn new(schema: Arc<Schema>) -> Self {
        Self {
            state: Arc::new(Mutex::new(None)),
            schema,
        }
    }

    fn initialize(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        if state.is_none() {
            *state = Some(LanceDfFieldDecoderState {
                rows_per_map: None,
                zone_map_buffers: Vec::new(),
            });
            true
        } else {
            false
        }
    }

    fn add_pushdown_field(&self, rows_per_map: u32, unloaded_pushdown: UnloadedPushdown) {
        let mut state = self.state.lock().unwrap();
        let state = state.as_mut().unwrap();
        match state.rows_per_map {
            Some(existing) if existing != rows_per_map => {
                panic!("Inconsistent rows per map");
            }
            _ => {
                state.rows_per_map = Some(rows_per_map);
            }
        }
        state.zone_map_buffers.push(unloaded_pushdown);
    }
}

impl FieldDecoderStrategy for LanceDfFieldDecoderStrategy {
    fn create_field_scheduler<'a>(
        &self,
        field: &Field,
        column_infos: &mut VecDeque<ColumnInfo>,
        buffers: FileBuffers,
        chain: DecoderMiddlewareChainCursor<'a>,
    ) -> Result<(
        DecoderMiddlewareChainCursor<'a>,
        BoxFuture<'static, Result<Arc<dyn FieldScheduler>>>,
    )> {
        let is_root = self.initialize();

        if let Some((rows_per_map, unloaded_pushdown)) = extract_zone_info(
            column_infos.front_mut().unwrap(),
            &field.data_type(),
            chain.current_path(),
        ) {
            // If there is pushdown info then record it and unwrap the
            // pushdown encoding layer.
            self.add_pushdown_field(rows_per_map, unloaded_pushdown);
        }
        // Delegate to the rest of the chain to create the decoder
        let (chain, next) = chain.next(field, column_infos, buffers)?;

        // If this is the top level decoder then wrap it with our
        // pushdown filtering scheduler.
        let state = if is_root {
            self.state.lock().unwrap().take()
        } else {
            None
        };
        let schema = self.schema.clone();
        let io = chain.io().clone();

        let scheduler_fut = async move {
            let next = next.await?;
            if is_root {
                let state = state.unwrap();
                let rows_per_map = state.rows_per_map;
                let zone_map_buffers = state.zone_map_buffers;
                let num_rows = next.num_rows();
                if rows_per_map.is_none() {
                    // No columns had any pushdown info
                    Ok(next)
                } else {
                    let mut scheduler = ZoneMapsFieldScheduler::new(
                        next,
                        schema,
                        zone_map_buffers,
                        rows_per_map.unwrap(),
                        num_rows,
                    );
                    // Load all the zone maps from disk
                    // TODO: it would be slightly more efficient to do this
                    // later when we know what columns are actually used
                    // for filtering.
                    scheduler.initialize(io.as_ref()).await?;
                    Ok(Arc::new(scheduler) as Arc<dyn FieldScheduler>)
                }
            } else {
                Ok(next)
            }
        }
        .boxed();
        Ok((chain, scheduler_fut))
    }
}

/// Wraps the core encoding strategy and adds the encoders from this
/// crate
#[derive(Debug)]
pub struct LanceDfFieldEncodingStrategy {
    core: CoreFieldEncodingStrategy,
    rows_per_map: u32,
}

impl Default for LanceDfFieldEncodingStrategy {
    fn default() -> Self {
        Self {
            core: CoreFieldEncodingStrategy::default(),
            rows_per_map: 10000,
        }
    }
}

impl FieldEncodingStrategy for LanceDfFieldEncodingStrategy {
    fn create_field_encoder(
        &self,
        encoding_strategy_root: &dyn FieldEncodingStrategy,
        field: &lance_core::datatypes::Field,
        column_index: &mut ColumnIndexSequence,
        cache_bytes_per_column: u64,
        keep_original_array: bool,
        config: &std::collections::HashMap<String, String>,
    ) -> lance_core::Result<Box<dyn lance_encoding::encoder::FieldEncoder>> {
        let data_type = field.data_type();
        if data_type.is_primitive()
            || matches!(
                data_type,
                DataType::Boolean | DataType::Utf8 | DataType::LargeUtf8
            )
        {
            let inner_encoder = self.core.create_field_encoder(
                // Don't collect stats on inner string fields
                &self.core,
                field,
                column_index,
                cache_bytes_per_column,
                keep_original_array,
                config,
            )?;
            Ok(Box::new(ZoneMapsFieldEncoder::try_new(
                inner_encoder,
                data_type.clone(),
                self.rows_per_map,
            )?))
        } else {
            self.core.create_field_encoder(
                encoding_strategy_root,
                field,
                column_index,
                cache_bytes_per_column,
                keep_original_array,
                config,
            )
        }
    }
}
