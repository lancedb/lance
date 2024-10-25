// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use arrow_schema::DataType;
use lance_core::datatypes::{Field, Schema};
use lance_encoding::encoder::{
    default_encoding_strategy, ColumnIndexSequence, EncodingOptions, FieldEncodingStrategy,
};
use lance_file::version::LanceFileVersion;
use zone::{UnloadedPushdown, ZoneMapsFieldEncoder};

pub mod format;
pub mod substrait;
pub mod zone;

#[derive(Debug)]
struct LanceDfFieldDecoderState {
    /// We assume that all columns have the same number of rows per map
    #[allow(unused)]
    rows_per_map: Option<u32>,
    /// As we visit the decoding tree we populate this with the pushdown
    /// information that is available.
    #[allow(unused)]
    zone_map_buffers: HashMap<u32, UnloadedPushdown>,
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
    #[allow(unused)]
    state: Arc<Mutex<Option<LanceDfFieldDecoderState>>>,
    #[allow(unused)]
    schema: Arc<Schema>,
}

impl LanceDfFieldDecoderStrategy {
    pub fn new(schema: Arc<Schema>) -> Self {
        Self {
            state: Arc::new(Mutex::new(None)),
            schema,
        }
    }

    #[allow(unused)]
    fn initialize(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        if state.is_none() {
            *state = Some(LanceDfFieldDecoderState {
                rows_per_map: None,
                zone_map_buffers: HashMap::new(),
            });
            true
        } else {
            false
        }
    }

    #[allow(unused)]
    fn add_pushdown_field(
        &self,
        field: &Field,
        rows_per_map: u32,
        unloaded_pushdown: UnloadedPushdown,
    ) {
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
        state
            .zone_map_buffers
            .insert(field.id as u32, unloaded_pushdown);
    }
}

// TODO: Reconnect...again
// impl FieldDecoderStrategy for LanceDfFieldDecoderStrategy {
//     fn create_field_scheduler<'a>(
//         &self,
//         field: &Field,
//         column_infos: &mut ColumnInfoIter,
//         buffers: FileBuffers,
//         chain: DecoderMiddlewareChainCursor<'a>,
//     ) -> Result<(
//         DecoderMiddlewareChainCursor<'a>,
//         Result<Box<dyn FieldScheduler>>,
//     )> {
//         let is_root = self.initialize();

//         if let Some((rows_per_map, unloaded_pushdown)) =
//             extract_zone_info(column_infos, &field.data_type(), chain.current_path())
//         {
//             // If there is pushdown info then record it and unwrap the
//             // pushdown encoding layer.
//             self.add_pushdown_field(field, rows_per_map, unloaded_pushdown);
//         }
//         // Delegate to the rest of the chain to create the decoder
//         let (chain, next) = chain.next(field, column_infos, buffers)?;

//         // If this is the top level decoder then wrap it with our
//         // pushdown filtering scheduler.
//         if is_root {
//             let state = self.state.lock().unwrap().take().unwrap();
//             let schema = self.schema.clone();
//             let rows_per_map = state.rows_per_map;
//             let zone_map_buffers = state.zone_map_buffers;
//             let next = next?;
//             let num_rows = next.num_rows();
//             if rows_per_map.is_none() {
//                 // No columns had any pushdown info
//                 Ok((chain, Ok(next)))
//             } else {
//                 let scheduler = ZoneMapsFieldScheduler::new(
//                     next.into(),
//                     schema,
//                     zone_map_buffers,
//                     rows_per_map.unwrap(),
//                     num_rows,
//                 );
//                 Ok((chain, Ok(Box::new(scheduler))))
//             }
//         } else {
//             Ok((chain, next))
//         }
//     }
// }

/// Wraps the core encoding strategy and adds the encoders from this
/// crate
#[derive(Debug)]
pub struct LanceDfFieldEncodingStrategy {
    inner: Box<dyn FieldEncodingStrategy>,
    rows_per_map: u32,
}

impl Default for LanceDfFieldEncodingStrategy {
    fn default() -> Self {
        Self {
            inner: default_encoding_strategy(LanceFileVersion::default()),
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
        options: &EncodingOptions,
    ) -> lance_core::Result<Box<dyn lance_encoding::encoder::FieldEncoder>> {
        let data_type = field.data_type();
        if data_type.is_primitive()
            || matches!(
                data_type,
                DataType::Boolean | DataType::Utf8 | DataType::LargeUtf8
            )
        {
            let inner_encoder = self.inner.create_field_encoder(
                // Don't collect stats on inner string fields
                self.inner.as_ref(),
                field,
                column_index,
                options,
            )?;
            Ok(Box::new(ZoneMapsFieldEncoder::try_new(
                inner_encoder,
                data_type.clone(),
                self.rows_per_map,
            )?))
        } else {
            self.inner
                .create_field_encoder(encoding_strategy_root, field, column_index, options)
        }
    }
}
