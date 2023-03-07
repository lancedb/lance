use arrow::array::{make_array, ArrayData, Int32Array};
use arrow::ffi::{ArrowArray, FFI_ArrowArray, FFI_ArrowSchema};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow::ipc::RecordBatch;
use arrow::record_batch::RecordBatchReader;
use jni::objects::{JClass, JLongArray, JObject, JString, JValue, JValueGen};
use jni::JNIEnv;
use std::alloc::alloc;
use std::fs::read;
use std::mem::ManuallyDrop;

use futures::executor;
use jni::sys::{jclass, jlong, jlongArray, jobjectArray, jstring};
use lance::dataset::{
    scanner::Scanner as LanceScanner, Dataset as LanceDataset, Version, WriteMode, WriteParams,
};

#[no_mangle]
pub extern "system" fn Java_lance_JNI_saveStreamToLance<'local>(
    mut env: JNIEnv<'local>,
    class: JClass<'local>,
    path: JString<'local>,
    reader: JObject,
    allocator: JObject,
) {
    let path: String = env
        .get_string(&path)
        .expect("Couldn't get java string!")
        .into();
    let stream = FFI_ArrowArrayStream::empty();
    let stream = Box::new(stream);
    let stream_ptr = Box::into_raw(stream);
    let stream_ptr_long = stream_ptr as i64;
    let result = env.call_static_method(
        class,
        "fillStream",
        "(JLorg/apache/arrow/vector/ipc/ArrowReader;Lorg/apache/arrow/memory/BufferAllocator;)V",
        &[
            JValue::from(stream_ptr_long),
            JValue::from(&reader),
            JValue::from(&allocator),
        ],
    );
    match result {
        Ok(_) => {
            println!("java call rust call java via JNI done");
            let reader = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) };
            match reader {
                Ok(reader) => {
                    println!("got reader");
                    let mut batch_reader: Box<dyn RecordBatchReader> = Box::new(reader);
                    // let dataset = Dataset::write(&mut reader, test_uri, None).await.unwrap();
                    println!("save to path {}", path);
                    let result = async {
                        LanceDataset::write(&mut batch_reader, path.as_str(), None)
                            .await
                            .unwrap();
                    };
                    executor::block_on(result);
                }
                Err(_) => {}
            }
        }
        Err(e) => {
            println!("error: {:?}", e)
        }
    }
}
