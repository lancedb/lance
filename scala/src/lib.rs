use std::mem::ManuallyDrop;
use arrow::array::{ArrayData, Int32Array};
use arrow::ffi::{ArrowArray, FFI_ArrowArray, FFI_ArrowSchema};
use jni::JNIEnv;

// These objects are what you should use as arguments to your native
// function. They carry extra lifetime information to prevent them escaping
// this context and getting used after being GC'd.
use jni::objects::{JClass, JLongArray, JObject, JString, JValue, JValueGen};

// This is just a pointer. We'll be returning it from our function. We
// can't return one of the objects with lifetime information because the
// lifetime checker won't let us.
use jni::sys::{jclass, jlong, jlongArray, jobjectArray, jstring};

#[no_mangle]
pub extern "system" fn Java_lance_JNI_saveToLance<'local>(
    mut env: JNIEnv<'local>,
    class: JClass<'local>,
    path: JString<'local>,
    vec: JObject,
    allocator: JObject) {
    // references:
    // https://github.com/apache/arrow-rs/blob/231ae9b31769b62da368b9f1eb355a840540cb06/arrow/src/ffi.rs#L567
    // https://arrow.apache.org/docs/java/cdata.html#java-to-c
    let path: String = env
        .get_string(&path)
        .expect("Couldn't get java string!")
        .into();
    let schema = FFI_ArrowSchema::empty();
    let array = FFI_ArrowArray::empty();

    let schema = Box::new(ManuallyDrop::new(schema));
    let array = Box::new(ManuallyDrop::new(array));

    let schema_ptr = &**schema as *const FFI_ArrowSchema;
    let array_ptr = &**array as *const FFI_ArrowArray;

    let schema_jlong = schema_ptr as i64;
    let array_jlong = array_ptr as i64;

    println!("jenv {:?}", env);
    println!("class {:?}", &class);
    println!("path {:?}", path);
    println!("vec {:?}", vec);
    println!("allocator {:?}", allocator);
    //javap -s
    let result = env.call_static_method(class, "fillVector",
                                        "(JJLorg/apache/arrow/vector/FieldVector;Lorg/apache/arrow/memory/BufferAllocator;)V",
                                        &[schema_jlong.into(), array_jlong.into(), JValue::from(&vec), JValue::from(&allocator)]);
    match result {
        Ok(_) => {
            println!("OK!");
            let array = unsafe {
                ArrowArray::new(std::ptr::read(array_ptr), std::ptr::read(schema_ptr))
            };
            let array = Int32Array::from(ArrayData::try_from(array).unwrap());
            println!("array: {:?}", array)
        }
        Err(e) => {
            println!("error: {:?}", e)
        }
    }
}