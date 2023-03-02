use jni::JNIEnv;

// These objects are what you should use as arguments to your native
// function. They carry extra lifetime information to prevent them escaping
// this context and getting used after being GC'd.
use jni::objects::{JClass, JLongArray, JString};

// This is just a pointer. We'll be returning it from our function. We
// can't return one of the objects with lifetime information because the
// lifetime checker won't let us.
use jni::sys::{jclass, jlong, jlongArray, jobjectArray, jstring};

#[no_mangle]
pub extern "system" fn Java_lance_JNI_saveToLance<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    path: JString<'local>,
    vec: jclass,
    allocator: jclass) {
    let path: String = env
        .get_string(&path)
        .expect("Couldn't get java string!")
        .into();

    println!("jenv {:?}", env);
    println!("class {:?}", _class);
    println!("path {:?}", path);
    println!("vec {:?}", vec);
    println!("allocator {:?}", allocator)
}