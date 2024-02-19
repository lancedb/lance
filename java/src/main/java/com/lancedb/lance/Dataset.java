/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.lancedb.lance;

import io.questdb.jar.jni.JarJniLoader;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.jni.JniLoader;

import java.io.Closeable;
import java.util.Map;

/**
 * Class representing a Lance dataset, interfacing with the native lance library.
 * This class provides functionality to open and manage datasets with native code.
 *
 * The native library is loaded statically and utilized through native methods.
 * It implements the {@link java.io.Closeable} interface to ensure proper resource management.
 */
public class Dataset implements Closeable {
  private long nativeDatasetHandle;

  static {
    JarJniLoader.loadLib(Dataset.class, "/nativelib", "lance_jni");
  }

  private Dataset() {
  }

  public static Dataset write(ArrowArrayStream stream, String path, WriteParams params) {
    return writeWithFFIStream(stream.memoryAddress(), path, params.toMap());
  }

  private static native Dataset writeWithFFIStream(long ArrowStreamMemoryAddress, String path,
      Map<String, Object> params);

  /**
   * Opens a dataset from the specified path using the native library.
   *
   * @param path The file path of the dataset to open.
   * @return A new instance of {@link Dataset} linked to the opened dataset.
   */
  public native static Dataset open(String path);

  /**
   * Count the number of rows in the dataset.
   * @return num of rows
   */
  public native int countRows();

  /**
   * Closes this dataset and releases any system resources associated with it.
   * If the dataset is already closed, then invoking this method has no effect.
   */
  @Override
  public void close() {
    if (nativeDatasetHandle != 0) {
      releaseNativeDataset(nativeDatasetHandle);
      nativeDatasetHandle = 0;
    }
  }

  /**
   * Native method to release the Lance dataset resources associated with the given handle.
   *
   * @param handle The native handle to the dataset resource.
   */
  private native void releaseNativeDataset(long handle);
}
