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
import java.io.Closeable;
import java.util.List;
import java.util.Map;
import org.apache.arrow.c.ArrowArrayStream;

/**
 * Class representing a Lance dataset, interfacing with the native lance library.
 * This class provides functionality to open and manage datasets with native code.
 * The native library is loaded statically and utilized through native methods.
 * It implements the {@link java.io.Closeable} interface to ensure proper resource management.
 */
public class Dataset implements Closeable {
  static {
    JarJniLoader.loadLib(Dataset.class, "/nativelib", "lance_jni");
  }

  private long nativeDatasetHandle;

  private Dataset() {
  }

  public static Dataset write(ArrowArrayStream stream, String path, WriteParams params) {
    return writeWithFfiStream(stream.memoryAddress(), path, params.toMap());
  }

  private static native Dataset writeWithFfiStream(long arrowStreamMemoryAddress, String path,
      Map<String, Object> params);

  /**
   * Opens a dataset from the specified path using the native library.
   *
   * @param path The file path of the dataset to open.
   * @return A new instance of {@link Dataset} linked to the opened dataset.
   */
  public static native Dataset open(String path);

  /**
   * Count the number of rows in the dataset.
   *
   * @return num of rows.
   */
  public native int countRows();

  /**
   * Get all fragments in this dataset.
   *
   * @return A list of {@link Fragment}.
   */
  public List<Fragment> getFragments() {
    return  this.getFragmentsNative().stream().peek(f -> f.dataset = this).toList();
  }

  private native List<Fragment> getFragmentsNative();

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
