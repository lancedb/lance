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

/**
 * Utility class to load the native library.
 */
public class JniLoader {
  static {
    JarJniLoader.loadLib(Dataset.class, "/nativelib", "lance_jni");
  }

  /**
   * Ensures the native library is loaded.
   * This method will trigger the static initializer
   */
  public static void ensureLoaded() {}

  private JniLoader() {}
}