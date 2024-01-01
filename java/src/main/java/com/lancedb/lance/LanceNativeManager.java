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

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

public class LanceNativeManager {

  private static boolean loaded;

  static synchronized void loadLanceNative() {
    if (loaded) {
      return;
    }
    loadLibrary("lance_jni");
    loaded = true;
  }

  /**
   * Constructs the path to the native library based on the platform and library name.
   *
   * @param name The name of the library.
   * @return The full path to the library.
   */
  private static String getLibraryPath(String name)
  {
    return "/nativelib/" + getPlatform() + "/" + System.mapLibraryName(name);
  }

  /**
   * Loads the native library specified by name.
   * The method locates the library, copies it to a temporary file, and loads it.
   *
   * @param name The name of the library to load.
   * @throws RuntimeException if the library cannot be found or loaded.
   */
  private static void loadLibrary(String name)
  {
    String libraryPath = getLibraryPath(name);
    URL url = Dataset.class.getResource(libraryPath);
    if (url == null) {
      throw new RuntimeException("library not found: " + libraryPath);
    }

    File file;
    try {
      file = File.createTempFile(name, null);
      file.deleteOnExit();
      try (InputStream in = url.openStream()) {
        Files.copy(in, file.toPath(), StandardCopyOption.REPLACE_EXISTING);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    System.load(file.getAbsolutePath());
  }

  /**
   * Determines the platform-specific string based on OS name and architecture.
   *
   * @return A string representation of the platform, combining OS name and architecture.
   */
  private static String getPlatform()
  {
    String name = System.getProperty("os.name");
    String arch = System.getProperty("os.arch");
    return (name + "-" + arch).replace(' ', '_');
  }
}
