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
package org.lance;

/** Constants for the Lance SDK. */
public final class LanceConstants {

  private LanceConstants() {}

  /** Legacy file format version (0.1). */
  public static final String FILE_FORMAT_VERSION_LEGACY = "legacy";

  /** Stable file format version (resolves to latest stable). */
  public static final String FILE_FORMAT_VERSION_STABLE = "stable";

  /** Next file format version (resolves to latest). */
  public static final String FILE_FORMAT_VERSION_NEXT = "next";

  /** File format version 0.1. */
  public static final String FILE_FORMAT_VERSION_0_1 = "0.1";

  /** File format version 2.0. */
  public static final String FILE_FORMAT_VERSION_2_0 = "2.0";

  /** File format version 2.1. */
  public static final String FILE_FORMAT_VERSION_2_1 = "2.1";

  /** File format version 2.2. */
  public static final String FILE_FORMAT_VERSION_2_2 = "2.2";
}
