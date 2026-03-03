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
package org.lance.file;

/**
 * Controls how blob-encoded columns are returned when reading a Lance file.
 *
 * <p>Blob columns can be read in two modes:
 *
 * <ul>
 *   <li>{@link #CONTENT} — materializes the full binary content (default)
 *   <li>{@link #DESCRIPTOR} — returns a struct with {@code position} and {@code size} fields
 * </ul>
 */
public enum BlobReadMode {
  /** Return blob columns as materialized binary content (default). */
  CONTENT(0),
  /** Return blob columns as descriptors (struct with position and size). */
  DESCRIPTOR(1);

  private final int value;

  BlobReadMode(int value) {
    this.value = value;
  }

  public int getValue() {
    return value;
  }
}
