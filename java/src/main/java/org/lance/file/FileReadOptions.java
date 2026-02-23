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
 * Options for reading a Lance file.
 *
 * <p>Use {@link #builder()} to create an instance. New options can be added here in the future
 * without breaking existing callers.
 */
public class FileReadOptions {
  private final BlobReadMode blobReadMode;

  private FileReadOptions(Builder builder) {
    this.blobReadMode = builder.blobReadMode;
  }

  /** Returns the blob read mode. Defaults to {@link BlobReadMode#CONTENT}. */
  public BlobReadMode getBlobReadMode() {
    return blobReadMode;
  }

  /** Creates a new builder with default options. */
  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private BlobReadMode blobReadMode = BlobReadMode.CONTENT;

    private Builder() {}

    /**
     * Sets how blob-encoded columns are returned.
     *
     * @param blobReadMode {@link BlobReadMode#CONTENT} to materialize binary content, or {@link
     *     BlobReadMode#DESCRIPTOR} to return position/size descriptors
     */
    public Builder blobReadMode(BlobReadMode blobReadMode) {
      this.blobReadMode = blobReadMode;
      return this;
    }

    public FileReadOptions build() {
      return new FileReadOptions(this);
    }
  }
}
