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

package com.lancedb.lance.ipc;

import java.io.IOException;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.compression.CompressionCodec;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;

/// Fragment Arrow Reader
class FragmentArrowReader extends ArrowReader {
  private long nativeHandle;

  FragmentArrowReader(BufferAllocator allocator, CompressionCodec.Factory compressionFactory) {
    super(allocator, compressionFactory);
  }

  @Override
  public boolean loadNextBatch() throws IOException {
    return true;
  }

  @Override
  public long bytesRead() {
    return 0;
  }

  @Override
  protected void closeReadSource() throws IOException {}

  @Override
  protected Schema readSchema() throws IOException {
    return null;
  }
}
