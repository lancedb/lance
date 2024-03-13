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

import com.lancedb.lance.Dataset;
import com.lancedb.lance.Utils;
import java.io.IOException;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;

/** Fragment Arrow Reader. */
class FragmentArrowReader extends ArrowReader {

  private ArrowArrayStream stream;

  FragmentArrowReader(
      Dataset dataset, int fragmentId, Schema projection, long batchSize, BufferAllocator allocator)
      throws IOException {
    super(allocator);

    var arrowProjection = Utils.toFfi(projection, allocator);
    long handle = openStream(dataset, fragmentId, arrowProjection, batchSize);
    this.stream = ArrowArrayStream.wrap(handle);
  }

  @Override
  public boolean loadNextBatch() throws IOException {
    ArrowArray array = ArrowArray.allocateNew(super.allocator);
    stream.getNext(array);
    return true;
  }

  @Override
  public long bytesRead() {
    return 0;
  }

  @Override
  protected void closeReadSource() throws IOException {
    stream.release();
  }

  @Override
  protected Schema readSchema() throws IOException {
    var arrowSchema = ArrowSchema.allocateNew(allocator);
    stream.getSchema(arrowSchema);
    return Data.importSchema(allocator, arrowSchema, null);
  }

  /** Open Stream from Rust. */
  private native long openStream(
      Dataset dataset, int fragmentId, ArrowSchema projection, long batchSize) throws IOException;
}
