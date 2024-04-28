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
import java.util.Optional;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.dataset.scanner.ScanOptions;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;

/** Scanner over a Fragment. */
public class FragmentScanner extends DatasetScanner {
  private final int fragmentId;

  /** Create FragmentScanner. */
  public FragmentScanner(
      Dataset dataset, int fragmentId, ScanOptions options,
      Optional<String> filter, BufferAllocator allocator) {
    super(dataset, options, filter, allocator);
    Preconditions.checkArgument(!(options.getSubstraitFilter().isPresent() && filter.isPresent()), 
        "cannot set both substrait filter and string filter");
    this.fragmentId = fragmentId;
  }

  private static native void importFfiSchema(Dataset dataset, long arrowSchemaMemoryAddress,
      int fragmentId, Optional<String[]> columns);

  @Override
  public ArrowReader scanBatches() {
    try (ArrowArrayStream s = ArrowArrayStream.allocateNew(allocator)) {
      openStream(dataset, Optional.of(fragmentId), Utils.convert(options.getColumns()),
          options.getSubstraitFilter(), filter, options.getBatchSize(), s.memoryAddress());
      return Data.importArrayStream(allocator, s);
    } catch (IOException e) {
      // TODO: handle IO exception?
      throw new RuntimeException(e);
    }
  }

  /** Get the schema of the Scanner. */
  @Override
  public Schema schema() {
    try (ArrowSchema ffiArrowSchema = ArrowSchema.allocateNew(allocator)) {
      importFfiSchema(dataset, ffiArrowSchema.memoryAddress(), fragmentId, options.getColumns());
      return Data.importSchema(allocator, ffiArrowSchema, null);
    }
  }
}
