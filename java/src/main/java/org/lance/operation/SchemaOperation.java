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
package org.lance.operation;

import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Schema related base operation.
 *
 * <p>Each field will be assigned a field id when transaction commits, in the following order:
 *
 * <ol>
 *   <li>Parse from field metadata with key {@code lance:field_id}.
 *   <li>Otherwise, set field id from txn read version dataset's schema field (with the same name).
 *   <li>Otherwise, allocate based on the max field id of the dataset.
 * </ol>
 */
public abstract class SchemaOperation implements Operation {
  private final Schema schema;
  private final Map<Long, ArrowSchema> inFlightSchemas = new HashMap<>();

  protected SchemaOperation(Schema schema) {
    this.schema = schema;
  }

  public Schema schema() {
    return schema;
  }

  /**
   * Export the schema to rust jni.
   *
   * @param allocator the buffer allocator
   * @return the schema address
   */
  public synchronized long exportSchema(BufferAllocator allocator) {
    ArrowSchema cSchema = ArrowSchema.allocateNew(allocator);
    try {
      Data.exportSchema(allocator, schema, null, cSchema);
      long address = cSchema.memoryAddress();
      inFlightSchemas.put(address, cSchema);
      return address;
    } catch (RuntimeException | Error error) {
      cSchema.close();
      throw error;
    }
  }

  /** Confirm that JNI imported an exported schema and release its Java holder. */
  private synchronized void finishSchemaExport(long address) {
    ArrowSchema cSchema = inFlightSchemas.remove(address);
    if (cSchema == null) {
      throw new IllegalStateException("Unknown or completed ArrowSchema export: " + address);
    }
    cSchema.close();
  }

  public synchronized void release() {
    // In-flight holders are released only by finishSchemaExport after JNI imports them.
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    SchemaOperation that = (SchemaOperation) o;
    return Objects.equals(schema, that.schema);
  }
}
