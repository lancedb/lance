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
package com.lancedb.lance.operation;

import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * Project to a new schema. This only changes the schema, not the data. This operation could be used
 * to: 1. add/remove columns. The data will be removed after compaction. 2. modify column positions
 */
public class Project implements Operation {

  private final Schema schema;
  private final BufferAllocator allocator;
  private ArrowSchema cSchema;

  private Project(Schema schema, BufferAllocator allocator) {
    this.schema = schema;
    this.allocator = allocator;
  }

  public Schema getSchema() {
    return schema;
  }

  @Override
  public String name() {
    return "Project";
  }

  @Override
  public void release() {
    if (cSchema != null) {
      cSchema.close();
    }
  }

  @Override
  public String toString() {
    return "Project{" + "schema=" + schema + '}';
  }

  public long exportSchema() {
    if (cSchema == null) {
      this.cSchema = ArrowSchema.allocateNew(allocator);
      Data.exportSchema(allocator, schema, null, cSchema);
    }
    return cSchema.memoryAddress();
  }

  // Builder class for Project
  public static class Builder {
    private Schema schema;
    private BufferAllocator allocator;

    public Builder() {}

    public Builder schema(Schema schema) {
      this.schema = schema;
      return this;
    }

    public Builder allocator(BufferAllocator allocator) {
      this.allocator = allocator;
      return this;
    }

    public Project build() {
      return new Project(schema, allocator);
    }
  }
}
