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

import com.lancedb.lance.index.IndexParams;
import com.lancedb.lance.index.IndexType;
import com.lancedb.lance.index.scalar.ScalarIndexParams;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.Optional;

public class ScalarIndexTest {

  @TempDir Path tempDir;

  @Test
  public void testCreateBTreeIndex() throws Exception {
    String datasetPath = tempDir.resolve("btree_test").toString();
    Schema schema =
        new Schema(
            Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("name", new ArrowType.Utf8())),
            null);

    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset dataset =
          Dataset.create(allocator, datasetPath, schema, new WriteParams.Builder().build())) {

        // Create BTree scalar index parameters
        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{\"zone_size\": 2048}");

        IndexParams indexParams =
            new IndexParams.Builder().setScalarIndexParams(scalarParams).build();

        // Create BTree index on 'id' column
        dataset.createIndex(
            Collections.singletonList("id"),
            IndexType.BTREE,
            Optional.of("btree_id_index"),
            indexParams,
            false);

        // Verify index was created
        assert dataset.listIndexes().contains("btree_id_index");
      }
    }
  }

  @Test
  public void testCreateZonemapIndex() throws Exception {
    String datasetPath = tempDir.resolve("zonemap_test").toString();
    Schema schema =
        new Schema(
            Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("value", new ArrowType.Utf8())),
            null);

    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset dataset =
          Dataset.create(allocator, datasetPath, schema, new WriteParams.Builder().build())) {

        // Create Zonemap scalar index parameters (no additional params needed)
        ScalarIndexParams scalarParams = ScalarIndexParams.create("zonemap");

        IndexParams indexParams =
            new IndexParams.Builder().setScalarIndexParams(scalarParams).build();

        // Create Zonemap index on 'value' column
        dataset.createIndex(
            Collections.singletonList("value"),
            IndexType.ZONEMAP,
            Optional.of("zonemap_value_index"),
            indexParams,
            false);

        // Verify index was created
        assert dataset.listIndexes().contains("zonemap_value_index");
      }
    }
  }
}
