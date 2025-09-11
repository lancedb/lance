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

import com.lancedb.lance.index.DistanceType;
import com.lancedb.lance.index.IndexParams;
import com.lancedb.lance.index.IndexType;
import com.lancedb.lance.index.vector.VectorIndexParams;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.complex.FixedSizeListVector;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.util.Text;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

public class TestVectorDataset implements AutoCloseable {
  public static final String vectorColumnName = "vec";
  public static final String indexName = "idx";
  private final Path datasetPath;
  private Schema schema;
  private BufferAllocator allocator;

  public TestVectorDataset(Path datasetPath) {
    this.datasetPath = datasetPath;
  }

  public Dataset create() throws IOException {
    this.allocator = new RootAllocator();
    this.schema = createSchema();
    return createDataset();
  }

  private Schema createSchema() {
    Map<String, String> metadata = new HashMap<>();
    metadata.put("dataset", "vector");

    List<Field> fields =
        Arrays.asList(
            new Field("i", FieldType.nullable(new ArrowType.Int(32, true)), null),
            new Field("s", FieldType.nullable(new ArrowType.Utf8()), null),
            new Field(
                vectorColumnName,
                FieldType.nullable(new ArrowType.FixedSizeList(32)),
                Collections.singletonList(
                    new Field(
                        "item",
                        FieldType.nullable(
                            new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)),
                        null))));

    return new Schema(fields, metadata);
  }

  private Dataset createDataset() throws IOException {
    WriteParams writeParams =
        new WriteParams.Builder().withMaxRowsPerGroup(10).withMaxRowsPerFile(200).build();

    Dataset.create(allocator, datasetPath.toString(), schema, writeParams).close();

    List<FragmentMetadata> fragments = new ArrayList<>();
    for (int batchIndex = 0; batchIndex < 5; batchIndex++) {
      fragments.add(createFragment(batchIndex));
    }

    FragmentOperation.Append appendOp = new FragmentOperation.Append(fragments);
    return Dataset.commit(allocator, datasetPath.toString(), appendOp, Optional.of(1L));
  }

  private FragmentMetadata createFragment(int batchIndex) throws IOException {
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      IntVector iVector = (IntVector) root.getVector("i");
      VarCharVector sVector = (VarCharVector) root.getVector("s");
      FixedSizeListVector vecVector = (FixedSizeListVector) root.getVector(vectorColumnName);
      Float4Vector vecItemsVector = (Float4Vector) vecVector.getDataVector();

      for (int i = 0; i < 80; i++) {
        int value = batchIndex * 80 + i;
        iVector.setSafe(i, value);
        sVector.setSafe(i, new Text("s-" + value));

        for (int j = 0; j < 32; j++) {
          vecItemsVector.setSafe(i * 32 + j, (float) (i * 32 + j));
        }
      }
      root.setRowCount(80);

      WriteParams fragmentWriteParams = new WriteParams.Builder().build();
      return Fragment.create(datasetPath.toString(), allocator, root, fragmentWriteParams).get(0);
    }
  }

  public Dataset appendNewData() throws IOException {
    FragmentMetadata fragmentMetadata;
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      IntVector iVector = (IntVector) root.getVector("i");
      VarCharVector sVector = (VarCharVector) root.getVector("s");
      FixedSizeListVector vecVector = (FixedSizeListVector) root.getVector(vectorColumnName);
      Float4Vector vecItemsVector = (Float4Vector) vecVector.getDataVector();

      for (int i = 0; i < 10; i++) {
        int value = 400 + i;
        iVector.setSafe(i, value);
        sVector.setSafe(i, new Text("s-" + value));

        for (int j = 0; j < 32; j++) {
          vecItemsVector.setSafe(i * 32 + j, (float) i);
        }
      }
      root.setRowCount(10);

      WriteParams writeParams = new WriteParams.Builder().build();
      fragmentMetadata =
          Fragment.create(datasetPath.toString(), allocator, root, writeParams).get(0);
    }
    FragmentOperation.Append appendOp =
        new FragmentOperation.Append(Collections.singletonList(fragmentMetadata));
    return Dataset.commit(allocator, datasetPath.toString(), appendOp, Optional.of(2L));
  }

  public void createIndex(Dataset dataset) {
    IndexParams params =
        new IndexParams.Builder()
            .setVectorIndexParams(VectorIndexParams.ivfPq(2, 8, 2, DistanceType.L2, 2))
            .build();
    dataset.createIndex(
        Arrays.asList(vectorColumnName), IndexType.VECTOR, Optional.of(indexName), params, true);
  }

  @Override
  public void close() {
    if (allocator != null) {
      allocator.close();
    }
  }
}
