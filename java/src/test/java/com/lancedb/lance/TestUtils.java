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

import com.lancedb.lance.fragment.FragmentMergeResult;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.dataset.scanner.Scanner;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowFileReader;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.ipc.SeekableReadChannel;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.util.ByteArrayReadableSeekableByteChannel;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestUtils {
  private abstract static class TestDataset {
    protected final BufferAllocator allocator;
    private final String datasetPath;

    public TestDataset(BufferAllocator allocator, String datasetPath) {
      this.allocator = allocator;
      this.datasetPath = datasetPath;
    }

    public abstract Schema getSchema();

    public Dataset createEmptyDataset() {
      Dataset dataset =
          Dataset.create(allocator, datasetPath, getSchema(), new WriteParams.Builder().build());
      assertEquals(0, dataset.countRows());
      assertEquals(getSchema(), dataset.getSchema());
      List<Fragment> fragments = dataset.getFragments();
      assertEquals(0, fragments.size());
      assertEquals(1, dataset.version());
      assertEquals(1, dataset.latestVersion());
      return dataset;
    }

    public Dataset createDatasetWithWriteParams(WriteParams writeParams) {
      return Dataset.create(allocator, datasetPath, getSchema(), writeParams);
    }

    public FragmentMetadata createNewFragment(int rowCount) {
      List<FragmentMetadata> fragmentMetas = createNewFragment(rowCount, Integer.MAX_VALUE);
      assertEquals(1, fragmentMetas.size());
      FragmentMetadata fragmentMeta = fragmentMetas.get(0);
      assertEquals(rowCount, fragmentMeta.getPhysicalRows());
      return fragmentMeta;
    }

    public List<FragmentMetadata> createNewFragment(int rowCount, int maxRowsPerFile) {
      List<FragmentMetadata> fragmentMetas;
      try (VectorSchemaRoot root = VectorSchemaRoot.create(getSchema(), allocator)) {
        root.allocateNew();
        IntVector idVector = (IntVector) root.getVector("id");
        VarCharVector nameVector = (VarCharVector) root.getVector("name");
        for (int i = 0; i < rowCount; i++) {
          int id = i;
          idVector.setSafe(i, id);
          String name = "Person " + i;
          nameVector.setSafe(i, name.getBytes(StandardCharsets.UTF_8));
        }
        root.setRowCount(rowCount);

        fragmentMetas =
            Fragment.create(
                datasetPath,
                allocator,
                root,
                new WriteParams.Builder().withMaxRowsPerFile(maxRowsPerFile).build());
      }
      return fragmentMetas;
    }

    public Dataset write(long version, int rowCount) {
      FragmentMetadata metadata = createNewFragment(rowCount);
      FragmentOperation.Append appendOp = new FragmentOperation.Append(Arrays.asList(metadata));
      return Dataset.commit(allocator, datasetPath, appendOp, Optional.of(version));
    }

    public Dataset writeSortByDataset(long version) {
      List<FragmentMetadata> fragmentMetas;
      try (VectorSchemaRoot root = VectorSchemaRoot.create(getSchema(), allocator)) {
        root.allocateNew();
        IntVector idVector = (IntVector) root.getVector("id");
        VarCharVector nameVector = (VarCharVector) root.getVector("name");
        /* dataset context
         * i: |  id   | name |
         * 0: |  0    | null |
         * 1: |  1    |  P0  |
         * 2: | null  |  P1  |
         * 3: |  2    |  P2  |
         * 4: |  2    |  P3  |
         * 5: | null  |  P3  |
         * 6: |  3    | null |
         * 7: |  4    |  P4  |
         * 8: |  4    |  P5  |
         * 9: |  5    |  P5  |
         */
        idVector.set(0, 0);
        idVector.set(1, 1);
        idVector.setNull(2);
        idVector.set(3, 2);
        idVector.set(4, 2);
        idVector.setNull(5);
        idVector.set(6, 3);
        idVector.set(7, 4);
        idVector.set(8, 4);
        idVector.set(9, 5);

        nameVector.setNull(0);
        nameVector.set(1, "P0".getBytes());
        nameVector.set(2, "P1".getBytes());
        nameVector.set(3, "P2".getBytes());
        nameVector.set(4, "P3".getBytes());
        nameVector.set(5, "P3".getBytes());
        nameVector.setNull(6);
        nameVector.set(7, "P4".getBytes());
        nameVector.set(8, "P5".getBytes());
        nameVector.set(9, "P5".getBytes());

        root.setRowCount(10);

        fragmentMetas =
            Fragment.create(
                datasetPath,
                allocator,
                root,
                new WriteParams.Builder().withMaxRowsPerFile(Integer.MAX_VALUE).build());
      }
      FragmentOperation.Append appendOp = new FragmentOperation.Append(fragmentMetas);
      return Dataset.commit(allocator, datasetPath, appendOp, Optional.of(version));
    }

    public void validateScanResults(Dataset dataset, Scanner scanner, int totalRows, int batchRows)
        throws IOException {
      try (ArrowReader reader = scanner.scanBatches()) {
        assertEquals(
            dataset.getSchema().getFields(), reader.getVectorSchemaRoot().getSchema().getFields());
        int rowcount = 0;
        while (reader.loadNextBatch()) {
          int currentRowCount = reader.getVectorSchemaRoot().getRowCount();
          assertEquals(batchRows, currentRowCount);
          rowcount += currentRowCount;
        }
        assertEquals(totalRows, rowcount);
      }
    }

    public void validateScanResults(
        Dataset dataset, Scanner scanner, int expectedRows, int batchRows, int offset)
        throws IOException {
      try (ArrowReader reader = scanner.scanBatches()) {
        assertEquals(
            dataset.getSchema().getFields(), reader.getVectorSchemaRoot().getSchema().getFields());
        int rowcount = 0;
        while (reader.loadNextBatch()) {
          VectorSchemaRoot root = reader.getVectorSchemaRoot();
          int currentRowCount = root.getRowCount();
          assertTrue(currentRowCount <= batchRows);
          rowcount += currentRowCount;

          IntVector idVector = (IntVector) root.getVector("id");
          for (int i = 0; i < currentRowCount; i++) {
            int expectedId = offset + rowcount - currentRowCount + i;
            assertEquals(
                expectedId, idVector.get(i), "Mismatch at row " + (rowcount - currentRowCount + i));
          }
        }
        assertEquals(expectedRows, rowcount);
      }
    }
  }

  public static class RandomAccessDataset {
    private static final String DATA_FILE = "/random_access.arrow";
    private static final int ROW_COUNT = 9;
    private final BufferAllocator allocator;
    private final String datasetPath;
    private Schema schema;

    public RandomAccessDataset(BufferAllocator allocator, String datasetPath) {
      this.allocator = allocator;
      this.datasetPath = datasetPath;
    }

    public void createDatasetAndValidate() throws IOException, URISyntaxException {
      Path path = Paths.get(DatasetTest.class.getResource(DATA_FILE).toURI());
      try (BufferAllocator allocator = new RootAllocator();
          ArrowFileReader reader =
              new ArrowFileReader(
                  new SeekableReadChannel(
                      new ByteArrayReadableSeekableByteChannel(Files.readAllBytes(path))),
                  allocator);
          ArrowArrayStream arrowStream = ArrowArrayStream.allocateNew(allocator)) {
        Data.exportArrayStream(allocator, reader, arrowStream);
        try (Dataset dataset =
            Dataset.create(
                allocator,
                arrowStream,
                datasetPath,
                new WriteParams.Builder()
                    .withMaxRowsPerFile(10)
                    .withMaxRowsPerGroup(20)
                    .withMode(WriteParams.WriteMode.CREATE)
                    .withStorageOptions(new HashMap<>())
                    .build())) {
          assertEquals(ROW_COUNT, dataset.countRows());
          schema = reader.getVectorSchemaRoot().getSchema();
          validateFragments(dataset);
          assertEquals(1, dataset.version());
          assertEquals(1, dataset.latestVersion());
        }
      }
    }

    public void openDatasetAndValidate() throws IOException {
      try (Dataset dataset = Dataset.open(datasetPath, allocator)) {
        assertEquals(1, dataset.version());
        assertEquals(1, dataset.latestVersion());
        assertEquals(ROW_COUNT, dataset.countRows());
        validateFragments(dataset);
      }
    }

    public Schema getSchema() {
      assertNotNull(schema);
      return schema;
    }

    private void validateFragments(Dataset dataset) {
      assertNotNull(schema);
      assertNotNull(dataset);
      List<Fragment> fragments = dataset.getFragments();
      assertEquals(1, fragments.size());
      assertEquals(0, fragments.get(0).getId());
      assertEquals(9, fragments.get(0).countRows());
      assertEquals(schema, dataset.getSchema());
    }
  }

  public static ByteBuffer getSubstraitByteBuffer(String substrait) {
    byte[] decodedSubstrait = substrait.getBytes();
    ByteBuffer substraitExpression = ByteBuffer.allocateDirect(decodedSubstrait.length);
    substraitExpression.put(decodedSubstrait);
    return substraitExpression;
  }

  public static class SimpleTestDataset extends TestDataset {
    private static final Schema schema =
        new Schema(
            Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("name", new ArrowType.Utf8())),
            null);

    public SimpleTestDataset(BufferAllocator allocator, String datasetPath) {
      super(allocator, datasetPath);
    }

    @Override
    public Schema getSchema() {
      return schema;
    }
  }

  public static class ComplexTestDataset extends TestDataset {
    public static final Schema COMPLETE_SCHEMA =
        new Schema(
            Arrays.asList(
                // basic scalar types
                Field.nullable("null_col", ArrowType.Null.INSTANCE),
                Field.nullable("bool_col", ArrowType.Bool.INSTANCE),
                Field.nullable("int8_col", new ArrowType.Int(8, true)),
                Field.nullable("uint32_col", new ArrowType.Int(32, false)),
                Field.nullable(
                    "float64_col", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)),

                // strings and binary types
                Field.nullable("utf8_col", ArrowType.Utf8.INSTANCE),
                Field.nullable("large_utf8_col", ArrowType.LargeUtf8.INSTANCE),
                Field.nullable("binary_col", ArrowType.Binary.INSTANCE),
                Field.nullable("fixed_binary_col", new ArrowType.FixedSizeBinary(16)),

                // time and date types
                Field.notNullable("date32_col", new ArrowType.Date(DateUnit.DAY)),
                Field.nullable(
                    "timestamp_col", new ArrowType.Timestamp(TimeUnit.MICROSECOND, "UTC")),
                Field.nullable("time64_nano_col", new ArrowType.Time(TimeUnit.NANOSECOND, 64)),

                // decimals
                Field.notNullable("decimal128_col", new ArrowType.Decimal(38, 10, 128)),
                Field.nullable("decimal256_col", new ArrowType.Decimal(76, 20, 256)),

                // nested types
                new Field(
                    "list_col",
                    FieldType.nullable(new ArrowType.List()),
                    Collections.singletonList(Field.nullable("item", new ArrowType.Int(32, true)))),

                // struct and union types
                new Field(
                    "struct_col",
                    FieldType.nullable(new ArrowType.Struct()),
                    Arrays.asList(
                        Field.nullable("field1", ArrowType.Utf8.INSTANCE),
                        Field.nullable("field2", new ArrowType.Int(16, true))))));

    public ComplexTestDataset(BufferAllocator allocator, String datasetPath) {
      super(allocator, datasetPath);
    }

    @Override
    public Schema getSchema() {
      return COMPLETE_SCHEMA;
    }
  }

  public static class MergeColumnTestDataset extends TestDataset {
    private static final Schema schema =
        new Schema(
            Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("name", new ArrowType.Utf8())),
            null);

    private static final Schema mergeColumnSchema =
        new Schema(
            Arrays.asList(
                Field.nullable("_rowid", new ArrowType.Int(64, false)),
                Field.nullable("new_col1", new ArrowType.Utf8()),
                Field.nullable("new_col2", new ArrowType.Utf8())),
            null);

    public MergeColumnTestDataset(BufferAllocator allocator, String datasetPath) {
      super(allocator, datasetPath);
    }

    @Override
    public Schema getSchema() {
      return schema;
    }

    public Schema getMergeColumnSchema() {
      return mergeColumnSchema;
    }

    /**
     * Test method to merge columns. Note that for simplicity, the merged column rowid is fixed with
     * [0, mergeNum). Please only use this method to test the first fragment.
     *
     * @param fragment fragment to merge.
     * @param mergeNum number of new rows.
     * @return merge result
     */
    public FragmentMergeResult mergeColumn(Fragment fragment, int mergeNum) {
      try (VectorSchemaRoot root = VectorSchemaRoot.create(getMergeColumnSchema(), allocator)) {
        root.allocateNew();
        UInt8Vector rowidVec = (UInt8Vector) root.getVector("_rowid");
        VarCharVector newCol1Vec = (VarCharVector) root.getVector("new_col1");
        VarCharVector newCol2Vec = (VarCharVector) root.getVector("new_col2");
        for (int i = 0; i < mergeNum; i++) {
          rowidVec.setSafe(i, i);
          newCol1Vec.setSafe(i, String.format("new_col1_%s", i).getBytes(StandardCharsets.UTF_8));
          newCol2Vec.setSafe(i, String.format("new_col2_%s", i).getBytes(StandardCharsets.UTF_8));
        }
        root.setRowCount(mergeNum);

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
          writer.start();
          writer.writeBatch();
          writer.end();
        } catch (IOException e) {
          throw new RuntimeException("Cannot write schema root", e);
        }

        byte[] arrowData = out.toByteArray();
        ByteArrayInputStream in = new ByteArrayInputStream(arrowData);

        try (ArrowStreamReader reader = new ArrowStreamReader(in, allocator);
            ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
          Data.exportArrayStream(allocator, reader, stream);
          return fragment.mergeColumns(stream, "_rowid", "_rowid");
        } catch (Exception e) {
          throw new RuntimeException("Cannot read arrow stream.", e);
        }
      }
    }
  }
}
