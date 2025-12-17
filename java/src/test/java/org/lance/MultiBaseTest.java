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
package org.lance;

import org.lance.fragment.DataFile;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class MultiBaseTest {
  private BufferAllocator allocator;
  @TempDir private Path tempDir;
  private String primary;
  private String base1;
  private String base2;

  @BeforeEach
  public void setup() throws Exception {
    allocator = new RootAllocator(Long.MAX_VALUE);
    Path primaryPath = tempDir.resolve("primary");
    Files.createDirectories(primaryPath);
    primary = primaryPath.toString();
    Path base1Path = tempDir.resolve("base1");
    Files.createDirectories(base1Path);
    base1 = base1Path.toString();
    Path base2Path = tempDir.resolve("base2");
    Files.createDirectories(base2Path);
    base2 = base2Path.toString();
  }

  @AfterEach
  public void teardown() throws Exception {
    if (allocator != null) {
      allocator.close();
    }
  }

  private ArrowStreamReader makeReader(int startId, int count) throws Exception {
    List<Field> fields =
        Arrays.asList(
            new Field("id", FieldType.notNullable(new ArrowType.Int(32, true)), null),
            new Field("value", FieldType.nullable(new ArrowType.Utf8()), null));

    Schema schema = new Schema(fields);

    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      IntVector idVec = (IntVector) root.getVector("id");
      idVec.allocateNew(count);
      VarCharVector valVec = (VarCharVector) root.getVector("value");
      valVec.allocateNew();
      for (int i = 0; i < count; i++) {
        int id = startId + i;
        idVec.setSafe(i, id);
        byte[] b = ("val_" + id).getBytes();
        valVec.setSafe(i, b, 0, b.length);
      }
      root.setRowCount(count);

      ByteArrayOutputStream out = new ByteArrayOutputStream();
      try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
        writer.start();
        writer.writeBatch();
        writer.end();
      }
      return new ArrowStreamReader(new ByteArrayInputStream(out.toByteArray()), allocator);
    }
  }

  @Test
  public void testCreateMode() throws Exception {
    ArrowStreamReader reader = makeReader(0, 500);
    List<BasePath> bases =
        Arrays.asList(
            new BasePath(0, Optional.of("base1"), base1, false),
            new BasePath(0, Optional.of("base2"), base2, false));

    Dataset ds =
        Dataset.write()
            .allocator(allocator)
            .reader(reader)
            .uri(primary)
            .mode(WriteParams.WriteMode.CREATE)
            .initialBases(bases)
            .targetBases(Arrays.asList("base2"))
            .maxRowsPerFile(100)
            .execute();

    assertNotNull(ds);
    assertEquals(primary, ds.uri());
    assertEquals(500, ds.countRows());
  }

  @Test
  public void testAppendMode() throws Exception {
    ArrowStreamReader initReader = makeReader(0, 300);
    List<BasePath> bases =
        Arrays.asList(
            new BasePath(0, Optional.of("base1"), base1, false),
            new BasePath(0, Optional.of("base2"), base2, false));

    Dataset base =
        Dataset.write()
            .allocator(allocator)
            .reader(initReader)
            .uri(primary)
            .mode(WriteParams.WriteMode.CREATE)
            .initialBases(bases)
            .targetBases(Arrays.asList("base1"))
            .maxRowsPerFile(100)
            .execute();

    ArrowStreamReader appendReader = makeReader(300, 100);
    Dataset appended =
        Dataset.write()
            .allocator(allocator)
            .reader(appendReader)
            .uri(base.uri())
            .mode(WriteParams.WriteMode.APPEND)
            .targetBases(Arrays.asList("base2"))
            .maxRowsPerFile(50)
            .execute();

    assertEquals(400, appended.countRows());
  }

  @Test
  public void testOverwriteInheritsBases() throws Exception {
    ArrowStreamReader initReader = makeReader(0, 200);
    List<BasePath> bases =
        Arrays.asList(
            new BasePath(0, Optional.of("base1"), base1, false),
            new BasePath(0, Optional.of("base2"), base2, false));

    Dataset.write()
        .allocator(allocator)
        .reader(initReader)
        .uri(primary)
        .mode(WriteParams.WriteMode.CREATE)
        .initialBases(bases)
        .targetBases(Arrays.asList("base1"))
        .maxRowsPerFile(100)
        .execute();

    ArrowStreamReader overwriteReader = makeReader(100, 150);
    Dataset updated =
        Dataset.write()
            .allocator(allocator)
            .reader(overwriteReader)
            .uri(primary)
            .mode(WriteParams.WriteMode.OVERWRITE)
            .targetBases(Arrays.asList("base2"))
            .maxRowsPerFile(75)
            .execute();

    assertEquals(150, updated.countRows());
  }

  @Test
  public void testTargetByPathUri() throws Exception {
    ArrowStreamReader reader = makeReader(0, 100);
    List<BasePath> bases =
        Arrays.asList(
            new BasePath(0, Optional.of("base1"), base1, true),
            new BasePath(0, Optional.of("base2"), base2, false));

    Dataset ds =
        Dataset.write()
            .allocator(allocator)
            .reader(reader)
            .uri(primary)
            .mode(WriteParams.WriteMode.CREATE)
            .initialBases(bases)
            .targetBases(Arrays.asList("base1"))
            .maxRowsPerFile(50)
            .execute();

    Set<Integer> baseIds =
        ds.getFragments().stream()
            .flatMap(f -> f.metadata().getFiles().stream().map(DataFile::getBaseId))
            .filter(Optional::isPresent)
            .map(Optional::get)
            .collect(Collectors.toSet());
    assertEquals(1, baseIds.size());

    ArrowStreamReader append = makeReader(100, 50);
    Dataset updated =
        Dataset.write()
            .allocator(allocator)
            .reader(append)
            .uri(ds.uri())
            .mode(WriteParams.WriteMode.APPEND)
            .targetBases(Arrays.asList(base2))
            .maxRowsPerFile(25)
            .execute();

    assertEquals(150, updated.countRows());
    baseIds =
        updated.getFragments().stream()
            .flatMap(f -> f.metadata().getFiles().stream().map(DataFile::getBaseId))
            .filter(Optional::isPresent)
            .map(Optional::get)
            .collect(Collectors.toSet());
    assertEquals(2, baseIds.size());
  }
}
