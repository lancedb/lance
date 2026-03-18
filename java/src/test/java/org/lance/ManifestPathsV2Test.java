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

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ManifestPathsV2Test {
  private static final Pattern V2_MANIFEST_PATTERN = Pattern.compile("\\d{20}\\.manifest");

  @Test
  void testMigrateManifestPathsFromV1ToV2(@TempDir Path tempDir) throws IOException {
    String datasetPath = tempDir.resolve("testMigrateManifestPathsFromV1ToV2").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      // Create v1 test.
      try (Dataset dataset = testDataset.createEmptyDataset(false)) {
        Path versionsDir = Paths.get(datasetPath).resolve("_versions");
        assertTrue(Files.isDirectory(versionsDir), "_versions directory should exist");
        List<Path> manifestsBefore;
        try (Stream<Path> stream = Files.list(versionsDir)) {
          manifestsBefore =
              stream
                  .filter(
                      p ->
                          Files.isRegularFile(p)
                              && p.getFileName().toString().endsWith(".manifest"))
                  .collect(Collectors.toList());
        }
        assertEquals(1, manifestsBefore.size(), "Expected single manifest before migration");
        assertEquals("1.manifest", manifestsBefore.get(0).getFileName().toString());

        // Migrate to v2.
        dataset.migrateManifestPathsV2();

        List<Path> manifestsAfter;
        try (Stream<Path> stream = Files.list(versionsDir)) {
          manifestsAfter =
              stream
                  .filter(
                      p ->
                          Files.isRegularFile(p)
                              && p.getFileName().toString().endsWith(".manifest"))
                  .collect(Collectors.toList());
        }
        assertEquals(1, manifestsAfter.size(), "Expected single manifest after migration");
        String fileName = manifestsAfter.get(0).getFileName().toString();
        assertTrue(
            V2_MANIFEST_PATTERN.matcher(fileName).matches(),
            "Manifest should use V2 naming scheme");
      }
    }
  }

  @Test
  void testCreateDatasetUsesV2ManifestByDefault(@TempDir Path tempDir) throws IOException {
    String datasetPath = tempDir.resolve("testCreateDatasetUsesV2ManifestByDefault").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      Schema schema =
          new Schema(
              Arrays.asList(
                  Field.nullable("id", new ArrowType.Int(32, true)),
                  Field.nullable("name", new ArrowType.Utf8())));
      WriteParams params = new WriteParams.Builder().withMode(WriteParams.WriteMode.CREATE).build();
      try (Dataset dataset = Dataset.create(allocator, datasetPath, schema, params)) {
        Path versionsDir = Paths.get(datasetPath).resolve("_versions");
        assertTrue(Files.isDirectory(versionsDir), "_versions directory should exist");
        List<Path> manifests;
        try (Stream<Path> stream = Files.list(versionsDir)) {
          manifests =
              stream
                  .filter(
                      p ->
                          Files.isRegularFile(p)
                              && p.getFileName().toString().endsWith(".manifest"))
                  .collect(Collectors.toList());
        }
        assertEquals(1, manifests.size(), "Expected single manifest file");
        String fileName = manifests.get(0).getFileName().toString();
        assertTrue(
            V2_MANIFEST_PATTERN.matcher(fileName).matches(),
            "Manifest should use V2 naming scheme");
      }
    }
  }
}
