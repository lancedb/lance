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

import com.lancedb.lance.index.IndexType;
import com.lancedb.lance.index.scalar.ScalarIndexParams;
import com.lancedb.lance.index.scalar.ScalarIndexType;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

public class CreateEmptyIndexTest {
    private static BufferAllocator allocator;

    @BeforeAll
    static void setup() {
        allocator = new RootAllocator();
    }

    @AfterAll
    static void tearDown() {
        allocator.close();
    }

    @Test
    void testCreateEmptyBtreeIndex(@TempDir Path tempPath) throws Exception {
        String datasetPath = tempPath.resolve("btree_dataset").toString();

        // Create dataset with numeric column
        List<Field> fields = Arrays.asList(
            Field.nullable("id", new FieldType(true, org.apache.arrow.vector.types.pojo.ArrowType.Int.createInt(32, true), null))
        );
        Schema schema = new Schema(fields);

        try (IntVector idVector = new IntVector("id", allocator)) {
            idVector.allocateNew(100);
            for (int i = 0; i < 100; i++) {
                idVector.set(i, i);
            }
            idVector.setValueCount(100);

            List<org.apache.arrow.vector.FieldVector> vectors = Arrays.asList(idVector);
            try (Dataset dataset = Dataset.create(allocator, datasetPath, schema, vectors, Optional.empty())) {
                // Create empty BTREE index with train=false
                ScalarIndexParams params = new ScalarIndexParams();
                dataset.createIndex(
                    Arrays.asList("id"),
                    IndexType.Scalar,
                    Optional.of("id_index"),
                    params,
                    false,
                    false  // train=false
                );

                // TODO: Verify index exists and has correct stats
                // This would require implementing list_indices and index_statistics methods
            }
        }
    }

    @Test
    void testCreateEmptyBitmapIndex(@TempDir Path tempPath) throws Exception {
        String datasetPath = tempPath.resolve("bitmap_dataset").toString();

        // Create dataset with low cardinality string column
        List<Field> fields = Arrays.asList(
            Field.nullable("category", new FieldType(true, org.apache.arrow.vector.types.pojo.ArrowType.Utf8.INSTANCE, null))
        );
        Schema schema = new Schema(fields);

        try (VarCharVector categoryVector = new VarCharVector("category", allocator)) {
            categoryVector.allocateNew(300);
            String[] categories = {"A", "B", "C"};
            for (int i = 0; i < 100; i++) {
                categoryVector.set(i, categories[i % 3].getBytes());
            }
            categoryVector.setValueCount(100);

            List<org.apache.arrow.vector.FieldVector> vectors = Arrays.asList(categoryVector);
            try (Dataset dataset = Dataset.create(allocator, datasetPath, schema, vectors, Optional.empty())) {
                // Create empty BITMAP index with train=false
                ScalarIndexParams params = new ScalarIndexParams(ScalarIndexType.BITMAP);
                dataset.createIndex(
                    Arrays.asList("category"),
                    IndexType.BITMAP,
                    Optional.of("category_bitmap"),
                    params,
                    false,
                    false  // train=false
                );

                // TODO: Verify index exists
            }
        }
    }

    @Test
    void testCreateEmptyInvertedIndex(@TempDir Path tempPath) throws Exception {
        String datasetPath = tempPath.resolve("inverted_dataset").toString();

        // Create dataset with text column
        List<Field> fields = Arrays.asList(
            Field.nullable("text", new FieldType(true, org.apache.arrow.vector.types.pojo.ArrowType.Utf8.INSTANCE, null))
        );
        Schema schema = new Schema(fields);

        try (VarCharVector textVector = new VarCharVector("text", allocator)) {
            textVector.allocateNew(300);
            String[] texts = {"hello world", "foo bar", "hello foo"};
            for (int i = 0; i < 30; i++) {
                textVector.set(i, texts[i % 3].getBytes());
            }
            textVector.setValueCount(30);

            List<org.apache.arrow.vector.FieldVector> vectors = Arrays.asList(textVector);
            try (Dataset dataset = Dataset.create(allocator, datasetPath, schema, vectors, Optional.empty())) {
                // Create empty INVERTED index with train=false
                dataset.createIndex(
                    Arrays.asList("text"),
                    IndexType.INVERTED,
                    Optional.of("text_inverted"),
                    new ScalarIndexParams(),
                    false,
                    false  // train=false
                );

                // TODO: Verify index exists
            }
        }
    }

    @Test
    void testAppendAfterEmptyIndex(@TempDir Path tempPath) throws Exception {
        String datasetPath = tempPath.resolve("append_dataset").toString();

        // Create initial dataset
        List<Field> fields = Arrays.asList(
            Field.nullable("id", new FieldType(true, org.apache.arrow.vector.types.pojo.ArrowType.Int.createInt(32, true), null))
        );
        Schema schema = new Schema(fields);

        try (IntVector idVector = new IntVector("id", allocator)) {
            idVector.allocateNew(50);
            for (int i = 0; i < 50; i++) {
                idVector.set(i, i);
            }
            idVector.setValueCount(50);

            List<org.apache.arrow.vector.FieldVector> vectors = Arrays.asList(idVector);
            try (Dataset dataset = Dataset.create(allocator, datasetPath, schema, vectors, Optional.empty())) {
                // Create empty index
                dataset.createIndex(
                    Arrays.asList("id"),
                    IndexType.Scalar,
                    Optional.of("id_index"),
                    new ScalarIndexParams(),
                    false,
                    false  // train=false
                );
            }

            // Append more data
            idVector.clear();
            idVector.allocateNew(50);
            for (int i = 0; i < 50; i++) {
                idVector.set(i, i + 50);
            }
            idVector.setValueCount(50);

            try (Dataset dataset = Dataset.open(allocator, datasetPath, Optional.empty())) {
                dataset.append(vectors);
                
                // TODO: Verify index still exists after append
                // TODO: Verify index statistics show it's still empty
            }
        }
    }
}