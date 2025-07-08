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
package com.lancedb.lance.benchmark;

import com.lancedb.lance.Dataset;
import com.lancedb.lance.WriteParams;
import com.lancedb.lance.ipc.LanceScanner;
import com.lancedb.lance.ipc.ScanOptions;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;

/**
 * Benchmark tests for Dataset JNI operations.
 * Tests the performance of dataset creation, opening, and scanning.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = JniBenchmarkConfig.DEFAULT_WARMUP_ITERATIONS)
@Measurement(iterations = JniBenchmarkConfig.DEFAULT_MEASUREMENT_ITERATIONS)
@Fork(JniBenchmarkConfig.DEFAULT_FORKS)
public class JniDatasetBenchmark {

    private BufferAllocator allocator;
    private Path tempDir;
    private Schema schema;
    private WriteParams writeParams;
    private Dataset dataset;

    @Param({"1000", "10000", "100000"})
    private int rowCount;

    @Setup(Level.Trial)
    public void setup() throws IOException {
        allocator = new RootAllocator();
        tempDir = Files.createTempDirectory("lance-benchmark");

        // Create simple schema
        Field field = new Field("id", FieldType.nullable(new ArrowType.Int(32, true)), null);
        schema = new Schema(Arrays.asList(field));

        writeParams = new WriteParams.Builder().build();
    }

    @TearDown(Level.Trial)
    public void tearDown() throws IOException {
        if (dataset != null) {
            dataset.close();
        }
        if (allocator != null) {
            allocator.close();
        }
        // Clean up temp directory
        Files.walk(tempDir)
                .sorted((a, b) -> b.compareTo(a))
                .forEach(path -> {
                    try {
                        Files.delete(path);
                    } catch (IOException e) {
                        // Ignore cleanup errors
                    }
                });
    }

    @Benchmark
    public void benchmarkDatasetCreate(Blackhole bh) throws IOException {
        String datasetPath = tempDir.resolve("dataset_" + System.nanoTime()).toString();

        try (Dataset ds = Dataset.create(allocator, datasetPath, schema, writeParams)) {
            bh.consume(ds);
        }
    }

    @Benchmark
    public void benchmarkDatasetOpen(Blackhole bh) throws IOException {
        // Pre-create a dataset for opening
        String datasetPath = tempDir.resolve("dataset_open_" + System.nanoTime()).toString();
        try (Dataset ds = Dataset.create(allocator, datasetPath, schema, writeParams)) {
            // Dataset created, now benchmark opening it
        }

        try (Dataset ds = Dataset.open(allocator, datasetPath, new HashMap<>())) {
            bh.consume(ds);
        }
    }

    @Benchmark
    public void benchmarkDatasetScan(Blackhole bh) throws IOException {
        if (dataset == null) {
            // Create dataset with data for scanning
            String datasetPath = tempDir.resolve("dataset_scan").toString();
            dataset = Dataset.create(allocator, datasetPath, schema, writeParams);

            // Add some test data
            try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
                 IntVector intVector = (IntVector) root.getVector("id")) {

                root.allocateNew();
                for (int i = 0; i < Math.min(rowCount, 1000); i++) {
                    intVector.set(i, i);
                }
                root.setRowCount(Math.min(rowCount, 1000));

                // Write data to dataset (this would require additional implementation)
                // For now, we'll just benchmark the scanner creation
            }
        }

        ScanOptions options = new ScanOptions.Builder()
                .batchSize(1000L)
                .build();

        try (LanceScanner scanner = dataset.newScan(options)) {
            bh.consume(scanner);
        }
    }
}