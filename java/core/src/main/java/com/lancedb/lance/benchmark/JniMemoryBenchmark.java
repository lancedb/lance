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
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Benchmark tests for JNI memory management operations.
 * Tests memory allocation, deallocation, and garbage collection impact.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = JniBenchmarkConfig.DEFAULT_WARMUP_ITERATIONS)
@Measurement(iterations = JniBenchmarkConfig.DEFAULT_MEASUREMENT_ITERATIONS)
@Fork(JniBenchmarkConfig.DEFAULT_FORKS)
public class JniMemoryBenchmark {

    private BufferAllocator allocator;
    private Path tempDir;
    private Schema schema;
    private WriteParams writeParams;
    private JniPerformanceMetrics metrics;

    @Param({"10", "100", "1000"})
    private int objectCount;

    @Setup(Level.Trial)
    public void setup() throws IOException {
        allocator = new RootAllocator();
        tempDir = Files.createTempDirectory("lance-memory-benchmark");

        Field field = new Field("id", FieldType.nullable(new ArrowType.Int(32, true)), null);
        schema = new Schema(Arrays.asList(field));
        writeParams = new WriteParams.Builder().build();
        metrics = new JniPerformanceMetrics();
    }

    @TearDown(Level.Trial)
    public void tearDown() throws IOException {
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
    public void benchmarkDatasetAllocation(Blackhole bh) throws IOException {
        metrics.startMeasurement();

        List<Dataset> datasets = new ArrayList<>();
        try {
            for (int i = 0; i < objectCount; i++) {
                String datasetPath = tempDir.resolve("dataset_alloc_" + i).toString();
                Dataset ds = Dataset.create(allocator, datasetPath, schema, writeParams);
                datasets.add(ds);
            }

            bh.consume(datasets);
        } finally {
            // Clean up datasets
            for (Dataset ds : datasets) {
                try {
                    ds.close();
                } catch (Exception e) {
                    // Ignore cleanup errors
                }
            }
        }

        JniPerformanceMetrics.MetricsResult result = metrics.endMeasurement();
        bh.consume(result);
    }

    @Benchmark
    public void benchmarkAllocatorAllocation(Blackhole bh) {
        metrics.startMeasurement();

        List<BufferAllocator> allocators = new ArrayList<>();
        try {
            for (int i = 0; i < objectCount; i++) {
                BufferAllocator childAllocator = allocator.newChildAllocator(
                        "child-" + i, 0, Long.MAX_VALUE);
                allocators.add(childAllocator);
            }

            bh.consume(allocators);
        } finally {
            // Clean up allocators
            for (BufferAllocator alloc : allocators) {
                try {
                    alloc.close();
                } catch (Exception e) {
                    // Ignore cleanup errors
                }
            }
        }

        JniPerformanceMetrics.MetricsResult result = metrics.endMeasurement();
        bh.consume(result);
    }

    @Benchmark
    public void benchmarkGarbageCollectionImpact(Blackhole bh) throws IOException {
        metrics.startMeasurement();

        // Create and immediately discard objects to trigger GC
        for (int i = 0; i < objectCount; i++) {
            String datasetPath = tempDir.resolve("dataset_gc_" + i).toString();
            try (Dataset ds = Dataset.create(allocator, datasetPath, schema, writeParams)) {
                bh.consume(ds);
                // Force some garbage collection by creating temporary objects
                byte[] garbage = new byte[1024 * 1024]; // 1MB
                bh.consume(garbage);
            }
        }

        // Force garbage collection
        System.gc();

        JniPerformanceMetrics.MetricsResult result = metrics.endMeasurement();
        bh.consume(result);
    }
}