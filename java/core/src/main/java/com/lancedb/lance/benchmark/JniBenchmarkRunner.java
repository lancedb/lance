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

import org.openjdk.jmh.results.format.ResultFormatType;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Main runner for JNI benchmark tests.
 * Provides convenient methods to run all or specific benchmark suites.
 */
public class JniBenchmarkRunner {

    private static final String RESULTS_DIR = "benchmark-results";
    private static final DateTimeFormatter TIMESTAMP_FORMAT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss");

    public static void main(String[] args) throws RunnerException, IOException {
        if (args.length > 0) {
            switch (args[0].toLowerCase()) {
                case "basic":
                    runBasicOperationsBenchmarks();
                    break;
                case "dataset":
                    runDatasetBenchmarks();
                    break;
                case "memory":
                    runMemoryBenchmarks();
                    break;
                case "all":
                default:
                    runAllBenchmarks();
                    break;
            }
        } else {
            runAllBenchmarks();
        }
    }

    public static void runAllBenchmarks() throws RunnerException, IOException {
        System.out.println("Running all JNI benchmarks...");

        String timestamp = LocalDateTime.now().format(TIMESTAMP_FORMAT);
        String resultFile = String.format("%s/jni-benchmark-all-%s.json", RESULTS_DIR, timestamp);

        Options opt = new OptionsBuilder()
                .include(".*JniBasicOperationsBenchmark.*")
                .include(".*JniDatasetBenchmark.*")
                .include(".*JniMemoryBenchmark.*")
                .warmupIterations(JniBenchmarkConfig.DEFAULT_WARMUP_ITERATIONS)
                .measurementIterations(JniBenchmarkConfig.DEFAULT_MEASUREMENT_ITERATIONS)
                .forks(JniBenchmarkConfig.DEFAULT_FORKS)
                .resultFormat(ResultFormatType.JSON)
                .result(resultFile)
                .build();

        new Runner(opt).run();
        System.out.println("Results saved to: " + resultFile);
    }

    public static void runBasicOperationsBenchmarks() throws RunnerException, IOException {
        System.out.println("Running basic JNI operations benchmarks...");

        String timestamp = LocalDateTime.now().format(TIMESTAMP_FORMAT);
        String resultFile = String.format("%s/jni-benchmark-basic-%s.json", RESULTS_DIR, timestamp);

        Options opt = new OptionsBuilder()
                .include(".*JniBasicOperationsBenchmark.*")
                .warmupIterations(JniBenchmarkConfig.DEFAULT_WARMUP_ITERATIONS)
                .measurementIterations(JniBenchmarkConfig.DEFAULT_MEASUREMENT_ITERATIONS)
                .forks(JniBenchmarkConfig.DEFAULT_FORKS)
                .resultFormat(ResultFormatType.JSON)
                .result(resultFile)
                .build();

        new Runner(opt).run();
        System.out.println("Results saved to: " + resultFile);
    }

    public static void runDatasetBenchmarks() throws RunnerException, IOException {
        System.out.println("Running dataset JNI benchmarks...");

        String timestamp = LocalDateTime.now().format(TIMESTAMP_FORMAT);
        String resultFile = String.format("%s/jni-benchmark-dataset-%s.json", RESULTS_DIR, timestamp);

        Options opt = new OptionsBuilder()
                .include(".*JniDatasetBenchmark.*")
                .warmupIterations(JniBenchmarkConfig.DEFAULT_WARMUP_ITERATIONS)
                .measurementIterations(JniBenchmarkConfig.DEFAULT_MEASUREMENT_ITERATIONS)
                .forks(JniBenchmarkConfig.DEFAULT_FORKS)
                .resultFormat(ResultFormatType.JSON)
                .result(resultFile)
                .build();

        new Runner(opt).run();
        System.out.println("Results saved to: " + resultFile);
    }

    public static void runMemoryBenchmarks() throws RunnerException, IOException {
        System.out.println("Running memory management JNI benchmarks...");

        String timestamp = LocalDateTime.now().format(TIMESTAMP_FORMAT);
        String resultFile = String.format("%s/jni-benchmark-memory-%s.json", RESULTS_DIR, timestamp);

        Options opt = new OptionsBuilder()
                .include(".*JniMemoryBenchmark.*")
                .warmupIterations(JniBenchmarkConfig.DEFAULT_WARMUP_ITERATIONS)
                .measurementIterations(JniBenchmarkConfig.DEFAULT_MEASUREMENT_ITERATIONS)
                .forks(JniBenchmarkConfig.DEFAULT_FORKS)
                .resultFormat(ResultFormatType.JSON)
                .result(resultFile)
                .build();

        new Runner(opt).run();
        System.out.println("Results saved to: " + resultFile);
    }

    private static void ensureResultsDirectory() throws IOException {
        Files.createDirectories(Paths.get(RESULTS_DIR));
    }
}