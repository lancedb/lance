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

import com.lancedb.lance.test.JniTestHelper;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

/**
 * Benchmark tests for basic JNI operations. Tests the performance of data type parsing and method
 * call overhead.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = JniBenchmarkConfig.DEFAULT_WARMUP_ITERATIONS)
@Measurement(iterations = JniBenchmarkConfig.DEFAULT_MEASUREMENT_ITERATIONS)
@Fork(JniBenchmarkConfig.DEFAULT_FORKS)
public class JniBasicOperationsBenchmark {

  private JniTestDataGenerator dataGenerator;

  @Param({"100", "1000", "10000"})
  private int dataSize;

  private List<Integer> integerList;
  private List<Long> longList;
  private Optional<List<Integer>> optionalIntegerList;

  @Setup(Level.Trial)
  public void setup() {
    dataGenerator = new JniTestDataGenerator(42L);
    integerList = dataGenerator.generateIntegerList(dataSize);
    longList = dataGenerator.generateLongList(dataSize);
    optionalIntegerList = Optional.of(integerList);
  }

  @Benchmark
  public void benchmarkParseInts(Blackhole bh) {
    JniTestHelper.parseInts(integerList);
    bh.consume(integerList);
  }

  @Benchmark
  public void benchmarkParseLongs(Blackhole bh) {
    JniTestHelper.parseLongs(longList);
    bh.consume(longList);
  }

  @Benchmark
  public void benchmarkParseIntsOpt(Blackhole bh) {
    JniTestHelper.parseIntsOpt(optionalIntegerList);
    bh.consume(optionalIntegerList);
  }

  @Benchmark
  public void benchmarkParseQuery(Blackhole bh) {
    Optional<com.lancedb.lance.ipc.Query> queryOpt =
        Optional.of(dataGenerator.generateQuery(128, 10));
    JniTestHelper.parseQuery(queryOpt);
    bh.consume(queryOpt);
  }

  @Benchmark
  public void benchmarkParseIndexParams(Blackhole bh) {
    com.lancedb.lance.index.IndexParams indexParams = dataGenerator.generateIndexParams();
    JniTestHelper.parseIndexParams(indexParams);
    bh.consume(indexParams);
  }
}
