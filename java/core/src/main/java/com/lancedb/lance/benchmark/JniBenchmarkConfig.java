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

import java.util.concurrent.TimeUnit;

/**
 * Configuration class for JNI benchmark tests. Provides standardized settings for performance
 * testing.
 */
public class JniBenchmarkConfig {
  // Benchmark execution parameters
  public static final int DEFAULT_WARMUP_ITERATIONS = 5;
  public static final int DEFAULT_MEASUREMENT_ITERATIONS = 10;
  public static final int DEFAULT_FORKS = 1;
  public static final TimeUnit DEFAULT_TIME_UNIT = TimeUnit.MICROSECONDS;

  // Data size configurations for testing
  public static final int[] DATA_SIZES = {100, 1000, 10000, 100000};
  public static final int[] VECTOR_DIMENSIONS = {128, 256, 512, 1024};
  public static final int[] BATCH_SIZES = {1, 10, 100, 1000};

  // Memory and performance thresholds
  public static final long MAX_MEMORY_USAGE_MB = 1024;
  public static final double MAX_GC_OVERHEAD_PERCENT = 5.0;

  private JniBenchmarkConfig() {
    // Utility class
  }
}
