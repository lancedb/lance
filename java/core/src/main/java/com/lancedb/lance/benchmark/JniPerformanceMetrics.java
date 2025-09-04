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

import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Collects and analyzes performance metrics for JNI operations. */
public class JniPerformanceMetrics {
  private final MemoryMXBean memoryBean;
  private final List<GarbageCollectorMXBean> gcBeans;
  private long startTime;
  private long endTime;
  private MemoryUsage startMemory;
  private MemoryUsage endMemory;
  private Map<String, Long> startGcCounts;
  private Map<String, Long> startGcTimes;

  public JniPerformanceMetrics() {
    this.memoryBean = ManagementFactory.getMemoryMXBean();
    this.gcBeans = ManagementFactory.getGarbageCollectorMXBeans();
    this.startGcCounts = new HashMap<String, Long>();
    this.startGcTimes = new HashMap<String, Long>();
  }

  public void startMeasurement() {
    this.startTime = System.nanoTime();
    this.startMemory = memoryBean.getHeapMemoryUsage();

    for (GarbageCollectorMXBean gcBean : gcBeans) {
      startGcCounts.put(gcBean.getName(), gcBean.getCollectionCount());
      startGcTimes.put(gcBean.getName(), gcBean.getCollectionTime());
    }
  }

  public MetricsResult endMeasurement() {
    this.endTime = System.nanoTime();
    this.endMemory = memoryBean.getHeapMemoryUsage();

    Map<String, Long> endGcCounts = new HashMap<String, Long>();
    Map<String, Long> endGcTimes = new HashMap<String, Long>();

    for (GarbageCollectorMXBean gcBean : gcBeans) {
      endGcCounts.put(gcBean.getName(), gcBean.getCollectionCount());
      endGcTimes.put(gcBean.getName(), gcBean.getCollectionTime());
    }

    return new MetricsResult(
        endTime - startTime,
        endMemory.getUsed() - startMemory.getUsed(),
        calculateGcOverhead(startGcCounts, endGcCounts, startGcTimes, endGcTimes));
  }

  private double calculateGcOverhead(
      Map<String, Long> startCounts,
      Map<String, Long> endCounts,
      Map<String, Long> startTimes,
      Map<String, Long> endTimes) {
    long totalGcTime = 0;
    for (String gcName : startTimes.keySet()) {
      totalGcTime += endTimes.get(gcName) - startTimes.get(gcName);
    }

    long totalTime = endTime - startTime;
    return totalTime > 0 ? (double) totalGcTime * 1000000 / totalTime * 100 : 0.0;
  }

  public static class MetricsResult {
    public final long executionTimeNanos;
    public final long memoryDeltaBytes;
    public final double gcOverheadPercent;

    public MetricsResult(long executionTimeNanos, long memoryDeltaBytes, double gcOverheadPercent) {
      this.executionTimeNanos = executionTimeNanos;
      this.memoryDeltaBytes = memoryDeltaBytes;
      this.gcOverheadPercent = gcOverheadPercent;
    }

    @Override
    public String toString() {
      return String.format(
          "Execution: %.2f μs, Memory: %d bytes, GC Overhead: %.2f%%",
          executionTimeNanos / 1000.0, memoryDeltaBytes, gcOverheadPercent);
    }
  }
}
