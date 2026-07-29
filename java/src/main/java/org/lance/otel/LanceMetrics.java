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
package org.lance.otel;

import org.lance.JniLoader;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.common.AttributesBuilder;
import io.opentelemetry.api.metrics.Meter;
import io.opentelemetry.api.metrics.MeterProvider;
import io.opentelemetry.api.metrics.ObservableDoubleMeasurement;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Logger;

/** Bridges Lance's internal Rust metrics into OpenTelemetry observable instruments. */
public final class LanceMetrics {
  private static final Logger LOGGER = Logger.getLogger(LanceMetrics.class.getName());
  private static final String METER_NAME = "lance";
  private static final List<AutoCloseable> INSTRUMENTS = new ArrayList<>();
  private static boolean instrumented;

  private LanceMetrics() {}

  /**
   * Register Lance metrics on the global OpenTelemetry {@link MeterProvider}.
   *
   * @return true if the recorder is installed and instruments are registered, false if a different
   *     Rust metrics recorder is already installed in this process
   */
  public static synchronized boolean instrument() {
    return instrument(GlobalOpenTelemetry.get().getMeterProvider());
  }

  /**
   * Register Lance metrics as OpenTelemetry observable instruments.
   *
   * <p>This installs a process-global Rust metrics recorder. If a different Rust metrics recorder
   * is already installed, this returns false and does not create instruments. Repeated successful
   * calls are safe and return true without creating duplicate instruments.
   *
   * @param meterProvider the OpenTelemetry meter provider that owns the instruments
   * @return true if the recorder is installed and instruments are registered, false otherwise
   */
  public static synchronized boolean instrument(MeterProvider meterProvider) {
    Objects.requireNonNull(meterProvider, "meterProvider");
    JniLoader.ensureLoaded();

    if (!registerLanceMetricsRecorderNative()) {
      LOGGER.warning(
          "Could not install the Lance metrics recorder: another Rust metrics recorder is already"
              + " installed in this process. Lance metrics will not be exported via"
              + " OpenTelemetry.");
      return false;
    }

    if (instrumented) {
      return true;
    }

    Meter meter = meterProvider.meterBuilder(METER_NAME).build();
    for (MetricDescription desc : catalog()) {
      String unit = desc.getUnit() == null ? "" : desc.getUnit();
      String description = desc.getDescription();
      switch (desc.getKind()) {
        case "counter":
          INSTRUMENTS.add(
              meter
                  .counterBuilder(desc.getName())
                  .ofDoubles()
                  .setUnit(unit)
                  .setDescription(description)
                  .buildWithCallback(measurement -> recordScalar(desc.getName(), measurement)));
          break;
        case "gauge":
          INSTRUMENTS.add(
              meter
                  .gaugeBuilder(desc.getName())
                  .setUnit(unit)
                  .setDescription(description)
                  .buildWithCallback(measurement -> recordScalar(desc.getName(), measurement)));
          break;
        case "histogram":
          INSTRUMENTS.add(
              meter
                  .counterBuilder(desc.getName() + "_bucket")
                  .ofDoubles()
                  .setDescription(description + " (cumulative buckets)")
                  .buildWithCallback(measurement -> recordBuckets(desc.getName(), measurement)));
          INSTRUMENTS.add(
              meter
                  .counterBuilder(desc.getName() + "_count")
                  .ofDoubles()
                  .setDescription(description + " (count)")
                  .buildWithCallback(
                      measurement -> recordField(desc.getName(), "count", measurement)));
          INSTRUMENTS.add(
              meter
                  .counterBuilder(desc.getName() + "_sum")
                  .ofDoubles()
                  .setUnit(unit)
                  .setDescription(description + " (sum)")
                  .buildWithCallback(
                      measurement -> recordField(desc.getName(), "sum", measurement)));
          break;
        default:
          throw new IllegalStateException("Unknown Lance metric kind: " + desc.getKind());
      }
    }

    instrumented = true;
    return true;
  }

  /** Return the catalog of Lance metrics described by the native recorder. */
  public static List<MetricDescription> catalog() {
    JniLoader.ensureLoaded();
    return Collections.unmodifiableList(lanceMetricsCatalogNative());
  }

  /** Return a point-in-time snapshot of all recorded Lance metric series. */
  public static List<MetricPoint> snapshot() {
    JniLoader.ensureLoaded();
    return Collections.unmodifiableList(snapshotLanceMetricsNative());
  }

  private static void recordScalar(String metricName, ObservableDoubleMeasurement measurement) {
    for (MetricPoint point : snapshot()) {
      if (metricName.equals(point.getName()) && point.getValue() != null) {
        measurement.record(point.getValue(), attributes(point.getAttributes()));
      }
    }
  }

  private static void recordBuckets(String metricName, ObservableDoubleMeasurement measurement) {
    for (MetricPoint point : snapshot()) {
      if (!metricName.equals(point.getName()) || point.getBuckets() == null) {
        continue;
      }
      for (MetricBucket bucket : point.getBuckets()) {
        AttributesBuilder attributes = Attributes.builder();
        for (Map.Entry<String, String> entry : point.getAttributes().entrySet()) {
          attributes.put(entry.getKey(), entry.getValue());
        }
        attributes.put("le", bucket.getLe());
        measurement.record(bucket.getCumulativeCount(), attributes.build());
      }
    }
  }

  private static void recordField(
      String metricName, String fieldName, ObservableDoubleMeasurement measurement) {
    for (MetricPoint point : snapshot()) {
      if (!metricName.equals(point.getName())) {
        continue;
      }
      Double value = fieldValue(point, fieldName);
      if (value != null) {
        measurement.record(value, attributes(point.getAttributes()));
      }
    }
  }

  private static Double fieldValue(MetricPoint point, String fieldName) {
    switch (fieldName) {
      case "count":
        return point.getCount() == null ? null : point.getCount().doubleValue();
      case "sum":
        return point.getSum();
      default:
        throw new IllegalArgumentException("Unknown Lance metric field: " + fieldName);
    }
  }

  private static Attributes attributes(Map<String, String> values) {
    AttributesBuilder builder = Attributes.builder();
    for (Map.Entry<String, String> entry : values.entrySet()) {
      builder.put(entry.getKey(), entry.getValue());
    }
    return builder.build();
  }

  private static native boolean registerLanceMetricsRecorderNative();

  private static native List<MetricDescription> lanceMetricsCatalogNative();

  private static native List<MetricPoint> snapshotLanceMetricsNative();
}
