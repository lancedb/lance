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
package org.lance.index.vector;

import org.lance.Dataset;
import org.lance.WriteParams;
import org.lance.index.DistanceType;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float2Vector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.FixedSizeListVector;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.util.ByteArrayReadableSeekableByteChannel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DistributedKMeansTest {

  private static final int DIM = 8;
  private static final int N = 1_000;

  private enum Precision {
    HALF,
    SINGLE,
    DOUBLE
  }

  /** Build a small in-memory Lance dataset of FixedSizeList&lt;Float?, DIM&gt; vectors. */
  private Dataset writeVectorDataset(String uri, BufferAllocator allocator, Precision precision)
      throws Exception {
    ArrowType.FloatingPoint floatType;
    switch (precision) {
      case HALF:
        floatType = new ArrowType.FloatingPoint(FloatingPointPrecision.HALF);
        break;
      case SINGLE:
        floatType = new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE);
        break;
      case DOUBLE:
        floatType = new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);
        break;
      default:
        throw new IllegalStateException("unreachable");
    }
    Field child = new Field("item", new FieldType(true, floatType, null), Collections.emptyList());
    Field vec =
        new Field(
            "vec",
            new FieldType(true, new ArrowType.FixedSizeList(DIM), null),
            Collections.singletonList(child));
    Schema schema = new Schema(Collections.singletonList(vec));

    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      FixedSizeListVector list = (FixedSizeListVector) root.getVector("vec");
      list.allocateNew();

      Random rng = new Random(42);
      switch (precision) {
        case HALF:
          {
            Float2Vector inner = (Float2Vector) list.getDataVector();
            inner.allocateNew(N * DIM);
            for (int i = 0; i < N; i++) {
              list.setNotNull(i);
              for (int d = 0; d < DIM; d++) {
                inner.setWithPossibleTruncate(i * DIM + d, (float) rng.nextGaussian());
              }
            }
            inner.setValueCount(N * DIM);
            break;
          }
        case SINGLE:
          {
            Float4Vector inner = (Float4Vector) list.getDataVector();
            inner.allocateNew(N * DIM);
            for (int i = 0; i < N; i++) {
              list.setNotNull(i);
              for (int d = 0; d < DIM; d++) {
                inner.set(i * DIM + d, (float) rng.nextGaussian());
              }
            }
            inner.setValueCount(N * DIM);
            break;
          }
        case DOUBLE:
          {
            Float8Vector inner = (Float8Vector) list.getDataVector();
            inner.allocateNew(N * DIM);
            for (int i = 0; i < N; i++) {
              list.setNotNull(i);
              for (int d = 0; d < DIM; d++) {
                inner.set(i * DIM + d, rng.nextGaussian());
              }
            }
            inner.setValueCount(N * DIM);
            break;
          }
      }
      list.setValueCount(N);
      root.setRowCount(N);

      ByteArrayOutputStream out = new ByteArrayOutputStream();
      try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
        writer.start();
        writer.writeBatch();
        writer.end();
      }
      try (ArrowStreamReader reader =
          new ArrowStreamReader(
              new ByteArrayReadableSeekableByteChannel(out.toByteArray()), allocator)) {
        return Dataset.write()
            .allocator(allocator)
            .reader(reader)
            .uri(uri)
            .mode(WriteParams.WriteMode.OVERWRITE)
            .execute();
      }
    }
  }

  private Dataset writeVectorDataset(String uri, BufferAllocator allocator) throws Exception {
    return writeVectorDataset(uri, allocator, Precision.SINGLE);
  }

  private static FloatingPointPrecision innerPrecision(VectorSchemaRoot root) {
    FixedSizeListVector list = (FixedSizeListVector) root.getVector("vec");
    ArrowType inner = list.getDataVector().getField().getType();
    return ((ArrowType.FloatingPoint) inner).getPrecision();
  }

  @Test
  void roundTripFourPrimitives(@TempDir Path tmp) throws Exception {
    String datasetUri = tmp.resolve("vec.lance").toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeVectorDataset(datasetUri, allocator)) {

      VectorSchemaRoot samples =
          DistributedKMeans.sampleRound0(
              dataset, "vec", 256, DistanceType.L2, 42L, null, allocator);
      try {
        assertEquals(256, samples.getRowCount());

        VectorSchemaRoot bootstrap =
            DistributedKMeans.bootstrapCentroids(
                Collections.singletonList(samples), 16, DistanceType.L2, 7L, allocator);
        try {
          assertEquals(16, bootstrap.getRowCount());
          assertEquals(FloatingPointPrecision.SINGLE, innerPrecision(bootstrap));
          FixedSizeListVector bootstrapList = (FixedSizeListVector) bootstrap.getVector("vec");
          Float4Vector bootstrapInner = (Float4Vector) bootstrapList.getDataVector();
          assertEquals(16 * DIM, bootstrapInner.getValueCount());
          for (int i = 0; i < bootstrapInner.getValueCount(); i++) {
            assertTrue(Float.isFinite(bootstrapInner.get(i)), "non-finite centroid value");
          }

          VectorSchemaRoot partial =
              DistributedKMeans.computePartialStats(
                  dataset, "vec", bootstrap, DistanceType.L2, null, allocator);
          try {
            assertEquals(16, partial.getRowCount());

            VectorSchemaRoot merged =
                DistributedKMeans.reducePartialStats(Collections.singletonList(partial), allocator);
            try {
              assertEquals(16, merged.getRowCount());
              VectorSchemaRoot next =
                  DistributedKMeans.finalizeCentroids(merged, bootstrap, allocator);
              try {
                assertEquals(16, next.getRowCount());
                assertEquals(FloatingPointPrecision.SINGLE, innerPrecision(next));
                Float4Vector nextInner =
                    (Float4Vector) ((FixedSizeListVector) next.getVector("vec")).getDataVector();
                assertEquals(16 * DIM, nextInner.getValueCount());
                for (int i = 0; i < nextInner.getValueCount(); i++) {
                  assertTrue(Float.isFinite(nextInner.get(i)), "non-finite centroid value");
                }
              } finally {
                next.close();
              }
            } finally {
              merged.close();
            }
          } finally {
            partial.close();
          }
        } finally {
          bootstrap.close();
        }
      } finally {
        samples.close();
      }
    }
  }

  @Test
  void roundTripFloat16(@TempDir Path tmp) throws Exception {
    String datasetUri = tmp.resolve("vec16.lance").toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeVectorDataset(datasetUri, allocator, Precision.HALF)) {
      runEndToEnd(dataset, allocator, FloatingPointPrecision.HALF);
    }
  }

  @Test
  void roundTripFloat64(@TempDir Path tmp) throws Exception {
    String datasetUri = tmp.resolve("vec64.lance").toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeVectorDataset(datasetUri, allocator, Precision.DOUBLE)) {
      runEndToEnd(dataset, allocator, FloatingPointPrecision.DOUBLE);
    }
  }

  private void runEndToEnd(
      Dataset dataset, BufferAllocator allocator, FloatingPointPrecision expected)
      throws Exception {
    VectorSchemaRoot samples =
        DistributedKMeans.sampleRound0(dataset, "vec", 256, DistanceType.L2, 42L, null, allocator);
    try {
      assertEquals(expected, innerPrecision(samples));
      VectorSchemaRoot bootstrap =
          DistributedKMeans.bootstrapCentroids(
              Collections.singletonList(samples), 16, DistanceType.L2, 7L, allocator);
      try {
        assertEquals(16, bootstrap.getRowCount());
        assertEquals(expected, innerPrecision(bootstrap));

        VectorSchemaRoot partial =
            DistributedKMeans.computePartialStats(
                dataset, "vec", bootstrap, DistanceType.L2, null, allocator);
        try {
          assertEquals(16, partial.getRowCount());

          VectorSchemaRoot merged =
              DistributedKMeans.reducePartialStats(Collections.singletonList(partial), allocator);
          try {
            assertEquals(16, merged.getRowCount());
            VectorSchemaRoot next =
                DistributedKMeans.finalizeCentroids(merged, bootstrap, allocator);
            try {
              assertEquals(16, next.getRowCount());
              assertEquals(expected, innerPrecision(next));
            } finally {
              next.close();
            }
          } finally {
            merged.close();
          }
        } finally {
          partial.close();
        }
      } finally {
        bootstrap.close();
      }
    } finally {
      samples.close();
    }
  }

  @Test
  void sampleRound0NegativeTargetThrows(@TempDir Path tmp) throws Exception {
    String datasetUri = tmp.resolve("vec.lance").toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeVectorDataset(datasetUri, allocator)) {
      assertThrows(
          IllegalArgumentException.class,
          () ->
              DistributedKMeans.sampleRound0(
                  dataset, "vec", -1L, DistanceType.L2, 42L, null, allocator));
    }
  }

  @Test
  void selectInitialCentroidsNegativeKThrows(@TempDir Path tmp) throws Exception {
    String datasetUri = tmp.resolve("vec.lance").toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeVectorDataset(datasetUri, allocator)) {
      VectorSchemaRoot samples =
          DistributedKMeans.sampleRound0(dataset, "vec", 64, DistanceType.L2, 1L, null, allocator);
      try {
        assertThrows(
            IllegalArgumentException.class,
            () ->
                DistributedKMeans.selectInitialCentroids(
                    Collections.singletonList(samples), -1, 1L, allocator));
      } finally {
        samples.close();
      }
    }
  }

  @Test
  void bootstrapCentroidsNegativeKThrows(@TempDir Path tmp) throws Exception {
    String datasetUri = tmp.resolve("vec.lance").toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeVectorDataset(datasetUri, allocator)) {
      VectorSchemaRoot samples =
          DistributedKMeans.sampleRound0(dataset, "vec", 64, DistanceType.L2, 1L, null, allocator);
      try {
        assertThrows(
            IllegalArgumentException.class,
            () ->
                DistributedKMeans.bootstrapCentroids(
                    Collections.singletonList(samples), -1, DistanceType.L2, 1L, allocator));
      } finally {
        samples.close();
      }
    }
  }
}
