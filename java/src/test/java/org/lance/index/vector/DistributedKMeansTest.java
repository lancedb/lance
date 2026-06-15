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
import org.apache.arrow.vector.Float4Vector;
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
import static org.junit.jupiter.api.Assertions.assertTrue;

class DistributedKMeansTest {

  private static final int DIM = 8;
  private static final int N = 1_000;

  /** Build a small in-memory Lance dataset of FixedSizeList&lt;Float32, DIM&gt; vectors. */
  private Dataset writeVectorDataset(String uri, BufferAllocator allocator) throws Exception {
    Field child =
        new Field(
            "item",
            new FieldType(true, new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE), null),
            Collections.emptyList());
    Field vec =
        new Field(
            "vec",
            new FieldType(true, new ArrowType.FixedSizeList(DIM), null),
            Collections.singletonList(child));
    Schema schema = new Schema(Collections.singletonList(vec));

    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      FixedSizeListVector list = (FixedSizeListVector) root.getVector("vec");
      list.allocateNew();
      Float4Vector inner = (Float4Vector) list.getDataVector();
      inner.allocateNew(N * DIM);

      Random rng = new Random(42);
      for (int i = 0; i < N; i++) {
        list.setNotNull(i);
        for (int d = 0; d < DIM; d++) {
          inner.set(i * DIM + d, (float) rng.nextGaussian());
        }
      }
      inner.setValueCount(N * DIM);
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

        float[] bootstrap =
            DistributedKMeans.bootstrapCentroids(
                Collections.singletonList(samples), 16, DistanceType.L2, 7L);
        assertEquals(16 * DIM, bootstrap.length);
        for (float v : bootstrap) {
          assertTrue(Float.isFinite(v), "non-finite centroid value");
        }

        // Wrap the flat float[] back into a VectorSchemaRoot for the next round.
        VectorSchemaRoot centroidsVsr = floatsToVsr(allocator, bootstrap, DIM);
        try {
          VectorSchemaRoot partial =
              DistributedKMeans.computePartialStats(
                  dataset, "vec", centroidsVsr, DistanceType.L2, null, allocator);
          try {
            assertEquals(16, partial.getRowCount());

            VectorSchemaRoot merged =
                DistributedKMeans.reducePartialStats(Collections.singletonList(partial), allocator);
            try {
              assertEquals(16, merged.getRowCount());
              float[] next = DistributedKMeans.finalizeCentroids(merged, centroidsVsr);
              assertEquals(bootstrap.length, next.length);
              for (float v : next) {
                assertTrue(Float.isFinite(v), "non-finite centroid value");
              }
            } finally {
              merged.close();
            }
          } finally {
            partial.close();
          }
        } finally {
          centroidsVsr.close();
        }
      } finally {
        samples.close();
      }
    }
  }

  /** Wrap a flat float array into a single VSR holding a FixedSizeList&lt;Float32, dim&gt;. */
  private VectorSchemaRoot floatsToVsr(BufferAllocator allocator, float[] values, int dim) {
    int k = values.length / dim;
    Field child =
        new Field(
            "item",
            new FieldType(true, new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE), null),
            Collections.emptyList());
    Field vec =
        new Field(
            "vec",
            new FieldType(true, new ArrowType.FixedSizeList(dim), null),
            Collections.singletonList(child));
    Schema schema = new Schema(Collections.singletonList(vec));
    VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
    FixedSizeListVector list = (FixedSizeListVector) root.getVector("vec");
    list.allocateNew();
    Float4Vector inner = (Float4Vector) list.getDataVector();
    inner.allocateNew(values.length);
    for (int i = 0; i < k; i++) {
      list.setNotNull(i);
      for (int d = 0; d < dim; d++) {
        inner.set(i * dim + d, values[i * dim + d]);
      }
    }
    inner.setValueCount(values.length);
    list.setValueCount(k);
    root.setRowCount(k);
    return root;
  }
}
