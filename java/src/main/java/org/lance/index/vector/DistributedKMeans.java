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
import org.lance.JniLoader;
import org.lance.index.DistanceType;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorLoader;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.VectorUnloader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;
import org.apache.arrow.vector.types.pojo.Schema;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.channels.Channels;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Distributed IVF centroid-training primitives.
 *
 * <p>Mirrors {@code lance::index::vector::ivf::distributed}. Callers (Spark, custom RPC) own
 * broadcast, tree-reduce, and convergence; this class exposes only the math.
 *
 * <p>Every payload that crosses the JNI boundary — sampled rows, partial stats, and centroid arrays
 * — moves as Arrow IPC byte arrays. The centroid-returning helpers ({@link #finalizeCentroids},
 * {@link #selectInitialCentroids}, {@link #bootstrapCentroids}) return a {@link VectorSchemaRoot}
 * whose child vector preserves the original Float16/Float32/Float64 element dtype.
 */
public final class DistributedKMeans {

  static {
    JniLoader.ensureLoaded();
  }

  private DistributedKMeans() {}

  /** Round-0 reservoir-sample on the worker's fragment slice. */
  public static VectorSchemaRoot sampleRound0(
      Dataset dataset,
      String column,
      long target,
      DistanceType distanceType,
      long rngSeed,
      int[] fragmentIds,
      BufferAllocator allocator) {
    Objects.requireNonNull(dataset, "dataset");
    Objects.requireNonNull(column, "column");
    Objects.requireNonNull(distanceType, "distanceType");
    Objects.requireNonNull(allocator, "allocator");
    byte[] ipc =
        nativeSampleRound0(dataset, column, target, distanceType.toString(), rngSeed, fragmentIds);
    return readIpc(ipc, allocator);
  }

  /** Round-r E-step on the worker's fragment slice. */
  public static VectorSchemaRoot computePartialStats(
      Dataset dataset,
      String column,
      VectorSchemaRoot centroids,
      DistanceType distanceType,
      int[] fragmentIds,
      BufferAllocator allocator) {
    Objects.requireNonNull(dataset, "dataset");
    Objects.requireNonNull(column, "column");
    Objects.requireNonNull(centroids, "centroids");
    Objects.requireNonNull(distanceType, "distanceType");
    Objects.requireNonNull(allocator, "allocator");
    byte[] centroidsIpc = writeIpc(centroids);
    byte[] statsIpc =
        nativeComputePartialStats(
            dataset, column, centroidsIpc, distanceType.toString(), fragmentIds);
    return readIpc(statsIpc, allocator);
  }

  /** Combine two partial stats. */
  public static VectorSchemaRoot mergePartialStats(
      VectorSchemaRoot a, VectorSchemaRoot b, BufferAllocator allocator) {
    Objects.requireNonNull(a, "a");
    Objects.requireNonNull(b, "b");
    return readIpc(nativeMergePartialStats(writeIpc(a), writeIpc(b)), allocator);
  }

  /** Fold a list of partial stats. */
  public static VectorSchemaRoot reducePartialStats(
      List<VectorSchemaRoot> stats, BufferAllocator allocator) {
    Objects.requireNonNull(stats, "stats");
    List<byte[]> serialized = new ArrayList<>(stats.size());
    for (VectorSchemaRoot s : stats) {
      serialized.add(writeIpc(s));
    }
    return readIpc(nativeReducePartialStats(serialized.toArray(new byte[0][])), allocator);
  }

  /**
   * Compute new centroids; the returned VectorSchemaRoot has a single FixedSizeList column whose
   * inner dtype matches {@code prev} (Float16/Float32/Float64).
   */
  public static VectorSchemaRoot finalizeCentroids(
      VectorSchemaRoot stats, VectorSchemaRoot prev, BufferAllocator allocator) {
    Objects.requireNonNull(stats, "stats");
    Objects.requireNonNull(prev, "prev");
    Objects.requireNonNull(allocator, "allocator");
    return readIpc(nativeFinalizeCentroids(writeIpc(stats), writeIpc(prev)), allocator);
  }

  /**
   * Driver-side: pick {@code k} rows uniformly at random from worker samples. The returned
   * VectorSchemaRoot has a single FixedSizeList column whose inner dtype matches the samples.
   */
  public static VectorSchemaRoot selectInitialCentroids(
      List<VectorSchemaRoot> samples, int k, long rngSeed, BufferAllocator allocator) {
    Objects.requireNonNull(samples, "samples");
    Objects.requireNonNull(allocator, "allocator");
    List<byte[]> serialized = new ArrayList<>(samples.size());
    for (VectorSchemaRoot s : samples) {
      serialized.add(writeIpc(s));
    }
    return readIpc(
        nativeSelectInitialCentroids(serialized.toArray(new byte[0][]), k, rngSeed), allocator);
  }

  /**
   * Driver-side: bootstrap centroids by running single-machine kmeans on worker samples. The
   * returned VectorSchemaRoot has a single FixedSizeList column whose inner dtype matches the
   * samples.
   */
  public static VectorSchemaRoot bootstrapCentroids(
      List<VectorSchemaRoot> samples,
      int k,
      DistanceType distanceType,
      long rngSeed,
      BufferAllocator allocator) {
    Objects.requireNonNull(samples, "samples");
    Objects.requireNonNull(distanceType, "distanceType");
    Objects.requireNonNull(allocator, "allocator");
    List<byte[]> serialized = new ArrayList<>(samples.size());
    for (VectorSchemaRoot s : samples) {
      serialized.add(writeIpc(s));
    }
    return readIpc(
        nativeBootstrapCentroids(
            serialized.toArray(new byte[0][]), k, distanceType.toString(), rngSeed),
        allocator);
  }

  // -- helpers -------------------------------------------------------------

  private static byte[] writeIpc(VectorSchemaRoot root) {
    try (ByteArrayOutputStream out = new ByteArrayOutputStream();
        ArrowStreamWriter writer = new ArrowStreamWriter(root, null, Channels.newChannel(out))) {
      writer.start();
      writer.writeBatch();
      writer.end();
      return out.toByteArray();
    } catch (Exception e) {
      throw new RuntimeException("failed to serialize Arrow IPC", e);
    }
  }

  private static VectorSchemaRoot readIpc(byte[] bytes, BufferAllocator allocator) {
    try (ArrowStreamReader reader =
        new ArrowStreamReader(new ByteArrayInputStream(bytes), allocator)) {
      Schema schema = reader.getVectorSchemaRoot().getSchema();
      VectorSchemaRoot dest = VectorSchemaRoot.create(schema, allocator);
      VectorLoader loader = new VectorLoader(dest);
      VectorUnloader unloader = new VectorUnloader(reader.getVectorSchemaRoot());
      reader.loadNextBatch();
      try (ArrowRecordBatch batch = unloader.getRecordBatch()) {
        loader.load(batch);
      }
      return dest;
    } catch (Exception e) {
      throw new RuntimeException("failed to deserialize Arrow IPC", e);
    }
  }

  // -- native ---------------------------------------------------------------

  private static native byte[] nativeSampleRound0(
      Dataset dataset,
      String column,
      long target,
      String distanceType,
      long rngSeed,
      int[] fragmentIds);

  private static native byte[] nativeComputePartialStats(
      Dataset dataset, String column, byte[] centroidsIpc, String distanceType, int[] fragmentIds);

  private static native byte[] nativeMergePartialStats(byte[] a, byte[] b);

  private static native byte[] nativeReducePartialStats(byte[][] stats);

  private static native byte[] nativeFinalizeCentroids(byte[] stats, byte[] prev);

  private static native byte[] nativeSelectInitialCentroids(byte[][] samples, int k, long rngSeed);

  private static native byte[] nativeBootstrapCentroids(
      byte[][] samples, int k, String distanceType, long rngSeed);
}
