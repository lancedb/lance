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

package com.lancedb.lance.spark.internal;

import com.lancedb.lance.Dataset;
import com.lancedb.lance.DatasetFragment;
import com.lancedb.lance.Fragment;
import com.lancedb.lance.FragmentMetadata;
import com.lancedb.lance.FragmentOperation;
import com.lancedb.lance.WriteParams;
import com.lancedb.lance.spark.LanceConfig;
import com.lancedb.lance.spark.read.LanceInputPartition;
import com.lancedb.lance.spark.utils.Optional;
import com.lancedb.lance.spark.write.LanceArrowWriter;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.ArrowUtils;

import java.time.ZoneId;
import java.util.List;
import java.util.stream.Collectors;

public class LanceDatasetAdapter {
  private static final BufferAllocator allocator = new RootAllocator(
      RootAllocator.configBuilder().from(RootAllocator.defaultConfig())
          .maxAllocation(4 * 1024 * 1024).build());

  public static Optional<StructType> getSchema(LanceConfig config) {
    return getSchema(config.getDatasetUri());
  }

  public static Optional<StructType> getSchema(String datasetUri) {
    try (Dataset dataset = Dataset.open(datasetUri, allocator)) {
      return Optional.of(ArrowUtils.fromArrowSchema(dataset.getSchema()));
    } catch (IllegalArgumentException e) {
      // dataset not found
      return Optional.empty();
    }
  }

  public static List<Integer> getFragmentIds(LanceConfig config) {
    try (Dataset dataset = Dataset.open(config.getDatasetUri(), allocator)) {
      return dataset.getFragments().stream()
          .map(DatasetFragment::getId).collect(Collectors.toList());
    }
  }

  public static LanceFragmentScanner getFragmentScanner(int fragmentId,
      LanceInputPartition inputPartition) {
    return LanceFragmentScanner.create(fragmentId, inputPartition, allocator);
  }

  public static void appendFragments(LanceConfig config, List<FragmentMetadata> fragments) {
    FragmentOperation.Append appendOp = new FragmentOperation.Append(fragments);
    try (Dataset datasetRead = Dataset.open(config.getDatasetUri(), allocator)) {
      Dataset.commit(allocator, config.getDatasetUri(),
          appendOp, java.util.Optional.of(datasetRead.version())).close();
    }
  }

  public static LanceArrowWriter getArrowWriter(StructType sparkSchema, int batchSize) {
    return new LanceArrowWriter(allocator,
        ArrowUtils.toArrowSchema(sparkSchema, "UTC", false, false), batchSize);
  }

  public static FragmentMetadata createFragment(String datasetUri, ArrowReader reader) {
    try (ArrowArrayStream arrowStream = ArrowArrayStream.allocateNew(allocator)) {
      Data.exportArrayStream(allocator, reader, arrowStream);
      return Fragment.create(datasetUri, arrowStream,
          java.util.Optional.empty(), new WriteParams.Builder().build());
    }
  }

  public static void createDataset(String datasetUri, StructType sparkSchema) {
    Dataset.create(allocator, datasetUri,
        ArrowUtils.toArrowSchema(sparkSchema, ZoneId.systemDefault().getId(), true, false),
        new WriteParams.Builder().build()).close();
  }
}
