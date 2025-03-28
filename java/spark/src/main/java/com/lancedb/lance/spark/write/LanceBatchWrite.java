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
package com.lancedb.lance.spark.write;

import com.lancedb.lance.FragmentMetadata;
import com.lancedb.lance.spark.LanceConfig;
import com.lancedb.lance.spark.SparkOptions;
import com.lancedb.lance.spark.internal.LanceDatasetAdapter;

import org.apache.spark.sql.connector.write.BatchWrite;
import org.apache.spark.sql.connector.write.DataWriterFactory;
import org.apache.spark.sql.connector.write.PhysicalWriteInfo;
import org.apache.spark.sql.connector.write.WriterCommitMessage;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class LanceBatchWrite implements BatchWrite {
  private final StructType schema;
  private final LanceConfig config;
  private final boolean overwrite;

  public LanceBatchWrite(StructType schema, LanceConfig config, boolean overwrite) {
    this.schema = schema;
    this.config = config;
    this.overwrite = overwrite;
  }

  @Override
  public DataWriterFactory createBatchWriterFactory(PhysicalWriteInfo info) {
    return new LanceDataWriter.WriterFactory(schema, config);
  }

  @Override
  public boolean useCommitCoordinator() {
    return false;
  }

  @Override
  public void commit(WriterCommitMessage[] messages) {
    List<FragmentMetadata> fragments =
        Arrays.stream(messages)
            .map(m -> (TaskCommit) m)
            .map(TaskCommit::getFragments)
            .flatMap(List::stream)
            .collect(Collectors.toList());
    if (overwrite || SparkOptions.overwrite(this.config)) {
      LanceDatasetAdapter.overwriteFragments(config, fragments, schema);
    } else {
      LanceDatasetAdapter.appendFragments(config, fragments);
    }
  }

  @Override
  public void abort(WriterCommitMessage[] messages) {
    throw new UnsupportedOperationException();
  }

  @Override
  public String toString() {
    return String.format("LanceBatchWrite(datasetUri=%s)", config.getDatasetUri());
  }

  public static class TaskCommit implements WriterCommitMessage {
    private final List<FragmentMetadata> fragments;

    TaskCommit(List<FragmentMetadata> fragments) {
      this.fragments = fragments;
    }

    List<FragmentMetadata> getFragments() {
      return fragments;
    }
  }
}
