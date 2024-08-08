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
import com.lancedb.lance.spark.internal.LanceDatasetAdapter;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.write.DataWriter;
import org.apache.spark.sql.connector.write.DataWriterFactory;
import org.apache.spark.sql.connector.write.WriterCommitMessage;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;

public class LanceDataWriter implements DataWriter<InternalRow> {
  private LanceArrowWriter arrowWriter;
  private FutureTask<FragmentMetadata> fragmentCreationTask;
  private Thread fragmentCreationThread;

  private LanceDataWriter(LanceArrowWriter arrowWriter,
      FutureTask<FragmentMetadata> fragmentCreationTask, Thread fragmentCreationThread) {
    // TODO support write to multiple fragments
    this.arrowWriter = arrowWriter;
    this.fragmentCreationThread = fragmentCreationThread;
    this.fragmentCreationTask = fragmentCreationTask;
  }

  @Override
  public void write(InternalRow record) throws IOException {
    arrowWriter.write(record);
  }

  @Override
  public WriterCommitMessage commit() throws IOException {
    arrowWriter.setFinished();
    try {
      FragmentMetadata fragmentMetadata = fragmentCreationTask.get();
      return new BatchAppend.TaskCommit(Arrays.asList(fragmentMetadata));
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted while waiting for reader thread to finish", e);
    } catch (ExecutionException e) {
      throw new IOException("Exception in reader thread", e);
    }
  }

  @Override
  public void abort() throws IOException {
    fragmentCreationThread.interrupt();
    try {
      fragmentCreationTask.get();
    } catch (InterruptedException | ExecutionException e) {
      throw new IOException("Failed to abort the reader thread", e);
    }
    close();
  }

  @Override
  public void close() throws IOException {
    arrowWriter.close();
  }

  public static class WriterFactory implements DataWriterFactory {
    private final LanceConfig config;
    private final StructType schema;

    protected WriterFactory(StructType schema, LanceConfig config) {
      // Everything passed to writer factory should be serializable
      this.schema = schema;
      this.config = config;
    }

    @Override
    public DataWriter<InternalRow> createWriter(int partitionId, long taskId) {
      LanceArrowWriter arrowWriter = LanceDatasetAdapter.getArrowWriter(schema, 1024);
      Callable<FragmentMetadata> fragmentCreator
          = () -> LanceDatasetAdapter.createFragment(config.getDatasetUri(), arrowWriter);
      FutureTask<FragmentMetadata> fragmentCreationTask = new FutureTask<>(fragmentCreator);
      Thread fragmentCreationThread = new Thread(fragmentCreationTask);
      fragmentCreationThread.start();

      return new LanceDataWriter(arrowWriter, fragmentCreationTask, fragmentCreationThread);
    }
  }
}