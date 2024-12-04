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

import com.lancedb.lance.spark.read.LanceInputPartition;

import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.spark.sql.vectorized.ArrowColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;

import java.io.IOException;

public class LanceFragmentColumnarBatchScanner implements AutoCloseable {
  private final LanceFragmentScanner fragmentScanner;
  private final ArrowReader arrowReader;
  private ColumnarBatch currentColumnarBatch;

  public LanceFragmentColumnarBatchScanner(
      LanceFragmentScanner fragmentScanner, ArrowReader arrowReader) {
    this.fragmentScanner = fragmentScanner;
    this.arrowReader = arrowReader;
  }

  public static LanceFragmentColumnarBatchScanner create(
      int fragmentId, LanceInputPartition inputPartition) {
    LanceFragmentScanner fragmentScanner =
        LanceDatasetAdapter.getFragmentScanner(fragmentId, inputPartition);
    return new LanceFragmentColumnarBatchScanner(fragmentScanner, fragmentScanner.getArrowReader());
  }

  public boolean loadNextBatch() throws IOException {
    // Batch close must be earlier than load next batch
    if (currentColumnarBatch != null) {
      currentColumnarBatch.close();
    }
    if (arrowReader.loadNextBatch()) {
      VectorSchemaRoot root = arrowReader.getVectorSchemaRoot();
      currentColumnarBatch =
          new ColumnarBatch(
              root.getFieldVectors().stream()
                  .map(ArrowColumnVector::new)
                  .toArray(ArrowColumnVector[]::new),
              root.getRowCount());
      return true;
    }
    return false;
  }

  /** @return the current batch, the caller responsible for closing the batch */
  public ColumnarBatch getCurrentBatch() {
    return currentColumnarBatch;
  }

  @Override
  public void close() throws IOException {
    if (currentColumnarBatch != null) {
      currentColumnarBatch.close();
    }
    if (currentColumnarBatch != null) {
      currentColumnarBatch.close();
    }
    arrowReader.close();
    fragmentScanner.close();
  }
}
