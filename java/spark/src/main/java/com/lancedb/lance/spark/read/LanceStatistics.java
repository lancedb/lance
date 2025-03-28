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
package com.lancedb.lance.spark.read;

import com.lancedb.lance.spark.LanceConfig;
import com.lancedb.lance.spark.internal.LanceDatasetAdapter;
import com.lancedb.lance.spark.utils.Optional;

import org.apache.spark.sql.connector.read.Statistics;

import java.util.OptionalLong;

public class LanceStatistics implements Statistics {
  private final Optional<Long> rowNumber;
  private final Optional<Long> dataBytesSize;

  public LanceStatistics(LanceConfig config) {
    this.rowNumber = LanceDatasetAdapter.getDatasetRowCount(config);
    this.dataBytesSize = LanceDatasetAdapter.getDatasetDataSize(config);
  }

  @Override
  public OptionalLong sizeInBytes() {
    if (dataBytesSize.isPresent()) {
      return OptionalLong.of(dataBytesSize.get());
    } else {
      return OptionalLong.empty();
    }
  }

  @Override
  public OptionalLong numRows() {
    if (rowNumber.isPresent()) {
      return OptionalLong.of(rowNumber.get());
    } else {
      return OptionalLong.empty();
    }
  }
}
