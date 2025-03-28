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
package com.lancedb.lance.spark;

import com.lancedb.lance.spark.read.LanceScanBuilder;
import com.lancedb.lance.spark.write.SparkWrite;

import com.google.common.collect.ImmutableSet;
import org.apache.spark.sql.connector.catalog.MetadataColumn;
import org.apache.spark.sql.connector.catalog.SupportsMetadataColumns;
import org.apache.spark.sql.connector.catalog.SupportsRead;
import org.apache.spark.sql.connector.catalog.SupportsWrite;
import org.apache.spark.sql.connector.catalog.TableCapability;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.connector.write.LogicalWriteInfo;
import org.apache.spark.sql.connector.write.WriteBuilder;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

import java.util.Set;

/** Lance Spark Dataset. */
public class LanceDataset implements SupportsRead, SupportsWrite, SupportsMetadataColumns {
  private static final Set<TableCapability> CAPABILITIES =
      ImmutableSet.of(
          TableCapability.BATCH_READ, TableCapability.BATCH_WRITE, TableCapability.TRUNCATE);

  public static final MetadataColumn[] METADATA_COLUMNS =
      new MetadataColumn[] {
        new MetadataColumn() {
          @Override
          public String name() {
            return LanceConstant.ROW_ID;
          }

          @Override
          public DataType dataType() {
            return DataTypes.LongType;
          }
        },
        new MetadataColumn() {
          @Override
          public String name() {
            return LanceConstant.ROW_ADDRESS;
          }

          @Override
          public DataType dataType() {
            return DataTypes.LongType;
          }
        },
      };

  LanceConfig config;
  private final StructType sparkSchema;

  /**
   * Creates a Lance dataset.
   *
   * @param config read config
   * @param sparkSchema spark struct type
   */
  public LanceDataset(LanceConfig config, StructType sparkSchema) {
    this.config = config;
    this.sparkSchema = sparkSchema;
  }

  @Override
  public ScanBuilder newScanBuilder(CaseInsensitiveStringMap caseInsensitiveStringMap) {
    return new LanceScanBuilder(sparkSchema, config);
  }

  @Override
  public String name() {
    return this.config.getDatasetName();
  }

  @Override
  public StructType schema() {
    return sparkSchema;
  }

  @Override
  public Set<TableCapability> capabilities() {
    return CAPABILITIES;
  }

  @Override
  public WriteBuilder newWriteBuilder(LogicalWriteInfo logicalWriteInfo) {
    return new SparkWrite.SparkWriteBuilder(sparkSchema, config);
  }

  @Override
  public MetadataColumn[] metadataColumns() {
    return METADATA_COLUMNS;
  }
}
