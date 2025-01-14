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

import com.lancedb.lance.spark.LanceConfig;

import org.apache.spark.sql.connector.write.BatchWrite;
import org.apache.spark.sql.connector.write.SupportsTruncate;
import org.apache.spark.sql.connector.write.Write;
import org.apache.spark.sql.connector.write.WriteBuilder;
import org.apache.spark.sql.connector.write.streaming.StreamingWrite;
import org.apache.spark.sql.types.StructType;

/** Spark write builder. */
public class SparkWrite implements Write {
  private final LanceConfig config;
  private final StructType schema;
  private final boolean overwrite;

  SparkWrite(StructType schema, LanceConfig config, boolean overwrite) {
    this.schema = schema;
    this.config = config;
    this.overwrite = overwrite;
  }

  @Override
  public BatchWrite toBatch() {
    return new LanceBatchWrite(schema, config, overwrite);
  }

  @Override
  public StreamingWrite toStreaming() {
    throw new UnsupportedOperationException();
  }

  /** Task commit. */
  public static class SparkWriteBuilder implements SupportsTruncate, WriteBuilder {
    private final LanceConfig config;
    private final StructType schema;
    private boolean overwrite = false;

    public SparkWriteBuilder(StructType schema, LanceConfig config) {
      this.schema = schema;
      this.config = config;
    }

    @Override
    public Write build() {
      return new SparkWrite(schema, config, overwrite);
    }

    @Override
    public WriteBuilder truncate() {
      this.overwrite = true;
      return this;
    }
  }
}
