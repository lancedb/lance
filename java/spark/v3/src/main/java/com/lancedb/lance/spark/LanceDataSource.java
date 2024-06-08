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

import com.lancedb.lance.spark.internal.LanceConfig;
import com.lancedb.lance.spark.internal.LanceReader;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableProvider;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.sources.DataSourceRegister;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

import java.util.Map;

public class LanceDataSource implements TableProvider, DataSourceRegister {
  private static final String name = "lance";

  @Override
  public StructType inferSchema(CaseInsensitiveStringMap options) {
    // Given options help identify a table, no schema filter is passed in
    return LanceReader.getSchema(LanceConfig.from(options));
  }

  @Override
  public Table getTable(StructType schema, Transform[] partitioning,
      Map<String, String> properties) {
    LanceConfig config = LanceConfig.from(properties);
    return new LanceTable(config, LanceReader.getSchema(config));
  }

  @Override
  public boolean supportsExternalMetadata() {
    return TableProvider.super.supportsExternalMetadata();
  }

  @Override
  public String shortName() {
    return name;
  }
}
