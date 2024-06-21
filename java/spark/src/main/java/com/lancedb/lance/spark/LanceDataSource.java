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

import com.lancedb.lance.spark.internal.LanceDatasetAdapter;
import com.lancedb.lance.spark.utils.Optional;
import org.apache.spark.sql.connector.catalog.Identifier;
import org.apache.spark.sql.connector.catalog.SupportsCatalogOptions;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.sources.DataSourceRegister;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

import java.util.Map;

public class LanceDataSource implements SupportsCatalogOptions, DataSourceRegister {
  public static final String name = "lance";

  @Override
  public StructType inferSchema(CaseInsensitiveStringMap options) {
    Optional<StructType> schema = LanceDatasetAdapter.getSchema(LanceConfig.from(options));
    return schema.isPresent() ? schema.get() : null;
  }

  @Override
  public Table getTable(StructType schema, Transform[] partitioning,
      Map<String, String> properties) {
    return new LanceTable(LanceConfig.from(properties), schema);
  }

  @Override
  public String shortName() {
    return name;
  }

  @Override
  public Identifier extractIdentifier(CaseInsensitiveStringMap options) {
    return new LanceIdentifier(LanceConfig.from(options).getTablePath());
  }

  @Override
  public String extractCatalog(CaseInsensitiveStringMap options) {
    return "lance";
  }
}
