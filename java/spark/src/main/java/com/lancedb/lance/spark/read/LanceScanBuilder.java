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
import com.lancedb.lance.spark.utils.Optional;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.connector.read.SupportsPushDownFilters;
import org.apache.spark.sql.connector.read.SupportsPushDownRequiredColumns;
import org.apache.spark.sql.sources.Filter;
import org.apache.spark.sql.types.StructType;

public class LanceScanBuilder implements ScanBuilder,
    SupportsPushDownRequiredColumns, SupportsPushDownFilters {
  private final LanceConfig options;
  private StructType schema;

  private Filter[] pushedFilters = new Filter[0];

  public LanceScanBuilder(StructType schema, LanceConfig options) {
    this.schema = schema;
    this.options = options;
  }

  @Override
  public Scan build() {
    Optional<String> whereCondition = FilterPushDown.compileFiltersToSqlWhereClause(pushedFilters);
    return new LanceScan(schema, options, whereCondition);
  }

  @Override
  public void pruneColumns(StructType requiredSchema) {
    if (!requiredSchema.isEmpty()) {
      // Get all columns if selecting columns empty(eg: resultDataFrame.count())
      this.schema = requiredSchema;
    }
  }

  @Override
  public Filter[] pushFilters(Filter[] filters) {
    if (!options.isPushDownFilters()) {
      return filters;
    }
    Filter[][] processFilters = FilterPushDown.processFilters(filters);
    pushedFilters = processFilters[0];
    return processFilters[1];
  }

  @Override
  public Filter[] pushedFilters() {
    return pushedFilters;
  }
}
