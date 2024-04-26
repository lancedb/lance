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

import com.lancedb.lance.Dataset;
import com.lancedb.lance.WriteParams;
import com.lancedb.lance.spark.source.SparkTable;
import java.nio.file.Path;
import java.time.ZoneId;
import java.util.Map;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.spark.sql.catalyst.analysis.NoSuchTableException;
import org.apache.spark.sql.connector.catalog.Identifier;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableCatalog;
import org.apache.spark.sql.connector.catalog.TableChange;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.ArrowUtils;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * Lance Spark Catalog.
 */
public class SparkCatalog implements TableCatalog {
  private static final BufferAllocator rootAllocator = new RootAllocator(Long.MAX_VALUE);
  private Path warehouse = null;

  public static BufferAllocator newChildAllocator(String name, long initialReservation,
      long maxAllocation) {
    return rootAllocator.newChildAllocator(name, initialReservation, maxAllocation);
  }

  @Override
  public Identifier[] listTables(String[] strings) {
    throw new UnsupportedOperationException("Lance spark listTables");
  }

  @Override
  public Table loadTable(Identifier identifier) throws NoSuchTableException {
    String datasetUri = warehouse.resolve(identifier.name()).toString();
    try (BufferAllocator allocator = newChildAllocator(
        "load table reader for Lance", 0, Long.MAX_VALUE);
        Dataset dataset = Dataset.open(datasetUri, allocator)) {
      // TODO(lu) Support type e.g. FixedSizeListArray
      return new SparkTable(datasetUri, identifier.name(), ArrowUtils.fromArrowSchema(
          dataset.getSchema()));
    } catch (RuntimeException e) {
      throw new NoSuchTableException(identifier);
    }
  }

  @Override
  public Table createTable(Identifier identifier, StructType structType,
      Transform[] transforms, Map<String, String> map) {
    String datasetUri = warehouse.resolve(identifier.name()).toString();
    Schema arrowSchema = ArrowUtils.toArrowSchema(
        structType, ZoneId.systemDefault().getId(), true, false);
    try (BufferAllocator allocator = newChildAllocator(
        "create table loader for Lance", 0, Long.MAX_VALUE)) {
      Dataset.create(allocator, datasetUri, arrowSchema,
          new WriteParams.Builder().build()).close();
      return new SparkTable(datasetUri, identifier.name(), structType);
    }
  }

  @Override
  public Table alterTable(Identifier identifier, TableChange... tableChanges) {
    throw new UnsupportedOperationException("Lance spark alterTable");
  }

  @Override
  public boolean dropTable(Identifier identifier) {
    throw new UnsupportedOperationException("Lance spark dropTable");
  }

  @Override
  public void renameTable(Identifier identifier, Identifier identifier1) {
    throw new UnsupportedOperationException("Lance spark renameTable");
  }

  @Override
  public void initialize(String s, CaseInsensitiveStringMap caseInsensitiveStringMap) {
    this.warehouse = Path.of(caseInsensitiveStringMap.get("warehouse"));
  }

  @Override
  public String name() {
    return "lance";
  }
}
