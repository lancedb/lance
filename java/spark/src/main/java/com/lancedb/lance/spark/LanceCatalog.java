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
import com.lancedb.lance.spark.internal.LanceDatasetAdapter;
import com.lancedb.lance.spark.utils.Optional;

import org.apache.arrow.vector.types.pojo.Field;
import org.apache.spark.sql.catalyst.analysis.NoSuchNamespaceException;
import org.apache.spark.sql.catalyst.analysis.NoSuchTableException;
import org.apache.spark.sql.catalyst.analysis.TableAlreadyExistsException;
import org.apache.spark.sql.connector.catalog.Identifier;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableCatalog;
import org.apache.spark.sql.connector.catalog.TableChange;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.apache.spark.sql.util.LanceArrowUtils;
import scala.Some;

import java.time.ZoneId;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class LanceCatalog implements TableCatalog {
  private CaseInsensitiveStringMap options;

  @Override
  public Identifier[] listTables(String[] namespace) throws NoSuchNamespaceException {
    throw new UnsupportedOperationException("Please use lancedb catalog for dataset listing");
  }

  @Override
  public Table loadTable(Identifier ident) throws NoSuchTableException {
    LanceConfig config = LanceConfig.from(options, ident.name());
    Optional<StructType> schema = LanceDatasetAdapter.getSchema(config);
    if (schema.isEmpty()) {
      throw new NoSuchTableException(config.getDbPath(), config.getDatasetName());
    }
    return new LanceDataset(config, schema.get());
  }

  @Override
  public Table createTable(
      Identifier ident, StructType schema, Transform[] partitions, Map<String, String> properties)
      throws TableAlreadyExistsException, NoSuchNamespaceException {
    try {
      LanceConfig config = LanceConfig.from(options, ident.name());
      WriteParams params = SparkOptions.genWriteParamsFromConfig(config);
      LanceDatasetAdapter.createDataset(ident.name(), schema, params);
    } catch (IllegalArgumentException e) {
      throw new TableAlreadyExistsException(ident.name(), new Some<>(e));
    }
    return new LanceDataset(LanceConfig.from(options, ident.name()), schema);
  }

  @Override
  public Table alterTable(Identifier ident, TableChange... changes) throws NoSuchTableException {
    LanceConfig config = LanceConfig.from(options, ident.name());
    try (Dataset ds = LanceDatasetAdapter.openDataset(config)) {
      for (TableChange change : changes) {
        if (change instanceof TableChange.DeleteColumn) {
          TableChange.DeleteColumn deleteColumn = (TableChange.DeleteColumn) change;
          ds.dropColumns(Arrays.asList(deleteColumn.fieldNames()));
        } else if (change instanceof TableChange.AddColumn) {
          TableChange.AddColumn addColumn = (TableChange.AddColumn) change;
          if (addColumn.defaultValue() != null) {
            throw new UnsupportedOperationException("Not support adding column with default value");
          }
          if (!addColumn.isNullable()) {
            throw new UnsupportedOperationException("Not support adding not null column");
          }
          if (addColumn.position() != null) {
            throw new UnsupportedOperationException("Not support adding column with position");
          }
          String timeZoneId = ZoneId.systemDefault().getId();
          List<Field> arrowFields =
              Arrays.stream(addColumn.fieldNames())
                  .map(
                      fieldName ->
                          LanceArrowUtils.toArrowField(
                              fieldName,
                              addColumn.dataType(),
                              addColumn.isNullable(),
                              timeZoneId,
                              false))
                  .collect(Collectors.toList());
          ds.addColumns(arrowFields, java.util.Optional.empty());
        } else {
          throw new UnsupportedOperationException();
        }
      }
    } catch (IllegalArgumentException e) {
      throw new NoSuchTableException(config.getDbPath(), config.getDatasetName());
    }
    return loadTable(ident);
  }

  @Override
  public boolean dropTable(Identifier ident) {
    LanceConfig config = LanceConfig.from(options, ident.name());
    LanceDatasetAdapter.dropDataset(config);
    return true;
  }

  @Override
  public void renameTable(Identifier oldIdent, Identifier newIdent)
      throws NoSuchTableException, TableAlreadyExistsException {
    throw new UnsupportedOperationException();
  }

  @Override
  public void initialize(String name, CaseInsensitiveStringMap options) {
    this.options = options;
  }

  @Override
  public String name() {
    return "lance";
  }
}
