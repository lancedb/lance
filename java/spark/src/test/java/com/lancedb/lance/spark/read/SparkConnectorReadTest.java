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
import com.lancedb.lance.spark.LanceDataSource;
import com.lancedb.lance.spark.TestUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SparkConnectorReadTest {
  private static SparkSession spark;
  private static String dbPath;
  private static Dataset<Row> data;

  @BeforeAll
  static void setup() {
    spark = SparkSession.builder()
        .appName("spark-lance-connector-test")
        .master("local")
        .config("spark.sql.catalog.lance", "com.lancedb.lance.spark.LanceCatalog")
        .getOrCreate();
    dbPath = TestUtils.TestTable1Config.dbPath;
    data = spark.read().format(LanceDataSource.name)
        .option(LanceConfig.CONFIG_TABLE_PATH, LanceConfig.getTablePath(dbPath, TestUtils.TestTable1Config.tableName))
        .load();
  }

  @AfterAll
  static void tearDown() {
    if (spark != null) {
      spark.stop();
    }
  }

  private void validateData(Dataset<Row> data, List<List<Long>> expectedValues) {
    List<Row> rows = data.collectAsList();
    assertEquals(expectedValues.size(), rows.size());

    for (int i = 0; i < rows.size(); i++) {
      Row row = rows.get(i);
      List<Long> expectedRow = expectedValues.get(i);
      assertEquals(expectedRow.size(), row.size());

      for (int j = 0; j < expectedRow.size(); j++) {
        long expectedValue = expectedRow.get(j);
        long actualValue = row.getLong(j);
        assertEquals(expectedValue, actualValue, "Mismatch at row " + i + " column " + j);
      }
    }
  }

  @Test
  public void readAll() {
    validateData(data, TestUtils.TestTable1Config.expectedValues);
  }

  @Test
  public void filter() {
    validateData(data.filter("x > 1"), TestUtils.TestTable1Config.expectedValues.stream()
        .filter(row -> row.get(0) > 1)
        .collect(Collectors.toList()));
    validateData(data.filter("y == 4"), TestUtils.TestTable1Config.expectedValues.stream()
        .filter(row -> row.get(1) == 4)
        .collect(Collectors.toList()));
    validateData(data.filter("b >= 6"), TestUtils.TestTable1Config.expectedValues.stream()
        .filter(row -> row.get(2) >= 6)
        .collect(Collectors.toList()));
    validateData(data.filter("c < -1"), TestUtils.TestTable1Config.expectedValues.stream()
        .filter(row -> row.get(3) < -1)
        .collect(Collectors.toList()));
    validateData(data.filter("c <= -1"), TestUtils.TestTable1Config.expectedValues.stream()
        .filter(row -> row.get(3) <= -1)
        .collect(Collectors.toList()));
    validateData(data.filter("c == -2"), TestUtils.TestTable1Config.expectedValues.stream()
        .filter(row -> row.get(3) == -2)
        .collect(Collectors.toList()));
    validateData(data.filter("x > 1").filter("y < 6"), TestUtils.TestTable1Config.expectedValues.stream()
        .filter(row -> row.get(0) > 1)
        .filter(row -> row.get(1) < 6)
        .collect(Collectors.toList()));
    validateData(data.filter("x > 1 and y < 6"), TestUtils.TestTable1Config.expectedValues.stream()
        .filter(row -> row.get(0) > 1)
        .filter(row -> row.get(1) < 6)
        .collect(Collectors.toList()));
    validateData(data.filter("x > 1 or y < 6"), TestUtils.TestTable1Config.expectedValues.stream()
        .filter(row -> (row.get(0) > 1) || (row.get(1) < 6))
        .collect(Collectors.toList()));
    validateData(data.filter("(x >= 1 and x <= 2) or (c >= -2 and c < 0)"), TestUtils.TestTable1Config.expectedValues.stream()
        .filter(row -> (row.get(0) >= 1 && row.get(0) <= 2) || (row.get(3) >= -2 && row.get(3) < 0))
        .collect(Collectors.toList()));
  }

  @Test
  public void select() {
    validateData(data.select("y", "b"), TestUtils.TestTable1Config.expectedValues.stream()
        .map(row -> Arrays.asList(row.get(1), row.get(2)))
        .collect(Collectors.toList()));
  }

  @Test
  public void filterSelect() {
    validateData(data.select("y", "b").filter("y > 3"),
        TestUtils.TestTable1Config.expectedValues.stream()
            .map(row -> Arrays.asList(row.get(1), row.get(2))) // "y" is at index 1, "b" is at index 2
            .filter(row -> row.get(0) > 3)
            .collect(Collectors.toList()));
  }
  
  // TODO(lu) support spark.read().format("lance")
  //    .load(dbPath.resolve(tableName).toString());
}
