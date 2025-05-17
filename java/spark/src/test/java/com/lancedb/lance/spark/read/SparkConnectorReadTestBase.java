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

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SparkConnectorReadTestBase {

  protected static SparkSession spark;
  protected static String dbPath;
  protected static Dataset<Row> data;

  @BeforeAll
  static void setup() {
    spark =
        SparkSession.builder()
            .appName("spark-lance-connector-test")
            .master("local")
            .config("spark.sql.catalog.lance", "com.lancedb.lance.spark.LanceCatalog")
            .getOrCreate();
    dbPath = TestUtils.TestTable1Config.dbPath;
    data =
        spark
            .read()
            .format(LanceDataSource.name)
            .option(
                LanceConfig.CONFIG_DATASET_URI,
                LanceConfig.getDatasetUri(dbPath, TestUtils.TestTable1Config.datasetName))
            .load();
    data.createOrReplaceTempView("test_dataset1");
  }

  @AfterAll
  static void tearDown() {
    if (spark != null) {
      spark.stop();
    }
  }

  protected void validateData(Dataset<Row> data, List<List<Long>> expectedValues) {
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
}
