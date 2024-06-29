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
import com.lancedb.lance.spark.LanceDataSource;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.analysis.NoSuchTableException;
import org.apache.spark.sql.catalyst.analysis.TableAlreadyExistsException;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.col;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class SparkWriteTest {
  private static SparkSession spark;
  private static Dataset<Row> testData;
  @TempDir
  static Path dbPath;

  @BeforeAll
  static void setup() {
    spark = SparkSession.builder()
        .appName("spark-lance-connector-test")
        .master("local")
        .config("spark.sql.catalog.lance", "com.lancedb.lance.spark.LanceCatalog")
        .getOrCreate();
    StructType schema = new StructType(new StructField[]{
        DataTypes.createStructField("id", DataTypes.IntegerType, false),
        DataTypes.createStructField("name", DataTypes.StringType, false)
    });

    Row row1 = RowFactory.create(1, "Alice");
    Row row2 = RowFactory.create(2, "Bob");
    List<Row> data = Arrays.asList(row1, row2);

    testData = spark.createDataFrame(data, schema);
  }

  @AfterAll
  static void tearDown() {
    if (spark != null) {
      spark.stop();
    }
  }

  @Test
  public void defaultWrite(TestInfo testInfo) {
    String datasetName = testInfo.getTestMethod().get().getName();
    testData.write().format(LanceDataSource.name)
        .option(LanceConfig.CONFIG_DATASET_URI, LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
        .save();

    validateData(datasetName, 1);
  }

  @Test
  public void errorIfExists(TestInfo testInfo) {
    String datasetName = testInfo.getTestMethod().get().getName();
    testData.write().format(LanceDataSource.name)
        .option(LanceConfig.CONFIG_DATASET_URI, LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
        .save();

    assertThrows(TableAlreadyExistsException.class, () -> {
      testData.write().format(LanceDataSource.name)
          .option(LanceConfig.CONFIG_DATASET_URI, LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
          .save();
    });
  }

  @Test
  public void append(TestInfo testInfo) {
    String datasetName = testInfo.getTestMethod().get().getName();
    testData.write().format(LanceDataSource.name)
        .option(LanceConfig.CONFIG_DATASET_URI, LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
        .save();
    testData.write().format(LanceDataSource.name)
        .option(LanceConfig.CONFIG_DATASET_URI, LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
        .mode("append")
        .save();
    validateData(datasetName, 2);
  }

  @Test
  public void appendErrorIfNotExist(TestInfo testInfo) {
    String datasetName = testInfo.getTestMethod().get().getName();
    assertThrows(NoSuchTableException.class, () -> {
      testData.write().format(LanceDataSource.name)
          .option(LanceConfig.CONFIG_DATASET_URI, LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
          .mode("append")
          .save();
    });
  }

  @Test
  public void saveToPath(TestInfo testInfo) {
    String datasetName = testInfo.getTestMethod().get().getName();
    testData.write().format(LanceDataSource.name)
        .save(LanceConfig.getDatasetUri(dbPath.toString(), datasetName));

    validateData(datasetName, 1);
  }

  @Disabled("Do not support overwrite")
  @Test
  public void overwrite(TestInfo testInfo) {
    String datasetName = testInfo.getTestMethod().get().getName();
    testData.write().format(LanceDataSource.name)
        .option(LanceConfig.CONFIG_DATASET_URI, LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
        .save();
    testData.write().format(LanceDataSource.name)
        .option(LanceConfig.CONFIG_DATASET_URI, LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
        .mode("overwrite")
        .save();

    validateData(datasetName, 1);
  }

  private void validateData(String datasetName, int iteration) {
    Dataset<Row> data = spark.read().format("lance")
        .option(LanceConfig.CONFIG_DATASET_URI, LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
        .load();

    assertEquals(2 * iteration, data.count());
    assertEquals(iteration, data.filter(col("id").equalTo(1)).count());
    assertEquals(iteration, data.filter(col("id").equalTo(2)).count());

    Dataset<Row> data1 = data.filter(col("id").equalTo(1)).select("name");
    Dataset<Row> data2 = data.filter(col("id").equalTo(2)).select("name");

    for (Row row : data1.collectAsList()) {
      assertEquals("Alice", row.getString(0));
    }

    for (Row row : data2.collectAsList()) {
      assertEquals("Bob", row.getString(0));
    }
  }
}