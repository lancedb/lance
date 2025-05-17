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
import org.apache.spark.sql.catalyst.analysis.NoSuchTableException;
import org.apache.spark.sql.catalyst.analysis.TableAlreadyExistsException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;

import java.io.File;

import static org.apache.spark.sql.functions.col;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class SparkConnectorWriteTest extends SparkConnectorWriteTestBase {

  @Test
  public void defaultWrite(TestInfo testInfo) {
    String datasetName = testInfo.getTestMethod().get().getName();
    testData
        .write()
        .format(LanceDataSource.name)
        .option(
            LanceConfig.CONFIG_DATASET_URI,
            LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
        .save();

    validateData(datasetName, 1);
  }

  @Test
  public void errorIfExists(TestInfo testInfo) {
    String datasetName = testInfo.getTestMethod().get().getName();
    testData
        .write()
        .format(LanceDataSource.name)
        .option(
            LanceConfig.CONFIG_DATASET_URI,
            LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
        .save();

    assertThrows(
        TableAlreadyExistsException.class,
        () -> {
          testData
              .write()
              .format(LanceDataSource.name)
              .option(
                  LanceConfig.CONFIG_DATASET_URI,
                  LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
              .save();
        });
  }

  @Test
  public void append(TestInfo testInfo) {
    String datasetName = testInfo.getTestMethod().get().getName();
    testData
        .write()
        .format(LanceDataSource.name)
        .option(
            LanceConfig.CONFIG_DATASET_URI,
            LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
        .save();
    testData
        .write()
        .format(LanceDataSource.name)
        .option(
            LanceConfig.CONFIG_DATASET_URI,
            LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
        .mode("append")
        .save();
    validateData(datasetName, 2);
  }

  @Test
  public void appendErrorIfNotExist(TestInfo testInfo) {
    String datasetName = testInfo.getTestMethod().get().getName();
    assertThrows(
        NoSuchTableException.class,
        () -> {
          testData
              .write()
              .format(LanceDataSource.name)
              .option(
                  LanceConfig.CONFIG_DATASET_URI,
                  LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
              .mode("append")
              .save();
        });
  }

  @Test
  public void saveToPath(TestInfo testInfo) {
    String datasetName = testInfo.getTestMethod().get().getName();
    testData
        .write()
        .format(LanceDataSource.name)
        .save(LanceConfig.getDatasetUri(dbPath.toString(), datasetName));

    validateData(datasetName, 1);
  }

  @Test
  public void overwrite(TestInfo testInfo) {
    String datasetName = testInfo.getTestMethod().get().getName();
    testData
        .write()
        .format(LanceDataSource.name)
        .option(
            LanceConfig.CONFIG_DATASET_URI,
            LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
        .save();
    testData
        .write()
        .format(LanceDataSource.name)
        .option(
            LanceConfig.CONFIG_DATASET_URI,
            LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
        .mode("overwrite")
        .save();

    validateData(datasetName, 1);
  }

  @Test
  public void appendAfterOverwrite(TestInfo testInfo) {
    String datasetName = testInfo.getTestMethod().get().getName();
    testData
        .write()
        .format(LanceDataSource.name)
        .option(
            LanceConfig.CONFIG_DATASET_URI,
            LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
        .save();
    testData
        .write()
        .format(LanceDataSource.name)
        .option(
            LanceConfig.CONFIG_DATASET_URI,
            LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
        .mode("overwrite")
        .save();
    testData
        .write()
        .format(LanceDataSource.name)
        .option(
            LanceConfig.CONFIG_DATASET_URI,
            LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
        .mode("append")
        .save();
    validateData(datasetName, 2);
  }

  @Test
  public void writeMultiFiles(TestInfo testInfo) {
    String datasetName = testInfo.getTestMethod().get().getName();
    String filePath = LanceConfig.getDatasetUri(dbPath.toString(), datasetName);
    testData
        .write()
        .format(LanceDataSource.name)
        .option(LanceConfig.CONFIG_DATASET_URI, filePath)
        .save();

    validateData(datasetName, 1);
    File directory = new File(filePath + "/data");
    assertEquals(2, directory.listFiles().length);
  }

  @Test
  public void writeEmptyTaskFiles(TestInfo testInfo) {
    String datasetName = testInfo.getTestMethod().get().getName();
    String filePath = LanceConfig.getDatasetUri(dbPath.toString(), datasetName);
    testData
        .repartition(4)
        .write()
        .format(LanceDataSource.name)
        .option(LanceConfig.CONFIG_DATASET_URI, filePath)
        .save();

    File directory = new File(filePath + "/data");
    assertEquals(2, directory.listFiles().length);
  }

  private void validateData(String datasetName, int iteration) {
    Dataset<Row> data =
        spark
            .read()
            .format("lance")
            .option(
                LanceConfig.CONFIG_DATASET_URI,
                LanceConfig.getDatasetUri(dbPath.toString(), datasetName))
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

  @Test
  public void dropAndReplaceTable(TestInfo testInfo) {
    String datasetName = testInfo.getTestMethod().get().getName();
    String path = LanceConfig.getDatasetUri(dbPath.toString(), datasetName);
    spark.sql("CREATE OR REPLACE TABLE lance.`" + path + "` AS SELECT * FROM tmp_view");
    spark.sql("CREATE OR REPLACE TABLE lance.`" + path + "` AS SELECT * FROM tmp_view");
    spark.sql("DROP TABLE lance.`" + path + "`");
  }

  @Test
  public void appendAndSelectData(TestInfo testInfo) {
    String datasetName = testInfo.getTestMethod().get().getName();
    String path = LanceConfig.getDatasetUri(dbPath.toString(), datasetName);
    spark.sql("CREATE OR REPLACE TABLE lance.`" + path + "` AS SELECT * FROM tmp_view");

    //    Dataset<Row> data = spark.sql("SELECT * FROM lance.`" + path + "`");
    //    assertEquals(2, data.count());
    spark.sql("INSERT INTO TABLE lance.`" + path + "` VALUES (3, 'Charlie')");
    Dataset<Row> data = spark.sql("SELECT * FROM lance.`" + path + "`");
    assertEquals(3, data.count());
    spark.sql("DROP TABLE lance.`" + path + "`");
  }
}
