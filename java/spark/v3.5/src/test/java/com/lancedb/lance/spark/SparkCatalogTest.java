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

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.catalyst.analysis.NoSuchTableException;
import org.apache.spark.sql.catalyst.analysis.TableAlreadyExistsException;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.TimeoutException;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SparkCatalogTest {
  private static final String CATALOG = "my_catalog";
  private static final String TABLE_PREFIX = CATALOG + ".default.";
  @TempDir
  static Path warehouse;

  private static SparkSession spark;

  @BeforeAll
  static void setup() {
    spark = SparkSession.builder()
        .appName("SparkCatalogTest")
        .master("local")
        .config("spark.sql.catalog." + CATALOG, "com.lancedb.lance.spark.SparkCatalog")
        .config("spark.sql.catalog." + CATALOG + ".warehouse", warehouse.toString())
        .getOrCreate();
  }

  @AfterAll
  static void tearDown() {
    if (spark != null) {
      spark.stop();
    }
  }

  @Test
  public void testCreate() {
    String tableName = "lance_create_table";
    createTable(TABLE_PREFIX + tableName);
    String datasetPath = warehouse.resolve(tableName).toString();
    try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
         com.lancedb.lance.Dataset dataset = com.lancedb.lance.Dataset.open(datasetPath, allocator)) {
      assertEquals(1, dataset.version());
      assertEquals(0, dataset.countRows());
    }
  }

  @Test
  public void testInsert() {
    String tableName = "lance_insert_table";
    createTable(TABLE_PREFIX + tableName);
    spark.sql("INSERT INTO " + TABLE_PREFIX + tableName
        + " VALUES ('100', '2015-01-01', '2015-01-01T13:51:39.340396Z'), ('101', '2015-01-01', '2015-01-01T12:14:58.597216Z')");
    String datasetPath = warehouse.resolve(tableName).toString();
    try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
         com.lancedb.lance.Dataset dataset = com.lancedb.lance.Dataset.open(datasetPath, allocator)) {
      assertEquals(2, dataset.version());
      assertEquals(2, dataset.countRows());
    }
  }

  @Test
  public void testBatchWriteTable() throws TableAlreadyExistsException {
    Dataset<Row> data = createSparkDataFrame();
    String tableName = "lance_bath_write_table";
    data.writeTo(TABLE_PREFIX + tableName).using("lance").create();
    String datasetPath = warehouse.resolve(tableName).toString();
    try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
         com.lancedb.lance.Dataset dataset = com.lancedb.lance.Dataset.open(datasetPath, allocator)) {
      assertEquals(2, dataset.version());
      assertEquals(4, dataset.countRows());
    }
  }

  @Test
  public void testBatchAppendTable() throws NoSuchTableException {
    Dataset<Row> data = createSparkDataFrame();
    String tableName = "lance_batch_append_table";
    String fullName = TABLE_PREFIX + "lance_batch_append_table";
    createTable(fullName);
    data.writeTo(fullName).append();
    String datasetPath = warehouse.resolve(tableName).toString();
    try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
         com.lancedb.lance.Dataset dataset = com.lancedb.lance.Dataset.open(datasetPath, allocator)) {
      assertEquals(2, dataset.version());
      assertEquals(4, dataset.countRows());
    }
  }

  @Test
  @Disabled
  public void testDrop() {
    String tableName = TABLE_PREFIX + "lance_drop_table";
    createTable(tableName);
    spark.sql("DROP TABLE " + tableName);
  }

  @Test
  @Disabled
  public void testScan() {
    String tableName = TABLE_PREFIX + "lance_insert_table";
    createTable(tableName);
    spark.sql("INSERT INTO " + tableName
        + " VALUES ('100', '2015-01-01', '2015-01-01T13:51:39.340396Z'), ('101', '2015-01-01', '2015-01-01T12:14:58.597216Z')");
    // Workflow: loadTable
    // -> [Gap] SparkScanBuilder.pushAggregation().build()
    // -> [Gap] LocalScan.readSchema() -> [Gap] LocalScan.rows[]
    spark.sql("SELECT * FROM " + tableName).show();
    spark.sql("SELECT COUNT(*) FROM " + tableName).show();
    spark.table(tableName).show();
  }

  @Test
  @Disabled
  public void testStreamingWriteTable() throws TimeoutException {
    Dataset<Row> data = createSparkDataFrame();
    String tableName = TABLE_PREFIX + "lance_streaming_table";
    data.writeStream().format("lance").outputMode("append").toTable(tableName);
    spark.table(tableName).show();
  }

  @Test
  @Disabled
  public void testStreamingAppendTable() throws TimeoutException {
    Dataset<Row> data = createSparkDataFrame();
    String tableName = TABLE_PREFIX + "lance_streaming_append_table";
    createTable(tableName);
    data.writeStream().format("lance").outputMode("append").toTable(tableName);
    spark.table(tableName).show();
  }

  private Dataset<Row> createSparkDataFrame() {
    StructType schema = new StructType(new StructField[]{
        DataTypes.createStructField("id", DataTypes.StringType, false),
        DataTypes.createStructField("creation_date", DataTypes.StringType, false),
        DataTypes.createStructField("last_update_time", DataTypes.StringType, false)
    });
    return spark.createDataFrame(java.util.Arrays.asList(
        RowFactory.create("100", "2015-01-01", "2015-01-01T13:51:39.340396Z"),
        RowFactory.create("101", "2015-01-01", "2015-01-01T12:14:58.597216Z"),
        RowFactory.create("102", "2015-01-01", "2015-01-01T13:51:40.417052Z"),
        RowFactory.create("103", "2015-01-01", "2015-01-01T13:51:40.519832Z")
    ), schema);
  }

  private void createTable(String tableName) {
    spark.sql("CREATE TABLE IF NOT EXISTS " + tableName +
        "(id STRING, " +
        "creation_date STRING, " +
        "last_update_time STRING) " +
        "USING lance");
  }
}
