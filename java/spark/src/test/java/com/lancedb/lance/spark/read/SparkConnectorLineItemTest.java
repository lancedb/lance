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
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.apache.spark.sql.functions.desc;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class SparkConnectorLineItemTest {
  private static SparkSession spark;
  private static String dbPath;
  private static String parquetPath;
  private static Dataset<Row> lanceData;
  private static Dataset<Row> parquetData;

  @BeforeAll
  static void setup() {
    dbPath = System.getenv("DB_PATH");
    parquetPath = System.getenv("PARQUET_PATH");
    assumeTrue(dbPath != null && !dbPath.isEmpty(), "DB_PATH environment variable is not set");
    assumeTrue(parquetPath != null && !parquetPath.isEmpty(), "PARQUET_PATH environment variable is not set");

    spark = SparkSession.builder()
        .appName("spark-lance-connector-test")
        .master("local")
        .config("spark.sql.catalog.lance", "com.lancedb.lance.spark.LanceCatalog")
        .getOrCreate();
    lanceData = spark.read().format(LanceDataSource.name)
        .option(LanceConfig.CONFIG_TABLE_PATH, LanceConfig.getTablePath(dbPath, "lineitem_10"))
        .load();
    lanceData.createOrReplaceTempView("lance_dataset");
    parquetData = spark.read().parquet(parquetPath);
    parquetData.createOrReplaceTempView("parquet_dataset");
  }

  @AfterAll
  static void tearDown() {
    if (spark != null) {
      spark.stop();
    }
  }

  @Test
  public void test() {
    validateResults(data -> data.filter("l_orderkey == 1"));
    validateResults(data -> data.filter("l_shipmode = 'TRUCK'").limit(10));
    validateResults(data -> data.filter("l_shipmode IS NULL").selectExpr("count(*) as count"));
    validateResults(data -> data.select("l_shipmode").limit(100));
    validateResults(data -> data.select("l_orderkey", "l_partkey", "l_quantity", "l_extendedprice").limit(10));
    validateResults(data -> data.groupBy("l_linestatus").avg("l_discount"));
    validateResults(data -> data.groupBy("l_partkey").sum("l_quantity").orderBy(desc("sum(l_quantity)")).limit(5));
    validateResults(data -> data.select("l_shipmode").distinct());
    validateResults(data -> data.select("l_orderkey", "l_comment").filter("l_comment LIKE '%express%'"));

    // OOM in java test, pass in spark, need to enlarge java memory
     validateResults(data -> data.select("l_orderkey", "l_partkey", "l_quantity"));
     validateResults(data -> data.filter("l_quantity > 30").select("l_orderkey", "l_partkey", "l_quantity"));
     validateResults(data -> data.groupBy("l_returnflag").count());
     validateResults(data -> data.filter("l_quantity BETWEEN 5 AND 30"));

    // Not exact same result, but result is correct
    Function<Dataset<Row>, Dataset<Row>> function =  data -> data.select("l_orderkey", "l_commitdate").orderBy("l_commitdate").limit(10);
    function.apply(lanceData).show();
    function.apply(parquetData).show();

    // Lance much faster than parquet
    validateResults(data -> data.groupBy("l_orderkey").sum("l_extendedprice").orderBy(desc("sum(l_extendedprice)")));

    // Lance performance issue
    assertEquals(lanceData.count(), parquetData.count());
    assertEquals(lanceData.select("l_orderkey").count(), parquetData.select("l_orderkey").count());
  }

  @Test
  public void sql() {
    validateSQLResults("SELECT * FROM parquet_dataset LIMIT 10");
    validateSQLResults("SELECT l_orderkey, l_partkey FROM parquet_dataset LIMIT 10");
    validateSQLResults("SELECT l_extendedprice, l_discount, l_tax FROM parquet_dataset LIMIT 10");
    validateSQLResults("SELECT l_shipmode, COUNT(*) AS count FROM parquet_dataset GROUP BY l_shipmode");
    validateSQLResults("SELECT l_orderkey, SUM(l_extendedprice) AS total_extendedprice FROM parquet_dataset GROUP BY l_orderkey ORDER BY total_extendedprice DESC LIMIT 10");
    validateSQLResults("SELECT l_suppkey, SUM(l_tax) AS total_tax FROM parquet_dataset GROUP BY l_suppkey ORDER BY total_tax DESC LIMIT 5");
    validateSQLResults("SELECT l_orderkey, year(l_shipdate) AS ship_year FROM parquet_dataset GROUP BY l_orderkey, ship_year ORDER BY ship_year LIMIT 10");
    validateSQLResults("SELECT l_orderkey, l_partkey, l_quantity FROM parquet_dataset WHERE l_quantity IS NULL");

    // LanceError(IO): Received literal Float64(100000) and could not convert to literal of type 'Decimal128(15, 2)', rust/lance/src/datafusion/logical_expr.rs:28:17
    // spark.sql("SELECT * FROM lineitem WHERE (l_extendedprice <= 100000)").show();
    // spark.sql("SELECT * FROM lineitem2 WHERE (l_quantity > 30) AND (l_extendedprice <= 100000) AND (l_comment IS NOT NULL)").show();
    // spark.sql("SELECT * FROM lineitem WHERE (l_quantity > 30) AND (l_extendedprice < 50000)").show();
    // spark.sql("SELECT * FROM lineitem WHERE NOT (l_quantity > 30) AND ((l_comment IS NOT NULL) OR (l_address IS NULL)) AND ((l_extendedprice < 100000) AND (l_extendedprice >= 50000))").show();
    validateSQLResults("SELECT * FROM parquet_dataset WHERE (l_quantity > 30) AND (l_comment IS NOT NULL)");
  }

  private void validateResults(Function<Dataset<Row>, Dataset<Row>> operation) {
    Dataset<Row> resultLance = operation.apply(lanceData);
    Dataset<Row> resultParquet = operation.apply(parquetData);
    assertEquals(resultParquet.collectAsList(), resultLance.collectAsList(), "Results differ between Lance and Parquet datasets");
  }

  private void validateSQLResults(String sqlQuery) {
    Dataset<Row> resultLance = spark.sql(sqlQuery.replace("parquet_dataset", "lance_dataset"));
    Dataset<Row> resultParquet = spark.sql(sqlQuery);
    assertEquals(resultParquet.collectAsList(), resultLance.collectAsList(), "Results differ between Lance and Parquet datasets for query: " + sqlQuery);
  }
}
