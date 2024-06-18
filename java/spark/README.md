# Spark-Lance Connector

The Spark-Lance Connector allows Apache Spark to efficiently read tables stored in Lance format.
Lance is a modern columnar data format optimized for machine learning workflows and datasets,
supporting distributed, parallel scans, and optimizations such as column and filter pushdown to improve performance.
Additionally, Lance provides high-performance random access that is 100 times faster than Parquet without sacrificing scan performance.
By using the Spark-Lance Connector, you can leverage Spark's powerful data processing, SQL querying, and machine learning training capabilities on the AI data lake powered by Lance.

## Features

* Query Lance Tables: Seamlessly query tables stored in the Lance format using Spark.
* Distributed, Parallel Scans: Leverage Spark's distributed computing capabilities to perform parallel scans on Lance tables.
* Column and Filter Pushdown: Optimize query performance by pushing down column selections and filters to the data source.

## Installation

### Requirements

Java: Version 8 or higher
Operating System: Linux x86 or macOS

### Download jar

For Scala 2.12
```
wget https://spark-lance-artifacts.s3.amazonaws.com/lance-spark-v3-2.12-0.0.1-jar-with-dependencies.jar
```

For Scala 2.13
```
wget https://spark-lance-artifacts.s3.amazonaws.com/lance-spark-v3-2.13-0.0.1-jar-with-dependencies.jar
```

## Quick Start

Launch `spark-shell` with your selected JAR according to your Spark Scala version:
```
spark-shell --jars lance-spark-v3-2.12-0.0.3-jar-with-dependencies.jar
```

Example Usage
```java
import org.apache.spark.sql.SparkSession;

SparkSession spark = SparkSession.builder()
    .appName("spark-lance-connector-test")
    .master("local")
    .getOrCreate();

Dataset<Row> data = spark.read().format("lance")
    .option("db", "/path/to/example_db")
    .option("table", "lance_example_table")
    .load();

data.show(100)
```

More examples can be found in [Connector Test 1](/src/test/java/com/lancedb/lance/spark/SparkLanceConnectorTest.java).

## Future Works

- Add Lance Write Support
- Add LanceDB Catalog Service Support

## Notes

Spark-Lance connector uses Spark DatasourceV2 API. Please check the [Databricks presentation for DatasourceV2 API](https://www.slideshare.net/databricks/apache-spark-data-source-v2-with-wenchen-fan-and-gengliang-wang).

## Compile

For scala 2.12
```
mvn clean install
```

For scala 2.13
```
mvn clean install -Pscala-2.13
```
