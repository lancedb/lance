# Development

## Building the project

This java project is built with [maven](https://maven.apache.org/).

It can be built in development mode with:

```shell
mvn clean package -DskipTests
```

This builds the Rust native module in place. You will need to re-run this
whenever you change the Rust code.

If you only need to build the java code, you can build it without building the rust code:

```shell
mvn package -DskipTests
```


Also you can build in release mode with:
```shell
mvn clean package -DskipTests -Drust.release.build=true
```

The release mode will take more time to build, plz be patient.

## Running tests

To run the whole test cases:

```shell
mvn test
```


## Spark Connector local development

Download the spark 3.5.1 pre-build [package](https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz) to local and tar the package.

Edit the spark-defaults.conf file in conf director and add these configurations:
```shell
## set the LanceCatalog for lance connector
spark.sql.catalog.lance com.lancedb.lance.spark.LanceCatalog

## the lance is case sensitive so you must set this configuration for spark.
spark.sql.caseSensitive true

## storage options for access s3 storage
spark.sql.catalog.lance.access_key_id AK
spark.sql.catalog.lance.secret_access_key SK
spark.sql.catalog.lance.aws_region region
spark.sql.catalog.lance.aws_endpoint endpoint
spark.sql.catalog.lance.virtual_hosted_style_request true
```

Start the spark shell with necessary jars, and there are five jars is necessary for spark connector:
1. lance-spark-{version}.jar: the spark connector jar which is build by yourself and it will be under the path `/path_of_lance/java/spark/target/`
2. lance-core-{version}.jar: the lance native jar which will contain the rust jni code and it will be under the path `/path_of_lance/java/core/target/`
3. jar-jni-1.1.1.jar: the rust java jni binding jar which will be under the path `/root/.m2/repository/org/questdb/jar-jni/1.1.1/`
4. arrow-c-data-12.0.1.jar: the arrow jars which lance core package depends on and it will be under the path `/root/.m2/repository/org/apache/arrow/arrow-c-data/12.0.1/`
5. arrow-dataset-12.0.1.jar: the arrow jars which lance core package depends on and it will be under the path `/root/.m2/repository/org/apache/arrow/arrow-dataset/12.0.1/`

```shell
bin/spark-shell --master "local" --jars /path_of_lance/java/spark/target/lance-spark-0.21.1.jar,/path_of_lance/java/core/target/lance-core-0.21.1.jar,/root/.m2/repository/org/questdb/jar-jni/1.1.1/jar-jni-1.1.1.jar,/root/.m2/repository/org/apache/arrow/arrow-c-data/12.0.1/arrow-c-data-12.0.1.jar,/root/.m2/repository/org/apache/arrow/arrow-dataset/12.0.1/arrow-dataset-12.0.1.jar
```

In the spark shell, you can use `scala` to operator the lance dataset:
```scala
val data = spark.read.format("lance").option("path", "s3://path_of_lance/ssb_10M.lance").load();
data.createOrReplaceTempView("lance_table")
spark.sql("select * from lance_table order by lo_orderkey limit 10").show()

spark.sql("select count(*) from lance_table").show()


spark.sql("select * from lance_table order by lo_orderkey limit 10").write.format("lance").option("path", "s3://path_of_lance/out.lance").save()
```

> The spark-sql can't be used since the spark lance connector did not have integrated with catalog.