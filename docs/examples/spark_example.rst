Writing and Reading a Dataset Using Spark
=========================================

.. attention::
   The Spark connector is currently an experimental feature undergoing rapid iteration.

In this example, we will read a local ``iris.csv`` file and write it as a Lance dataset using Apache Spark, then demonstrate how to query the dataset.

Preparing the Environment and Raw Dataset
-----------------------------------------

Download the Spark binary package from the `official website <https://archive.apache.org/dist/spark/>`_. We recommend downloading Spark 3.5+ for Scala 2.12 (as the Spark connector currently only supports Scala 2.12).

You can directly download Spark 3.5.1 using this `link <https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz>`_.

Prepare the dataset by downloading `iris.csv <https://gist.github.com/netj/8836201>`_ to your local machine.

Create a Scala file named ``iris_to_lance_via_spark_shell.scala`` and open it.

Reading the Raw Dataset and Writing to a Lance Dataset
-------------------------------------------------------

Add necessary imports and create a Spark session:

.. code-block:: scala

   import org.apache.spark.sql.types.{StructType, StructField, DoubleType, StringType}
   import org.apache.spark.sql.{SparkSession, DataFrame}
   import com.lancedb.lance.spark.{LanceConfig, LanceDataSource}

   val spark = SparkSession.builder()
     .appName("Iris CSV to Lance Converter")
     .config("spark.sql.catalog.lance", "com.lancedb.lance.spark.LanceCatalog")
     .getOrCreate()

Specifying your input and output path:

.. code-block:: scala

   val irisPath = "/path/to/your/input/iris.csv"
   val outputPath = "/path/to/your/output/iris.lance"

Reading the ``iris.csv`` via the following snippet:

.. code-block:: scala

   val rawDF = spark.read
     .option("header", "true")
     .option("inferSchema", "true")
     .csv(irisPath)

   rawDF.printSchema()

Preparing the lance schema and write a lance dataset:

.. code-block:: scala

   val lanceSchema = new StructType()
     .add(StructField("sepal_length", DoubleType))
     .add(StructField("sepal_width", DoubleType))
     .add(StructField("petal_length", DoubleType))
     .add(StructField("petal_width", DoubleType))
     .add(StructField("species", StringType))

   val lanceDF = spark.createDataFrame(rawDF.rdd, lanceSchema)

   lanceDF.write
     .format(LanceDataSource.name)
     .option(LanceConfig.CONFIG_DATASET_URI, outputPath)
     .save()

Reading a Lance dataset
-----------------------

After writing the dataset, we can read it back and examine its properties:

.. code-block:: scala

   val lanceDF = spark.read
     .format("lance")
     .option(LanceConfig.CONFIG_DATASET_URI, outputPath)
     .load()

   println(s"The total count: ${lanceDF.count()}")
   lanceDF.printSchema()
   println("\n The top 5 data:")
   lanceDF.show(5, truncate = false)

   println("\n Species distribution statistics:")
   lanceDF.groupBy("species").count().show()

First, we open the dataset and count the total rows. Then we print the dataset schema. Finally, we analyze the species distribution statistics.

Running the Spark Application
-----------------------------

To execute the application, download these dependencies:

* lance-core JAR: Core Rust Spark binding exposing Lance features to Java (available `here <https://repo1.maven.org/maven2/com/lancedb/lance-core/0.23.0/lance-core-0.23.0.jar>`_)
* lance-spark JAR: Spark connector for reading/writing Lance format (available `here <https://repo1.maven.org/maven2/com/lancedb/lance-spark/0.23.0/lance-spark-0.23.0.jar>`_)

Place these JARs in the ``${SPARK_HOME}/jars`` directory, then run:

.. code-block:: bash

   ./bin/spark-shell --jars ./jars/lance-core-0.23.0.jar,./jars/lance-spark-0.23.0.jar -i ./iris_to_lance_via_spark_shell.scala

It should be work! Have fun!
