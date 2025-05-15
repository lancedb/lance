# Java bindings and SDK for Lance Data Format

> :warning: **Under heavy development**

<div align="center">
<p align="center">

<img width="257" alt="Lance Logo" src="https://user-images.githubusercontent.com/917119/199353423-d3e202f7-0269-411d-8ff2-e747e419e492.png">

Lance is a new columnar data format for data science and machine learning
</p></div>

Why you should use Lance
1. It is an order of magnitude faster than Parquet for point queries and nested data structures common to DS/ML
2. It comes with a fast vector index that delivers sub-millisecond nearest neighbor search performance
3. It is automatically versioned and supports lineage and time-travel for full reproducibility
4. It is integrated with duckdb/pandas/polars already. Easily convert from/to Parquet in 2 lines of code

## Quick start

Introduce the Lance SDK Java Maven dependency(It is recommended to choose the latest version.):

```shell
<dependency>
    <groupId>com.lancedb</groupId>
    <artifactId>lance-core</artifactId>
    <version>0.18.0</version>
</dependency>
```

### Basic I/O

* create empty dataset

```java
void createDataset() throws IOException, URISyntaxException {
    String datasetPath = tempDir.resolve("write_stream").toString();
    Schema schema =
            new Schema(
                    Arrays.asList(
                            Field.nullable("id", new ArrowType.Int(32, true)),
                            Field.nullable("name", new ArrowType.Utf8())),
                    null);
    try (BufferAllocator allocator = new RootAllocator();) {
        Dataset.create(allocator, datasetPath, schema, new WriteParams.Builder().build());
        try (Dataset dataset = Dataset.create(allocator, datasetPath, schema, new WriteParams.Builder().build());) {
            dataset.version();
            dataset.latestVersion();
        }
    }
}
```

* create and write a Lance dataset

```java
void createAndWriteDataset() throws IOException, URISyntaxException {
    Path path = "";     // the original source path
    String datasetPath = "";    // specify a path point to a dataset
    try (BufferAllocator allocator = new RootAllocator();
        ArrowFileReader reader =
            new ArrowFileReader(
                new SeekableReadChannel(
                    new ByteArrayReadableSeekableByteChannel(Files.readAllBytes(path))), allocator);
        ArrowArrayStream arrowStream = ArrowArrayStream.allocateNew(allocator)) {
        Data.exportArrayStream(allocator, reader, arrowStream);
        try (Dataset dataset =
                     Dataset.create(
                             allocator,
                             arrowStream,
                             datasetPath,
                             new WriteParams.Builder()
                                     .withMaxRowsPerFile(10)
                                     .withMaxRowsPerGroup(20)
                                     .withMode(WriteParams.WriteMode.CREATE)
                                     .withStorageOptions(new HashMap<>())
                                     .build())) {
            // access dataset
        }
    }
}
```
* read dataset

```java
void readDataset() {
    String datasetPath = ""; // specify a path point to a dataset
    try (BufferAllocator allocator = new RootAllocator()) {
        try (Dataset dataset = Dataset.open(datasetPath, allocator)) {
            dataset.countRows();
            dataset.getSchema();
            dataset.version();
            dataset.latestVersion();
            // access more information
        }
    }
}
```

* drop dataset

```java
void dropDataset() {
    String datasetPath = tempDir.resolve("drop_stream").toString();
    Dataset.drop(datasetPath, new HashMap<>());
}
```

### Random Access

```java
void randomAccess() {
    String datasetPath = ""; // specify a path point to a dataset
    try (BufferAllocator allocator = new RootAllocator()) {
        try (Dataset dataset = Dataset.open(datasetPath, allocator)) {
            List<Long> indices = Arrays.asList(1L, 4L);
            List<String> columns = Arrays.asList("id", "name");
            try (ArrowReader reader = dataset.take(indices, columns)) {
                while (reader.loadNextBatch()) {
                    VectorSchemaRoot result = reader.getVectorSchemaRoot();
                    result.getRowCount();

                    for (int i = 0; i < indices.size(); i++) {
                        result.getVector("id").getObject(i);
                        result.getVector("name").getObject(i);
                    }
                }
            }
        }
    }
}
```

### Schema evolution

* add columns

```java
feat(java): support adding columns by schema and field definitionsvoid addColumnsByExpressions() {
    String datasetPath = ""; // specify a path point to a dataset
    try (BufferAllocator allocator = new RootAllocator()) {
        try (Dataset dataset = Dataset.open(datasetPath, allocator)) {
            SqlExpressions sqlExpressions = new SqlExpressions.Builder().withExpression("double_id", "id * 2").build();
            dataset.addColumns(sqlExpressions, Optional.empty());
        }
    }
}

void addColumnsBySchema() {
  String datasetPath = ""; // specify a path point to a dataset
  try (BufferAllocator allocator = new RootAllocator()) {
    try (Dataset dataset = Dataset.open(datasetPath, allocator)) {
      SqlExpressions sqlExpressions = new SqlExpressions.Builder().withExpression("double_id", "id * 2").build();
      dataset.addColumns(new Schema(
          Arrays.asList(
              Field.nullable("id", new ArrowType.Int(32, true)),
              Field.nullable("name", new ArrowType.Utf8()),
              Field.nullable("age", new ArrowType.Int(32, true)))), Optional.empty());
    }
  }
}
```

* alter columns

```java
void alterColumns() {
    String datasetPath = ""; // specify a path point to a dataset
    try (BufferAllocator allocator = new RootAllocator()) {
        try (Dataset dataset = Dataset.open(datasetPath, allocator)) {
            ColumnAlteration nameColumnAlteration =
                    new ColumnAlteration.Builder("name")
                            .rename("new_name")
                            .nullable(true)
                            .castTo(new ArrowType.Utf8())
                            .build();

            dataset.alterColumns(Collections.singletonList(nameColumnAlteration));
        }
    }
}
```

* drop columns

```java
void dropColumns() {
    String datasetPath = ""; // specify a path point to a dataset
    try (BufferAllocator allocator = new RootAllocator()) {
        try (Dataset dataset = Dataset.open(datasetPath, allocator)) {
            dataset.dropColumns(Collections.singletonList("name"));
        }
    }
}
```

## Integrations

This section introduces the ecosystem integration with Lance format.
With the integration, users are able to access lance dataset with other technology or tools.

### Spark connector

The [spark](https://github.com/lancedb/lance/tree/main/java/spark) module is a standard maven module.
It is the implementation of spark-lance connector that allows Apache Spark to efficiently access datasets stored in Lance format.
More details please see the [README](https://github.com/lancedb/lance/blob/main/java/spark/README.md) file.

## Contributing

From the codebase dimension, the lance project is a multiple-lang project. All Java-related code is located in the `java` directory.
And the whole `java` dir is a standard maven project(named `lance-parent`) can be imported into any IDEs support java project.

Overall, it contains two Maven sub-modules:

* lance-core: the core module of Lance Java binding, including `lance-jni`.
* lance-spark: the spark connector module.

To build the project, you can run the following command:

```shell
mvn clean package
```

if you only want to build rust code(`lance-jni`), you can run the following command:

```shell
cargo build
```

The java module uses `spotless` maven plugin to format the code and check the license header. 
And it is applied in the `validate` phase automatically.

### Environment(IDE) setup

Firstly, clone the repository into your local machine:

```shell
git clone https://github.com/lancedb/lance.git
```

Then, import the `java` directory into your favorite IDEs, such as IntelliJ IDEA, Eclipse, etc.

Due to the java module depends on the features provided by rust module. So, you also need to make sure you have installed rust in your local.

To install rust, please refer to the [official documentation](https://www.rust-lang.org/tools/install).

And you also need to install the rust plugin for your IDE.

Then, you can build the whole java module:

```shell
mvn clean package
```

Running these commands, it builds the rust jni binding codes automatically.
