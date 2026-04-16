# DuckDB

Lance 数据集可以使用 [DuckDB](https://duckdb.org/) 进行 SQL 查询，
DuckDB 是一个进程内的 OLAP 关系型数据库。使用 DuckDB 意味着你可以编写复杂的 SQL 查询（Lance 可能尚未支持的），而无需将数据从 Lance 中迁出。

!!! note
    该集成通过 DuckDB 扩展实现，其源代码和最新文档（通过 `README.md`）可在
    [这里](https://github.com/lance-format/lance-duckdb)获取。
    要查看最新的示例和语法，请查阅该仓库和
    [DuckDB 扩展](https://duckdb.org/community_extensions/extensions/lance)
    文档页面。

## 安装

### Python 依赖

- 要使用 DuckDB 的 CLI，请按照[官方文档](https://duckdb.org/install/)中的步骤安装。
- 要在 Python 中运行代码，请按照以下方式安装 Lance、DuckDB 和 PyArrow。

```bash
pip install pylance duckdb pyarrow
```

### 在 DuckDB 中安装 Lance 扩展

现在我们准备好使用 DuckDB 查询 Lance 了！首先安装扩展。

=== "SQL"

    ```sql
    INSTALL lance FROM community;
    LOAD lance;
    ```

=== "Python"

    ```python
    import duckdb

    duckdb.sql(
        """
        INSTALL lance FROM community;
        LOAD lance;
        """
    )
    ```

???+ info "更新扩展"
    如果你已经在本地安装了该扩展，运行以下命令将其更新到最新版本：
    ```
    UPDATE EXTENSIONS;
    ```

## 示例

以下所有示例都使用一个包含三行数据（duck、horse、dragon）和一个带有代表性值的 `vector` 列的小数据集。在实际场景中，你的数组将是由嵌入模型生成的高维数组，Lance 数据集也会大得多。

### 将 DuckDB 表写为 Lance 数据集

使用 DuckDB 的 `COPY ... TO ...` 将查询结果物化为 Lance 数据集。

=== "SQL"

    ```sql
    COPY (
      SELECT *
      FROM (
        VALUES
          ('duck', 'quack', [0.9, 0.7, 0.1]::FLOAT[]),
          ('horse', 'neigh', [0.3, 0.1, 0.5]::FLOAT[]),
          ('dragon', 'roar', [0.5, 0.2, 0.7]::FLOAT[])
      ) AS t(animal, noise, vector)
    ) TO './lance_duck.lance' (FORMAT lance, mode 'overwrite');
    ```

=== "Python"

    ```python
    import duckdb

    duckdb.sql(
        """
        COPY (
          SELECT *
          FROM (
            VALUES
              ('duck', 'quack', [0.9, 0.7, 0.1]::FLOAT[]),
              ('horse', 'neigh', [0.3, 0.1, 0.5]::FLOAT[]),
              ('dragon', 'roar', [0.5, 0.2, 0.7]::FLOAT[])
          ) AS t(animal, noise, vector)
        ) TO './lance_duck.lance' (FORMAT lance, mode 'overwrite');
        """
    )
    ```

### 从 DuckDB 查询 Lance 数据集

Lance 数据集写入完成后，让我们在 DuckDB 中使用 SQL 查询它。

=== "SQL"

    ```sql
    SELECT *
      FROM './lance_duck.lance'
      LIMIT 5;
    ```

=== "Python"

    ```python
    import duckdb

    r1 = duckdb.sql(
        """
        SELECT *
          FROM './lance_duck.lance'
          LIMIT 5;
        """
    )
    print(r1)
    ```


返回结果：

```
┌─────────┬─────────┬─────────────────┐
│ animal  │  noise  │     vector      │
│ varchar │ varchar │     float[]     │
├─────────┼─────────┼─────────────────┤
│ duck    │ quack   │ [0.9, 0.7, 0.1] │
│ horse   │ neigh   │ [0.3, 0.1, 0.5] │
│ dragon  │ roar    │ [0.5, 0.2, 0.7] │
└─────────┴─────────┴─────────────────┘
```

???+ info "直接查询 S3 路径"
    要访问对象存储 URI（如 `s3://...`），请配置 `TYPE LANCE` 密钥。

    ```sql
    CREATE SECRET (
      TYPE LANCE,
      PROVIDER credential_chain,
      SCOPE 's3://bucket/'
    );

    SELECT *
      FROM 's3://bucket/path/to/dataset.lance'
      LIMIT 5;
    ```

### 通过 CREATE TABLE 创建 Lance 数据集（目录命名空间）

当你将一个目录 `ATTACH` 为 Lance 命名空间时，可以使用 `CREATE TABLE` 或 `CREATE TABLE AS SELECT` 创建新数据集。数据集将写入 `<namespace_root>/<table_name>.lance`。

=== "SQL"

    ```sql
    ATTACH './lance_ns' AS lance_ns (TYPE LANCE);

    CREATE TABLE lance_ns.main.duck_animals AS
      SELECT *
      FROM (
        VALUES
          ('duck', 'quack', [0.9, 0.7, 0.1]::FLOAT[]),
          ('horse', 'neigh', [0.3, 0.1, 0.5]::FLOAT[]),
          ('dragon', 'roar', [0.5, 0.2, 0.7]::FLOAT[])
      ) AS t(animal, noise, vector);
    ```

=== "Python"

    ```python
    import duckdb

    duckdb.sql(
        """
        ATTACH './lance_ns' AS lance_ns (TYPE LANCE);

        CREATE TABLE lance_ns.main.duck_animals AS
          SELECT *
          FROM (
            VALUES
              ('duck', 'quack', [0.9, 0.7, 0.1]::FLOAT[]),
              ('horse', 'neigh', [0.3, 0.1, 0.5]::FLOAT[]),
              ('dragon', 'roar', [0.5, 0.2, 0.7]::FLOAT[])
          ) AS t(animal, noise, vector);
        """
    )
    ```

然后你可以按如下方式查询命名空间：

```sql
SELECT count(*) FROM lance_ns.main.duck_animals;
```

```
┌──────────────┐
│ count_star() │
│    int64     │
├──────────────┤
│      3       │
└──────────────┘
```

### 向量搜索（Vector Search）

你可以对列执行向量搜索。这会返回 `_distance`（值越小越近，因此按升序排列以获取最近邻）。这里的示例向量与查询 "duck" 相似。

=== "SQL"

    ```sql
    SELECT animal, noise, vector, _distance
      FROM lance_vector_search(
        './lance_duck.lance',
        'vector',
        [0.8, 0.7, 0.2]::FLOAT[],
        k = 1,
        prefilter = true
      )
      ORDER BY _distance ASC;
    ```

=== "Python"

    ```python
    import duckdb

    r2 = duckdb.sql(
        """
        SELECT animal, noise, vector, _distance
          FROM lance_vector_search(
            './lance_duck.lance',
            'vector',
            [0.8, 0.7, 0.2]::FLOAT[],
            k = 1,
            prefilter = true
          )
          ORDER BY _distance ASC;
        """
    )
    print(r2)
    ```

返回结果：
```
┌─────────┬─────────┬─────────────────┐
│ animal  │  noise  │     vector      │
│ varchar │ varchar │     float[]     │
├─────────┼─────────┼─────────────────┤
│ duck    │ quack   │ [0.9, 0.7, 0.1] │
└─────────┴─────────┴─────────────────┘
```

### 全文搜索（Full-Text Search）

运行基于关键词的 BM25 搜索，如下所示。这会返回一个 `_score`，按降序排列以获取最相关的结果。

=== "SQL"

    ```sql
    SELECT animal, noise, vector, _score
      FROM lance_fts(
        './lance_duck.lance',
        'animal',
        'the brave knight faced the dragon',
        k = 1,
        prefilter = true
      )
      ORDER BY _score DESC;
    ```

=== "Python"

    ```python
    import duckdb

    r3 = duckdb.sql(
        """
        SELECT animal, noise, vector, _score
          FROM lance_fts(
            './lance_duck.lance',
            'animal',
            'the brave knight faced the dragon',
            k = 1,
            prefilter = true
          )
          ORDER BY _score DESC;
        """
    )
    print(r3)
    ```

返回结果：

```
┌─────────┬─────────┬─────────────────┐
│ animal  │  noise  │     vector      │
│ varchar │ varchar │     float[]     │
├─────────┼─────────┼─────────────────┤
│ dragon  │ roar    │ [0.5, 0.2, 0.7] │
└─────────┴─────────┴─────────────────┘
```

### 混合搜索（Hybrid Search）

混合搜索结合了向量搜索和全文搜索的分数，除了 `_distance` / `_score` 之外还返回 `_hybrid_score`。要获取最相关的结果，按降序排列。

=== "SQL"

    ```sql
    SELECT animal, noise, vector, _hybrid_score, _distance, _score
      FROM lance_hybrid_search(
        './lance_duck.lance',
        'vector',
        [0.8, 0.7, 0.2]::FLOAT[],
        'animal',
        'the duck surprised the dragon',
        k = 2,
        prefilter = false,
        alpha = 0.5,
        oversample_factor = 4
      )
      ORDER BY _hybrid_score DESC;
    ```

=== "Python"

    ```python
    import duckdb

    r4 = duckdb.sql(
        """
        SELECT animal, noise, vector, _hybrid_score, _distance, _score
          FROM lance_hybrid_search(
            './lance_duck.lance',
            'vector',
            [0.8, 0.7, 0.2]::FLOAT[],
            'animal',
            'the duck surprised the dragon',
            k = 2,
            prefilter = false,
            alpha = 0.5,
            oversample_factor = 4
          )
          ORDER BY _hybrid_score DESC;
        """
    )
    print(r4)
    ```

返回结果：
```
┌─────────┬─────────┬─────────────────┐
│ animal  │  noise  │     vector      │
│ varchar │ varchar │     float[]     │
├─────────┼─────────┼─────────────────┤
│ duck    │ quack   │ [0.9, 0.7, 0.1] │
│ dragon  │ roar    │ [0.5, 0.2, 0.7] │
└─────────┴─────────┴─────────────────┘
```

!!! warning
    DuckDB 在某些上下文中将 `column` 视为关键字。建议使用 `text_column` / `vector_column` 作为 Lance 扩展的列名。

## 源代码仓库

查阅 [lance-duckdb](https://github.com/lance-format/lance-duckdb) 项目获取最新源代码，并通过 `README.md` 了解最新的 API 文档。
以下列出了其他页面。

### 完整 SQL 参考

[sql.md](https://github.com/lance-format/lance-duckdb/blob/main/docs/sql.md)
列出了该扩展当前支持的 SQL 接口。建议参考此页面获取最新信息。

### 云存储参考

[cloud.md](https://github.com/lance-format/lance-duckdb/blob/main/docs/cloud.md) 列出了
当前支持的后端，允许你访问各种云提供商上的数据。

- S3 / S3 兼容：`s3://...`（也接受 `s3a://...` 和 `s3n://...`，标准化为 `s3://...`）
- Google Cloud Storage：`gs://...`
- Azure Blob Storage：`az://...`
- 阿里云 OSS：`oss://...`
- Hugging Face Hub (OpenDAL)：`hf://...`
