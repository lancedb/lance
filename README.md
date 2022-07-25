# Lance: A Columnar Data Format for Deep Learning

Lance is a *cloud-native columnar data format* designed for unstructured machine learning datasets, featuring:

* Fast columnar scan for ML dataset analysis, ML training and evaluation.
* Encodings capable of fast point queries for interactive data inspection.
* Extensible design for index and predicates pushdown.
* Strong-typed data and extensible type system.
* Schema evolution and update.
* Cloud-native optimizations on low-cost cloud storage, for instances, AWS S3 and Google GCS.
* First-class [Apache Arrow](https://arrow.apache.org/) Integration and multi-languages support.

## Why

Why do you build Lance from scratch, instead of using [Parquet](https://parquet.apache.org/)
or [ORC](https://orc.apache.org/)?

|            | Lance | Parquet & ORC | JSON & XML | Tfrecord |
|------------|-------|---------------|------------|----------|
| Analytics  | Fast  | Fast          | Slow       | Slow     |
| Training   | Fast  | Decent        | Slow       | Good     |
| Inspection | Fast  | Slow          | Fast       | Slow     |
| Tooling    | Rich  | Rich          | Rich       | Limited  |

## Presentations and Talks

* [Lance: A New Columnar Data Format](https://docs.google.com/presentation/d/1a4nAiQAkPDBtOfXFpPg7lbeDAxcNDVKgoUkw3cUs2rE/edit#slide=id.p)
  .
  [Scipy 2022, Austin, TX](https://www.scipy2022.scipy.org/posters). July, 2022.
