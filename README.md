# Lance: A Columnar Data Format for Deep Learning

![CI](https://github.com/eto-ai/lance/actions/workflows/cpp.yml/badge.svg)

Lance is a *cloud-native columnar data format* designed for unstructured machine learning datasets, featuring:

* Fast columnar scan for ML dataset analysis, ML training and evaluation.
* Encodings that are capable of fast point queries for interactive data exploration.
* Extensible design for index and predicates pushdown.
* Self-describable, nested and strong-typed data with an extensible type system.
* Schema evolution and update (TODO).
* Cloud-native optimizations on low-cost cloud storage, i.e., AWS S3, Google GCS or Azure Blob Storage.
* First-class [Apache Arrow](https://arrow.apache.org/) integration and multi-languages support.

## Why

Why do you build Lance from scratch, instead of using [Parquet](https://parquet.apache.org/)
, [ORC](https://orc.apache.org/) or [Tfrecord](https://www.tensorflow.org/tutorials/load_data/tfrecord).

We envision that a typical Machine Learning cycle involves the following steps:

```mermaid
graph LR
    A[Collection] --> B[Exploration];
    B --> C[Analytics];
    C --> D[Feature Engineer];
    D --> E[Training];
    E -- Evaluation --> C;
```


|                     | Lance | Parquet & ORC | JSON & XML | Tfrecord | Database | Warehouse |
|---------------------|-------|---------------|------------|----------|----------|-----------|
| Analytics           | Fast  | Fast          | Slow       | Slow     | Decent   | Fast      |
| Feature Engineering | Fast  | Fast          | Decent     | Slow     | Decent   | Good      |
| Training            | Fast  | Decent        | Slow       | Fast     | N/A      | N/A       |
| Exploration         | Fast  | Slow          | Fast       | Slow     | Fast     | Decent    |
| Tooling             | Rich  | Rich          | Rich       | Limited  | Good     | Rich      |

## Presentations and Talks

* [Lance: A New Columnar Data Format](https://docs.google.com/presentation/d/1a4nAiQAkPDBtOfXFpPg7lbeDAxcNDVKgoUkw3cUs2rE/edit#slide=id.p)
  .
  [Scipy 2022, Austin, TX](https://www.scipy2022.scipy.org/posters). July, 2022.
