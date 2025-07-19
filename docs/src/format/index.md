# Lance Format Specification

The Lance format contains both a table format and a columnar file format.
When combined, we refer to it as a data format. 
Because Lance can store both structured and unstructured multimodal data, Lance typically refers to tables as "datasets".
A Lance dataset is designed to efficiently handle secondary indices, fast ingestion and modification of data, 
and a rich set of schema and data evolution features.

## Feature Flags

As the file format and dataset evolve, new feature flags are added to the format. 
There are two separate fields for checking for feature flags, 
depending on whether you are trying to read or write the table. 
Readers should check the `reader_feature_flags` to see if there are any flag it is not aware of. 
Writers should check `writer_feature_flags`. If either sees a flag they don't know, 
they should return an "unsupported" error on any read or write operation.