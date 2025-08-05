# Blob As Files

Unlike other data formats, large multimodal data is a first-class citizen in the Lance columnar format.
Lance provides a high-level API to store and retrieve large binary objects (blobs) in Lance datasets.

![Blob](../images/blob.png)

Lance serves large binary data using `lance.BlobFile`, which
is a file-like object that lazily reads large binary objects.

To create a Lance dataset with large blob data, you can mark a large binary column as a blob column by
adding the metadata `lance-encoding:blob` to `true`.

```python
import pyarrow as pa

schema = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("video",
            pa.large_binary(),
            metadata={"lance-encoding:blob": "true"}
        ),
    ]
)
```

To write blob data to a Lance dataset, create a PyArrow table with the blob schema and use `lance.write_dataset`:

```python
import lance

# First, download a sample video file for testing
# wget https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4
import urllib.request
urllib.request.urlretrieve(
    "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4",
    "sample_video.mp4"
)

# Then read the video file content
with open("sample_video.mp4", 'rb') as f:
    video_data = f.read()

# Create table with blob data
table = pa.table({
    "id": [1],
    "video": [video_data],
}, schema=schema)

# Write to Lance dataset
ds = lance.write_dataset(
    table,
    "./youtube.lance",
    schema=schema
)
```

To fetch blobs from a Lance dataset, you can use `lance.dataset.LanceDataset.take_blobs`.

For example, it's easy to use `BlobFile` to extract frames from a video file without
loading the entire video into memory.

```python
import av # pip install av
import lance

ds = lance.dataset("./youtube.lance")
start_time, end_time = 500, 1000
# Get blob data from the first row (id=0)
blobs = ds.take_blobs("video", ids=[0])
with av.open(blobs[0]) as container:
    stream = container.streams.video[0]
    stream.codec_context.skip_frame = "NONKEY"

    start_time = start_time / stream.time_base
    start_time = start_time.as_integer_ratio()[0]
    end_time = end_time / stream.time_base
    container.seek(start_time, stream=stream)

    for frame in container.decode(stream):
        if frame.time > end_time:
            break
        display(frame.to_image())
        clear_output(wait=True) 
```