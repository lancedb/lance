Creating Multi-Modal datasets using Lance
-----------------------------------------
Thanks to Lance file format's ability to store data of different modalities, one of the important use-cases that Lance shines in is storing Multi-modal datasets.
In this brief example we will be going over how you can take a Multi-modal dataset and store it in Lance file format. 

The dataset of choice here is `Flickr8k dataset <https://github.com/goodwillyoga/Flickr8k_dataset>`_. Flickr8k is a benchmark collection for sentence-based image description and search, consisting of 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events. 
The images were chosen from six different Flickr groups, and tend not to contain any well-known people or locations, but were manually selected to depict a variety of scenes and situations.

We will be creating an Image-caption pair dataset for Multi-modal model training by using the above mentioned Flickr8k dataset, saving it in form of a Lance dataset with image file names, all captions for every image (order preserved) and the image itself (in binary format).

Imports and Setup
~~~~~~~~~~~~~~~~~
We assume that you downloaded the dataset, more specifically the "Flickr8k.token.txt" file and the "Flicker8k_Dataset/" folder and both are present in the current directory.
These can be downloaded from `here <https://github.com/goodwillyoga/Flickr8k_dataset?tab=readme-ov-file>`_ (download both the dataset and text zip files).

We also assume you have pyarrow and pylance installed as well as opencv (for reading in images) and tqdm (for pretty progress bars).

Now let's start with imports and defining the caption file and image dataset folder.

.. code-block:: python

    import os
    import cv2
    import random

    import lance
    import pyarrow as pa

    import matplotlib.pyplot as plt

    from tqdm.auto import tqdm

    captions = "Flickr8k.token.txt"
    image_folder = "Flicker8k_Dataset/"


Loading and Processing
~~~~~~~~~~~~~~~~~~~~~~

In flickr8k dataset, each image has multiple corresponding captions that are ordered. 
We are going to put all these captions in a list corresponding to each image with their position in the list representing the order in which they originally appear.
Let's load the annotations (the image path and corresponding captions) in a list with each element of the list being a tuple consisting of image name, caption number and caption itself.

.. code-block:: python

    with open(captions, "r") as fl:
        annotations = fl.readlines()

    # Converts the annotations where each element of this list is a tuple consisting of image file name, caption number and caption itself
    annotations = list(map(lambda x: tuple([*x.split('\t')[0].split('#'), x.split('\t')[1]]), annotations))

Now, for all captions of the same image, we will put them in a list sorted by their ordering.

.. code-block:: python

    captions = []
    image_ids = set(ann[0] for ann in annotations)
    for img_id in tqdm(image_ids):
        current_img_captions = []
        for ann_img_id, num, caption in annotations:
            if img_id == ann_img_id:
                current_img_captions.append((num, caption))
                
        # Sort by the annotation number
        current_img_captions.sort(key=lambda x: x[0])
        captions.append((img_id, tuple([x[1] for x in current_img_captions])))

Converting to a Lance Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that our captions list is in a proper format, we will write a :meth:`process()` function that will take the said captions as argument and yield a Pyarrow record batch consisting of the :meth:`image_id`, :meth:`image` and :meth:`captions`.
The image in this record batch will be in binary format and all the captions for an image will be in a list with their ordering preserved.

.. code-block:: python

    def process(captions):
        for img_id, img_captions in tqdm(captions):
            try:
                with open(os.path.join(image_folder, img_id), 'rb') as im:
                    binary_im = im.read()
                    
            except FileNotFoundError:
                print(f"img_id '{img_id}' not found in the folder, skipping.")
                continue
            
            img_id = pa.array([img_id], type=pa.string())
            img = pa.array([binary_im], type=pa.binary())
            capt = pa.array([img_captions], pa.list_(pa.string(), -1))
            
            yield pa.RecordBatch.from_arrays(
                [img_id, img, capt], 
                ["image_id", "image", "captions"]
            )

Let's also define the same schema to tell Pyarrow the type of data it should be expecting in the table.

.. code-block:: python

    schema = pa.schema([
        pa.field("image_id", pa.string()),
        pa.field("image", pa.binary()),
        pa.field("captions", pa.list_(pa.string(), -1)),
    ])

We are including the :meth:`image_id` (which is the original image name) so it can be easier to reference and debug in the future.

Finally, we define a reader to iteratively read those record batches and then write them to a lance dataset on the disk.

.. code-block:: python
    
    reader = pa.RecordBatchReader.from_batches(schema, process(captions))
    lance.write_dataset(reader, "flickr8k.lance", schema)

And that's basically it! If you want to execute this in a notebook form, you can check out this example in our deeplearning-recipes repository `here <https://github.com/lancedb/lance-deeplearning-recipes/tree/main/examples/flickr8k-dataset>`_.

For more Deep learning related examples using Lance dataset, be sure to check out the `lance-deeplearning-recipes <https://github.com/lancedb/lance-deeplearning-recipes>`_ repository!