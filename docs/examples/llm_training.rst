Training LLMs using a Lance text dataset
-----------------------------------------

Using a Lance text dataset for pre-training / fine-tuning a Large Language model is straightforward and memory-efficient. 
This example follows up on the  `Creating text dataset for LLM training using Lance <https://lancedb.github.io/lance/examples/llm_dataset_creation.html>`_ example. 
Check it out if you haven't already.

In this example, we will be training an LLM using 🤗 transformers on the tokenized "wikitext_500K" lance dataset we created in the aforementioned example.

Imports and Setup
~~~~~~~~~~~~~~~~~
Let's setup our enviornment by doing all the necessary imports and defining a few basic things.

.. code-block:: python

    import numpy as np
    import lance

    import torch
    from torch.utils.data import Dataset, DataLoader, Sampler

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # We'll be training the pre-trained GPT2 model in this example
    model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Also define some hyperparameters
    lr = 3e-4
    nb_epochs = 10
    block_size = 1024
    batch_size = 8
    device = 'cuda:0'
    dataset_path = 'wikitext_500K.lance'

Now that the basic setup is out of the way, let's define our custom Dataset and a Sampler for streaming the tokens from our Lance dataset.

Data-loading Setup
~~~~~~~~~~~~~~~~~~
We start by defining a utility function that will help us load any number of tokens from our lance dataset in a 'chunk'.

.. code-block:: python

    def from_indices(dataset, indices):
        """Load the elements on given indices from the dataset"""
        chunk = dataset.take(indices).to_pylist()
        chunk = list(map(lambda x: x['input_ids'], chunk))
        return chunk

Now let's define our custom dataset and sampler for loading the tokens.

.. code-block:: python

    class LanceDataset(Dataset):
        def __init__(
            self,
            dataset_path,
            block_size,
        ):
            # Load the lance dataset from the saved path
            self.ds = lance.dataset(dataset_path)
            self.block_size = block_size

            # Doing this so the sampler never asks for an index at the end of text
            self.length = self.ds.count_rows() - block_size

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            """
            Generate a window of indices starting from the current idx to idx+block_size
            and return the tokens at those indices
            """
            window = np.arange(idx, idx + self.block_size)
            sample = from_indices(self.ds, window)

            return {"input_ids": torch.tensor(sample), "labels": torch.tensor(sample)}

When given a random index by the sampler, the dataset will load the next :meth:`block_size` number of tokens starting from current index.
This would in-essence form a sample as the loaded tokens would be causal.

However we also need to make sure that the tokens we get from the dataset aren't overlapping. Let's understand this from an example:

Let's say, for some arbitrary block size, during the training loop the dataset return the following tokens:

`"Vienna is the capital of Austria"` at index = 12 for sample #1, and,

`"is the capital of Austria and"` at index = 13 for sample #2, and so on

The problem here is that if we allow the dataloader to fetch the 'samples' for any arbitrary number of indices, they may overlap (as we see above).
This is not good for the model as it may start to overfit after seeing sufficient overlapping tokens.

To solve this problem, we define a custom Sampler that only returns the indices that are 'block_size' apart from each other, ensuring that we don't see any overlapping samples.

.. code-block:: python

    class LanceSampler(Sampler):
        r"""Samples tokens randomly but `block_size` indices apart.

        Args:
            data_source (Dataset): dataset to sample from
            block_size (int): minimum index distance between each random sample
        """

        def __init__(self, data_source, block_size=512):
            self.data_source = data_source
            self.num_samples = len(self.data_source)
            self.available_indices = list(range(0, self.num_samples, block_size))
            np.random.shuffle(self.available_indices)

        def __iter__(self):
            yield from self.available_indices

        def __len__(self) -> int:
            return len(self.available_indices)

Now when we fetch the tokens from our dataset with sampler being the :meth:`LanceSampler`, all samples in all 
the batches that our model sees during the training are guaranteed to be non-overlapping.

This is done by generating a list of indices starting from 0 to the end of the dataset (which if you remember is lance dataset length - block size) with each index 'block_size' apart from the other.
We then shuffle this list and yield indices from it.

And that's basically it for the Dataloading! Now all we are left is to train the model!

Model Training
~~~~~~~~~~~~~~
Now you train the model just like you would with any other dataset!

.. code-block:: python

    # Define the dataset, sampler and dataloader
    dataset = LanceDataset(dataset_path, block_size)
    sampler = LanceSampler(dataset, block_size)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True
    )

    # Define the optimizer, training loop and train the model!
    model = model.to(device)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(nb_epochs):
        print(f"========= Epoch: {epoch+1} / {nb_epochs} =========")
        epoch_loss = []
        prog_bar = tqdm(dataloader, total=len(dataloader))
        for batch in prog_bar:
            optimizer.zero_grad(set_to_none=True)

            # Put both input_ids and labels to the device
            for k, v in batch.items():
                batch[k] = v.to(device)

            # Perform one forward pass and get the loss
            outputs = model(**batch)
            loss = outputs.loss

            # Perform backward pass
            loss.backward()
            optimizer.step()

            prog_bar.set_description(f"loss: {loss.item():.4f}")

            epoch_loss.append(loss.item())

        # Calculate training perplexity for this epoch
        try:
            perplexity = np.exp(np.mean(epoch_loss))
        except OverflowError:
            perplexity = float("-inf")

        print(f"train_perplexity: {perplexity}")


One tip: If your lance dataset is huge (like the wikitext_500K is), and you want to debug the model to look out for errors, you may want to wrap the dataloader in an :meth:`iter()` function and only run it for a couple batches.

And that's basically it! 

The best part about using Lance, the custom Dataset and Sampler is that you get a whooping **95%** average GPU utilisation and minimal CPU overhead thanks to the lightning fast random access that Lance provides 🚀