Tokenizers
============================

Currently, Lance has built-in support for Jieba and Lindera. However, it doesn't come with its own language models.
If tokenization is needed, you can download language models by yourself.
You can specify the location where the language models are stored by setting the environment variable LANCE_LANGUAGE_MODEL_HOME.
If it's not set, the default value is

.. code-block:: bash

    ${system data directory}/lance/language_models

It also supports configuring user dictionaries,
which makes it convenient for users to expand their own dictionaries without retraining the language models.

Language Models of Jieba
------------------------

Downloading the Model
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python -m lance.download jieba

The language model is stored by default in `${LANCE_LANGUAGE_MODEL_HOME}/jieba/default`.

Using the Model
~~~~~~~~~~~~~~~

.. code-block:: python
    ds.create_scalar_index("text", "INVERTED", base_tokenizer="jieba/default")

User Dictionaries
~~~~~~~~~~~~~~~~~
Create a file named config.json in the root directory of the current model.

.. code-block:: json

    {
        "main": "dict.txt",
        "users": ["path/to/user/dict.txt"]
    }

- The "main" field is optional. If not filled, the default is "dict.txt".
- "users" is the path of the user dictionary. For the format of the user dictionary, please refer to https://github.com/messense/jieba-rs/blob/main/src/data/dict.txt.


Language Models of Lindera
--------------------------

Downloading the Model
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python -m lance.download lindera -l [ipadic|ko-dic|unidic]

Note that the language models of Lindera need to be compiled. Please install lindera-cli first. For detailed steps, please refer to https://github.com/lindera/lindera/tree/main/lindera-cli.

The language model is stored by default in ${LANCE_LANGUAGE_MODEL_HOME}/lindera/[ipadic|ko-dic|unidic]

Using the Model
~~~~~~~~~~~~~~~

.. code-block:: python

    ds.create_scalar_index("text", "INVERTED", base_tokenizer="lindera/ipadic")

User Dictionaries
~~~~~~~~~~~~~~~~~

Create a file named config.yml in the root directory of your model.
For more detailed configuration methods, see the lindera documentation at https://github.com/lindera/lindera/.

NOTE: if `LINDERA_CONFIG_PATH` is set in lance, an exception will be raised.
This is to prevent loading a single configuration file, since lance allows multiple tokenizers to be specified.

.. code-block:: yaml

    segmenter:
        mode: "normal"
        dictionary:
            # Note: in lance, the `kind` field is not supported. You need to specify the model path using the `path` field instead.
            path: /path/to/lindera/ipadic/main


Create your own language model
------------------------------

Put your language model into `LANCE_LANGUAGE_MODEL_HOME`.


