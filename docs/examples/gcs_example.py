#
# Lance example loading a dataset from Google Cloud Storage
#
# You need to set one of the following environment variables in order to authenticate with GS
#   - GOOGLE_SERVICE_ACCOUNT: location of service account file
#   - GOOGLE_SERVICE_ACCOUNT_KEY: JSON serialized service account key
#
# Follow this doc in order to create an service key: https://cloud.google.com/iam/docs/keys-create-delete
#

import lance
import pandas as pd

ds = lance.dataset("gs://eto-public/datasets/oxford_pet/oxford_pet.lance")
count = ds.count_rows()
print(f"There are {count} pets")

# You can also write to GCS

uri = "gs://eto-public/datasets/oxford_pet/example.lance"
lance.write_dataset(
    pd.DataFrame({"a": pd.array([10], dtype="Int32")}), uri, mode="create"
)
assert lance.dataset(uri).version == 1

lance.write_dataset(
    pd.DataFrame({"a": pd.array([5], dtype="Int32")}), uri, mode="append"
)
assert lance.dataset(uri).version == 2
