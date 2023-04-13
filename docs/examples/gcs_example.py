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

ds = lance.dataset("gs://eto-public/datasets/oxford_pet/oxford_pet.lance")
count = ds.count_rows()
print(f"There are {count} pets")

