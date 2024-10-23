#  Copyright (c) 2023. Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import boto3
import os

# Your AWS credentials from environment variables
ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Initialize the S3 client with your credentials
s3 = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

# Define the bucket name and the file to download
bucket_name = 'lance-ci-bucket'
file_key = 'drug-reviews.parquet'  # S3 path to the file without the 's3://' prefix
local_file_path = 'drug-reviews.parquet'  # Path to save the downloaded file locally

# Download the file
try:
    s3.download_file(bucket_name, file_key, local_file_path)
    print(f"Downloaded {file_key} from bucket {bucket_name} to {local_file_path}")
except Exception as e:
    print(f"Error downloading file: {e}")
