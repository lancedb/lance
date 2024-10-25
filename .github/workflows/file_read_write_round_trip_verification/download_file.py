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
ACCESS_KEY = os.getenv('LANCEDB_S3_READER_ACCESS_KEY')
SECRET_KEY = os.getenv('LANCEDB_S3_READER_SECRET')

# Initialize the S3 client with your credentials
s3 = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

# Define the bucket name and the file to download
bucket_name = 'lance-ci-bucket'

# List all objects in the bucket
try:
    response = s3.list_objects_v2(Bucket=bucket_name)
    if 'Contents' in response:
        for obj in response['Contents']:
            file_key = obj['Key']
            local_file_path = os.path.join(os.getcwd(), file_key)  # Save to current directory
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)  # Create directories if needed

            # Download the file
            try:
                s3.download_file(bucket_name, file_key, local_file_path)
                print(f"Downloaded {file_key} from bucket {bucket_name} to {local_file_path}")
            except Exception as e:
                print(f"Error downloading file {file_key}: {e}")
    else:
        print(f"No files found in bucket {bucket_name}")
except Exception as e:
    print(f"Error listing objects in bucket {bucket_name}: {e}")