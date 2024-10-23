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
