# Object Store Configuration

Lance supports object stores such as AWS S3 (and compatible stores), Azure Blob Store,
and Google Cloud Storage. Which object store to use is determined by the URI scheme of
the dataset path. For example, `s3://bucket/path` will use S3, `az://bucket/path`
will use Azure, and `gs://bucket/path` will use GCS.

These object stores take additional configuration objects. There are two ways to
specify these configurations: by setting environment variables or by passing them
to the `storage_options` parameter of `lance.dataset` and
`lance.write_dataset`. So for example, to globally set a higher timeout,
you would run in your shell:

```bash
export TIMEOUT=60s
```

If you only want to set the timeout for a single dataset, you can pass it as a
storage option:

```python
import lance
ds = lance.dataset("s3://path", storage_options={"timeout": "60s"})
```

## General Configuration

These options apply to all object stores.

| Key                          | Description                                                                                                                                                                                                                                                                                             |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `allow_http`                 | Allow non-TLS, i.e. non-HTTPS connections. Default, `False`.                                                                                                                                                                                                                                            |
| `download_retry_count`       | Number of times to retry a download. Default, `3`. This limit is applied when the HTTP request succeeds but the response is not fully downloaded, typically due to a violation of `request_timeout`.                                                                                                    |
| `allow_invalid_certificates` | Skip certificate validation on https connections. Default, `False`. Warning: This is insecure and should only be used for testing.                                                                                                                                                                      |
| `connect_timeout`            | Timeout for only the connect phase of a Client. Default, `5s`.                                                                                                                                                                                                                                          |
| `request_timeout`            | Timeout for the entire request, from connection until the response body has finished. Default, `30s`.                                                                                                                                                                                                   |
| `user_agent`                 | User agent string to use in requests.                                                                                                                                                                                                                                                                   |
| `proxy_url`                  | URL of a proxy server to use for requests. Default, `None`.                                                                                                                                                                                                                                             |
| `proxy_ca_certificate`       | PEM-formatted CA certificate for proxy connections                                                                                                                                                                                                                                                      |
| `proxy_excludes`             | List of hosts that bypass proxy. This is a comma separated list of domains and IP masks. Any subdomain of the provided domain will be bypassed. For example, `example.com, 192.168.1.0/24` would bypass `https://api.example.com`, `https://www.example.com`, and any IP in the range `192.168.1.0/24`. |
| `client_max_retries`         | Number of times for a s3 client to retry the request. Default, `10`.                                                                                                                                                                                                                                    |
| `client_retry_timeout`       | Timeout for a s3 client to retry the request in seconds. Default, `180`.                                                                                                                                                                                                                                |

## S3 Configuration

S3 (and S3-compatible stores) have additional configuration options that configure
authorization and S3-specific features (such as server-side encryption).

AWS credentials can be set in the environment variables `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN`. Alternatively, they can be
passed as parameters to the `storage_options` parameter:

```python
import lance
ds = lance.dataset(
    "s3://bucket/path",
    storage_options={
        "access_key_id": "my-access-key",
        "secret_access_key": "my-secret-key",
        "session_token": "my-session-token",
    }
)
```

If you are using AWS SSO, you can specify the `AWS_PROFILE` environment variable.
It cannot be specified in the `storage_options` parameter.

The following keys can be used as both environment variables or keys in the
`storage_options` parameter:

| Key                                                                 | Description                                                                                                                                      |
|---------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `aws_region` / `region`                                             | The AWS region the bucket is in. This can be automatically detected when using AWS S3, but must be specified for S3-compatible stores.           |
| `aws_access_key_id` / `access_key_id`                               | The AWS access key ID to use.                                                                                                                    |
| `aws_secret_access_key` / `secret_access_key`                       | The AWS secret access key to use.                                                                                                                |
| `aws_session_token` / `session_token`                               | The AWS session token to use.                                                                                                                    |
| `aws_endpoint` / `endpoint`                                         | The endpoint to use for S3-compatible stores.                                                                                                    |
| `aws_virtual_hosted_style_request` / `virtual_hosted_style_request` | Whether to use virtual hosted-style requests, where bucket name is part of the endpoint. Meant to be used with `aws_endpoint`. Default, `False`. |
| `aws_s3_express` / `s3_express`                                     | Whether to use S3 Express One Zone endpoints. Default, `False`. See more details below.                                                          |
| `aws_server_side_encryption`                                        | The server-side encryption algorithm to use. Must be one of `"AES256"`, `"aws:kms"`, or `"aws:kms:dsse"`. Default, `None`.                       |
| `aws_sse_kms_key_id`                                                | The KMS key ID to use for server-side encryption. If set, `aws_server_side_encryption` must be `"aws:kms"` or `"aws:kms:dsse"`.                  |
| `aws_sse_bucket_key_enabled`                                        | Whether to use bucket keys for server-side encryption.                                                                                           |

### S3-compatible stores

Lance can also connect to S3-compatible stores, such as MinIO. To do so, you must
specify both region and endpoint:

```python
import lance
ds = lance.dataset(
    "s3://bucket/path",
    storage_options={
        "region": "us-east-1",
        "endpoint": "http://minio:9000",
    }
)
```

This can also be done with the `AWS_ENDPOINT` and `AWS_DEFAULT_REGION` environment variables.

### S3 Express (Directory Bucket)

Lance supports [S3 Express One Zone](https://aws.amazon.com/s3/storage-classes/express-one-zone/) buckets,
a.k.a. S3 directory buckets.
S3 Express buckets only support connecting from an EC2 instance within the same
region.
By default, Lance automatically recognize the `--x-s3` suffix of an express bucket,
there is no special configuration needed.

In case of an access point or private link that hides the bucket name,
you can configure express bucket access explicitly through storage option `s3_express`.

```python
import lance
ds = lance.dataset(
    "s3://my-bucket--use1-az4--x-s3/path/imagenet.lance",
    storage_options={
        "region": "us-east-1",
        "s3_express": "true",
    }
)
```

## Google Cloud Storage Configuration

GCS credentials are configured by setting the `GOOGLE_SERVICE_ACCOUNT` environment
variable to the path of a JSON file containing the service account credentials.
Alternatively, you can pass the path to the JSON file in the `storage_options`

```python
import lance
ds = lance.dataset(
    "gs://my-bucket/my-dataset",
    storage_options={
        "service_account": "path/to/service-account.json",
    }
)
```

!!! note

    By default, GCS uses HTTP/1 for communication, as opposed to HTTP/2. This improves
    maximum throughput significantly. However, if you wish to use HTTP/2 for some reason,
    you can set the environment variable `HTTP1_ONLY` to `false`.

The following keys can be used as both environment variables or keys in the
`storage_options` parameter:

| Key | Description |
|-----|-------------|
| `google_service_account` / `service_account` | Path to the service account JSON file. |
| `google_service_account_key` / `service_account_key` | The serialized service account key. |
| `google_application_credentials` / `application_credentials` | Path to the application credentials. |

## Azure Blob Storage Configuration

Azure Blob Storage credentials can be configured by setting the `AZURE_STORAGE_ACCOUNT_NAME`
and `AZURE_STORAGE_ACCOUNT_KEY` environment variables. Alternatively, you can pass
the account name and key in the `storage_options` parameter:

```python
import lance
ds = lance.dataset(
    "az://my-container/my-dataset",
    storage_options={
        "account_name": "some-account",
        "account_key": "some-key",
    }
)
```

These keys can be used as both environment variables or keys in the `storage_options` parameter:

| Key | Description |
|-----|-------------|
| `azure_storage_account_name` / `account_name` | The name of the azure storage account. |
| `azure_storage_account_key` / `account_key` | The serialized service account key. |
| `azure_client_id` / `client_id` | Service principal client id for authorizing requests. |
| `azure_client_secret` / `client_secret` | Service principal client secret for authorizing requests. |
| `azure_tenant_id` / `tenant_id` | Tenant id used in oauth flows. |
| `azure_storage_sas_key` / `azure_storage_sas_token` / `sas_key` / `sas_token` | Shared access signature. The signature is expected to be percent-encoded, much like they are provided in the azure storage explorer or azure portal. |
| `azure_storage_token` / `bearer_token` / `token` | Bearer token. |
| `azure_storage_use_emulator` / `object_store_use_emulator` / `use_emulator` | Use object store with azurite storage emulator. |
| `azure_endpoint` / `endpoint` | Override the endpoint used to communicate with blob storage. |
| `azure_use_fabric_endpoint` / `use_fabric_endpoint` | Use object store with url scheme account.dfs.fabric.microsoft.com. |
| `azure_msi_endpoint` / `azure_identity_endpoint` / `identity_endpoint` / `msi_endpoint` | Endpoint to request a imds managed identity token. |
| `azure_object_id` / `object_id` | Object id for use with managed identity authentication. |
| `azure_msi_resource_id` / `msi_resource_id` | Msi resource id for use with managed identity authentication. |
| `azure_federated_token_file` / `federated_token_file` | File containing token for Azure AD workload identity federation. |
| `azure_use_azure_cli` / `use_azure_cli` | Use azure cli for acquiring access token. |
| `azure_disable_tagging` / `disable_tagging` | Disables tagging objects. This can be desirable if not supported by the backing store. | 