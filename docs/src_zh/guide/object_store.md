# 对象存储配置

Lance 支持 AWS S3（及兼容存储）、Azure Blob Storage 和 Google Cloud Storage 等对象存储。使用哪个对象存储由数据集路径的 URI 方案决定。例如，`s3://bucket/path` 将使用 S3，`az://bucket/path` 将使用 Azure，`gs://bucket/path` 将使用 GCS。

这些对象存储需要额外的配置对象。有两种方式指定这些配置：设置环境变量或将它们传递给 `lance.dataset` 和 `lance.write_dataset` 的 `storage_options` 参数。例如，要全局设置更长的超时时间，你可以在 shell 中运行：

```bash
export TIMEOUT=60s
```

如果你只想为单个数据集设置超时，可以将其作为存储选项传递：

```python
import lance
ds = lance.dataset("s3://path", storage_options={"timeout": "60s"})
```

## 通用配置

这些选项适用于所有对象存储。

| 键                           | 描述                                                                                                                                                                                                                                                                                                    |
|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `allow_http`                 | 允许非 TLS（即非 HTTPS）连接。默认为 `False`。                                                                                                                                                                                                                                                          |
| `download_retry_count`       | 下载重试次数。默认为 `3`。此限制在 HTTP 请求成功但响应未完全下载时应用，通常是由于违反了 `request_timeout`。                                                                                                                                                                                              |
| `allow_invalid_certificates` | 跳过 HTTPS 连接的证书验证。默认为 `False`。警告：这是不安全的，仅应用于测试。                                                                                                                                                                                                                            |
| `connect_timeout`            | 仅连接阶段的超时时间。默认为 `5s`。                                                                                                                                                                                                                                                                     |
| `request_timeout`            | 整个请求的超时时间，从连接到响应体完成。默认为 `30s`。                                                                                                                                                                                                                                                   |
| `user_agent`                 | 请求中使用的用户代理字符串。                                                                                                                                                                                                                                                                             |
| `proxy_url`                  | 用于请求的代理服务器 URL。默认为 `None`。                                                                                                                                                                                                                                                                |
| `proxy_ca_certificate`       | 代理连接的 PEM 格式 CA 证书。                                                                                                                                                                                                                                                                            |
| `proxy_excludes`             | 绕过代理的主机列表。这是一个逗号分隔的域名和 IP 掩码列表。所提供域名的任何子域名都将被绕过。例如，`example.com, 192.168.1.0/24` 将绕过 `https://api.example.com`、`https://www.example.com` 以及 `192.168.1.0/24` 范围内的任何 IP。 |
| `client_max_retries`         | 对象存储客户端重试请求的次数。默认为 `3`。                                                                                                                                                                                                                                                               |
| `client_retry_timeout`       | 对象存储客户端重试请求的超时时间（秒）。默认为 `180`。                                                                                                                                                                                                                                                   |

## S3 配置

S3（及 S3 兼容存储）有额外的配置选项，用于配置授权和 S3 特定功能（如服务器端加密）。

AWS 凭证可以通过环境变量 `AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY` 和 `AWS_SESSION_TOKEN` 设置。或者，它们可以作为参数传递给 `storage_options`：

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

如果你使用 AWS SSO，可以指定 `AWS_PROFILE` 环境变量。它不能在 `storage_options` 参数中指定。

以下键可以同时用作环境变量或 `storage_options` 参数中的键：

| 键                                                                    | 描述                                                                                                                                             |
|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| `aws_region` / `region`                                              | 存储桶所在的 AWS 区域。使用 AWS S3 时可以自动检测，但 S3 兼容存储必须指定。                                                                      |
| `aws_access_key_id` / `access_key_id`                                | 要使用的 AWS 访问密钥 ID。                                                                                                                       |
| `aws_secret_access_key` / `secret_access_key`                        | 要使用的 AWS 秘密访问密钥。                                                                                                                       |
| `aws_session_token` / `session_token`                                | 要使用的 AWS 会话令牌。                                                                                                                           |
| `aws_endpoint` / `endpoint`                                          | S3 兼容存储使用的端点。                                                                                                                           |
| `aws_virtual_hosted_style_request` / `virtual_hosted_style_request`  | 是否使用虚拟主机风格请求，其中存储桶名称是端点的一部分。旨在与 `aws_endpoint` 一起使用。默认为 `False`。                                            |
| `aws_s3_express` / `s3_express`                                      | 是否使用 S3 Express One Zone 端点。默认为 `False`。更多详情见下文。                                                                               |
| `aws_server_side_encryption`                                         | 要使用的服务器端加密算法。必须是 `"AES256"`、`"aws:kms"` 或 `"aws:kms:dsse"` 之一。默认为 `None`。                                                |
| `aws_sse_kms_key_id`                                                 | 用于服务器端加密的 KMS 密钥 ID。如果设置，`aws_server_side_encryption` 必须为 `"aws:kms"` 或 `"aws:kms:dsse"`。                                    |
| `aws_sse_bucket_key_enabled`                                         | 是否为服务器端加密使用存储桶密钥。                                                                                                                 |

### S3 兼容存储

Lance 也可以连接到 S3 兼容存储，如 MinIO。为此，你必须同时指定区域和端点：

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

这也可以通过 `AWS_ENDPOINT` 和 `AWS_DEFAULT_REGION` 环境变量来完成。

### S3 Express（目录存储桶）

Lance 支持 [S3 Express One Zone](https://aws.amazon.com/s3/storage-classes/express-one-zone/) 存储桶，即 S3 目录存储桶。S3 Express 存储桶仅支持从同一区域内的 EC2 实例连接。默认情况下，Lance 自动识别 express 存储桶的 `--x-s3` 后缀，无需特殊配置。

如果访问点或私有链接隐藏了存储桶名称，你可以通过存储选项 `s3_express` 显式配置 express 存储桶访问。

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

## Google Cloud Storage 配置

GCS 凭证通过将 `GOOGLE_SERVICE_ACCOUNT` 环境变量设置为包含服务账户凭证的 JSON 文件路径来配置。或者，你可以在 `storage_options` 中传递 JSON 文件的路径：

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

    默认情况下，GCS 使用 HTTP/1 进行通信，而非 HTTP/2。这显著提高了最大吞吐量。
    但如果你出于某种原因希望使用 HTTP/2，可以将环境变量 `HTTP1_ONLY` 设置为 `false`。

以下键可以同时用作环境变量或 `storage_options` 参数中的键：

| 键 | 描述 |
|----|------|
| `google_service_account` / `service_account` | 服务账户 JSON 文件的路径。 |
| `google_service_account_key` / `service_account_key` | 序列化的服务账户密钥。 |
| `google_application_credentials` / `application_credentials` | 应用程序凭证的路径。 |

## Azure Blob Storage 配置

Azure Blob Storage 凭证可以通过设置 `AZURE_STORAGE_ACCOUNT_NAME` 和 `AZURE_STORAGE_ACCOUNT_KEY` 环境变量来配置。或者，你可以在 `storage_options` 参数中传递账户名称和密钥：

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

以下键可以同时用作环境变量或 `storage_options` 参数中的键：

| 键 | 描述 |
|----|------|
| `azure_storage_account_name` / `account_name` | Azure 存储账户名称。 |
| `azure_storage_account_key` / `account_key` | 序列化的服务账户密钥。 |
| `azure_client_id` / `client_id` | 用于授权请求的服务主体客户端 ID。 |
| `azure_client_secret` / `client_secret` | 用于授权请求的服务主体客户端密钥。 |
| `azure_tenant_id` / `tenant_id` | OAuth 流程中使用的租户 ID。 |
| `azure_storage_sas_key` / `azure_storage_sas_token` / `sas_key` / `sas_token` | 共享访问签名。签名应进行百分比编码，与 Azure 存储资源管理器或 Azure 门户中提供的方式相同。 |
| `azure_storage_token` / `bearer_token` / `token` | Bearer 令牌。 |
| `azure_storage_use_emulator` / `object_store_use_emulator` / `use_emulator` | 使用 Azurite 存储模拟器的对象存储。 |
| `azure_endpoint` / `endpoint` | 覆盖与 Blob 存储通信使用的端点。 |
| `azure_use_fabric_endpoint` / `use_fabric_endpoint` | 使用 URL 方案 account.dfs.fabric.microsoft.com 的对象存储。 |
| `azure_msi_endpoint` / `azure_identity_endpoint` / `identity_endpoint` / `msi_endpoint` | 请求 IMDS 托管标识令牌的端点。 |
| `azure_object_id` / `object_id` | 用于托管标识认证的对象 ID。 |
| `azure_msi_resource_id` / `msi_resource_id` | 用于托管标识认证的 MSI 资源 ID。 |
| `azure_federated_token_file` / `federated_token_file` | 包含 Azure AD 工作负载标识联合令牌的文件。 |
| `azure_use_azure_cli` / `use_azure_cli` | 使用 Azure CLI 获取访问令牌。 |
| `azure_disable_tagging` / `disable_tagging` | 禁用对象标记。如果后端存储不支持标记，这可能很有用。 |

## 阿里云对象存储服务（OSS）配置

OSS 凭证可以通过环境变量 `OSS_ACCESS_KEY_ID`、`OSS_ACCESS_KEY_SECRET`、`OSS_REGION` 和 `OSS_SECURITY_TOKEN` 设置。或者，它们可以作为参数传递给 `storage_options`：

```python
import lance
ds = lance.dataset(
    "oss://bucket/path",
    storage_options={
        "oss_region": "oss-region",
        "oss_endpoint": "oss-endpoint",
        "oss_access_key_id": "my-access-key",
        "oss_secret_access_key": "my-secret-key",
        "oss_security_token": "my-session-token",
    }
)
```

| 键 | 描述 |
|----|------|
| `oss_endpoint` | OSS 端点。必填（例如 `https://oss-cn-hangzhou.aliyuncs.com`）。 |
| `oss_access_key_id` | 用于 OSS 认证的访问密钥 ID。如果环境提供了凭证则可选。 |
| `oss_secret_access_key` | 用于 OSS 认证的访问密钥密文。如果环境提供了凭证则可选。 |
| `oss_region` | OSS 区域（例如 `cn-hangzhou`）。可选。 |
| `oss_security_token` | 临时凭证（STS）的安全令牌。可选。 |
