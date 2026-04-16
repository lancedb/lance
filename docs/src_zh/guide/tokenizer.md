# 分词器（Tokenizer）

目前，Lance 内置支持 Jieba 和 Lindera。但它不自带语言模型。如果需要分词功能，你可以自行下载语言模型。你可以通过设置环境变量 `LANCE_LANGUAGE_MODEL_HOME` 来指定语言模型的存储位置。如果未设置，默认值为：

```bash
${system data directory}/lance/language_models
```

它还支持配置用户词典，方便用户扩展自己的词典而无需重新训练语言模型。

## Jieba 语言模型

### 下载模型

```bash
python -m lance.download jieba
```

语言模型默认存储在 `${LANCE_LANGUAGE_MODEL_HOME}/jieba/default`。

### 使用模型

```python
ds.create_scalar_index("text", "INVERTED", base_tokenizer="jieba/default")
```

### 用户词典

在当前模型的根目录下创建名为 config.json 的文件。

```json
{
    "main": "dict.txt",
    "users": ["path/to/user/dict.txt"]
}
```

- "main" 字段可选。如果不填，默认为 "dict.txt"。
- "users" 是用户词典的路径。用户词典的格式请参考 https://github.com/messense/jieba-rs/blob/main/src/data/dict.txt。

## Lindera 语言模型

### 下载模型

```bash
python -m lance.download lindera -l [ipadic|ko-dic|unidic]
```

注意 Lindera 的语言模型需要编译。请先安装 lindera-cli。详细步骤请参考 https://github.com/lindera/lindera/tree/main/lindera-cli。

语言模型默认存储在 ${LANCE_LANGUAGE_MODEL_HOME}/lindera/[ipadic|ko-dic|unidic]

### 使用模型

```python
ds.create_scalar_index("text", "INVERTED", base_tokenizer="lindera/ipadic")
```

### 用户词典

在模型根目录下创建名为 config.yml 的文件，或使用 `LINDERA_CONFIG_PATH` 环境变量指定自定义 YAML 文件。如果两者都提供，将使用根目录下的 config.yml。更详细的配置方法请参阅 lindera 文档 https://github.com/lindera/lindera/。

```yaml
segmenter:
    mode: "normal"
    dictionary:
        # Note: in lance, the `kind` field is not supported. You need to specify the model path using the `path` field instead.
        path: /path/to/lindera/ipadic/main
```

## 创建自定义语言模型

将你的语言模型放入 `LANCE_LANGUAGE_MODEL_HOME`。
