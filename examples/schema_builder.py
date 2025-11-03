import pyarrow as pa


def build_arrow_schema(records, compression_algo="zstd", compression_level="22"):
    table = pa.Table.from_pylist(records)  # records 是获得的 scenes 的记录
    new_fields = []
    for field in table.schema:  # 根据 table 解析得到的 schema ，一个一个遍历类型
        if pa.types.is_binary(field.type):
            field = field.with_metadata(
                {
                    "lance-encoding:compression": compression_algo,
                    "lance-encoding:compression-level": compression_level,
                }
            )
        new_fields.append(
            field
        )  # 遍历得到的类型添加到数组中，如果是特殊类型二进制的，需要添加压缩情况
    return pa.schema(new_fields)  # 把字段中所有的类型转换为新的
