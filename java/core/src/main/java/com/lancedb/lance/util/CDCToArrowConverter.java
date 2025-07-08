/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.lancedb.lance.flink;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.flink.cdc.common.event.*;
import org.apache.flink.cdc.common.data.RecordData;
import org.apache.flink.cdc.common.types.DataType;
import org.apache.flink.cdc.common.types.DataTypes;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class CDCToArrowConverter {
    private final BufferAllocator allocator;
    private final Schema arrowSchema;
    private final Map<String, Integer> fieldIndexMap;

    public CDCToArrowConverter(BufferAllocator allocator, Schema arrowSchema) {
        this.allocator = allocator;
        this.arrowSchema = arrowSchema;
        this.fieldIndexMap = createFieldIndexMap();
    }

    private Map<String, Integer> createFieldIndexMap() {
        Map<String, Integer> indexMap = new HashMap<>();
        List<Field> fields = arrowSchema.getFields();
        for (int i = 0; i < fields.size(); i++) {
            indexMap.put(fields.get(i).getName(), i);
        }
        return indexMap;
    }

    public VectorSchemaRoot convertToArrow(List<ChangeEvent> events) {
        // 创建 VectorSchemaRoot
        VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator);

        // 预分配向量容量
        int eventCount = events.size();
        for (FieldVector vector : root.getFieldVectors()) {
            vector.allocateNew(eventCount);
        }

        // 处理每个 CDC 事件
        int rowIndex = 0;
        for (ChangeEvent event : events) {
            if (event instanceof DataChangeEvent) {
                DataChangeEvent dataEvent = (DataChangeEvent) event;
                processDataChangeEvent(dataEvent, root, rowIndex);
                rowIndex++;
            } else if (event instanceof SchemaChangeEvent) {
                // 处理 schema 变更事件
                handleSchemaChangeEvent((SchemaChangeEvent) event);
            }
        }

        // 设置实际行数
        root.setRowCount(rowIndex);

        // 设置向量值计数
        for (FieldVector vector : root.getFieldVectors()) {
            vector.setValueCount(rowIndex);
        }

        return root;
    }

    private void processDataChangeEvent(DataChangeEvent event, VectorSchemaRoot root, int rowIndex) {
        RowKind op = event.op();
        RecordData recordData = null;
        String opType = null;

        switch (op) {
            case INSERT:
                recordData = event.after();
                opType = "INSERT";
                break;
            case UPDATE:
                recordData = event.after();
                opType = "UPDATE";
                break;
            case DELETE:
                recordData = event.before();
                opType = "DELETE";
                break;
        }

        if (recordData != null) {
            populateVectors(recordData, root, rowIndex);

            // 设置操作类型
            Integer opTypeIndex = fieldIndexMap.get("__op_type__");
            if (opTypeIndex != null) {
                VarCharVector opVector = (VarCharVector) root.getVector(opTypeIndex);
                opVector.setSafe(rowIndex, opType.getBytes());
            }
        }
    }

    private void populateVectors(RecordData record, VectorSchemaRoot root, int rowIndex) {
        for (int i = 0; i < record.getArity(); i++) {
            Object value = record.isNullAt(i) ? null : extractValue(record, i);
            String fieldName = getFieldNameByIndex(i);

            Integer vectorIndex = fieldIndexMap.get(fieldName);
            if (vectorIndex != null && vectorIndex < root.getFieldVectors().size()) {
                FieldVector vector = root.getVector(vectorIndex);
                setVectorValue(vector, rowIndex, value);
            }
        }
    }

    private String getFieldNameByIndex(int index) {
        // 这里需要根据实际的 CDC 事件结构来实现
        // 可能需要维护一个字段名到索引的映射
        return "field_" + index; // 占位符实现
    }

    private Object extractValue(RecordData record, int index) {
        // 根据字段类型提取值
        // 这里需要根据具体的 CDC 数据类型来实现
        try {
            return record.getString(index); // 简化实现，实际需要根据类型处理
        } catch (Exception e) {
            return null;
        }
    }

    private void setVectorValue(FieldVector vector, int index, Object value) {
        if (value == null) {
            vector.setNull(index);
            return;
        }

        try {
            if (vector instanceof IntVector) {
                ((IntVector) vector).setSafe(index, convertToInt(value));
            } else if (vector instanceof BigIntVector) {
                ((BigIntVector) vector).setSafe(index, convertToLong(value));
            } else if (vector instanceof VarCharVector) {
                ((VarCharVector) vector).setSafe(index, value.toString().getBytes());
            } else if (vector instanceof Float8Vector) {
                ((Float8Vector) vector).setSafe(index, convertToDouble(value));
            } else if (vector instanceof Float4Vector) {
                ((Float4Vector) vector).setSafe(index, convertToFloat(value));
            } else if (vector instanceof BitVector) {
                ((BitVector) vector).setSafe(index, convertToBoolean(value) ? 1 : 0);
            } else if (vector instanceof DecimalVector) {
                ((DecimalVector) vector).setSafe(index, convertToBigDecimal(value));
            } else if (vector instanceof TimeStampVector) {
                ((TimeStampVector) vector).setSafe(index, convertToTimestamp(value));
            }
            // 添加更多类型支持...
        } catch (Exception e) {
            // 类型转换失败时设置为 null
            vector.setNull(index);
        }
    }

    // 类型转换辅助方法
    private int convertToInt(Object value) {
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        return Integer.parseInt(value.toString());
    }

    private long convertToLong(Object value) {
        if (value instanceof Number) {
            return ((Number) value).longValue();
        }
        return Long.parseLong(value.toString());
    }

    private double convertToDouble(Object value) {
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        return Double.parseDouble(value.toString());
    }

    private float convertToFloat(Object value) {
        if (value instanceof Number) {
            return ((Number) value).floatValue();
        }
        return Float.parseFloat(value.toString());
    }

    private boolean convertToBoolean(Object value) {
        if (value instanceof Boolean) {
            return (Boolean) value;
        }
        return Boolean.parseBoolean(value.toString());
    }

    private BigDecimal convertToBigDecimal(Object value) {
        if (value instanceof BigDecimal) {
            return (BigDecimal) value;
        }
        return new BigDecimal(value.toString());
    }

    private long convertToTimestamp(Object value) {
        if (value instanceof LocalDateTime) {
            return ((LocalDateTime) value).toEpochSecond(java.time.ZoneOffset.UTC) * 1000;
        }
        // 尝试解析字符串格式的时间戳
        try {
            LocalDateTime dateTime = LocalDateTime.parse(value.toString(),
                    DateTimeFormatter.ISO_LOCAL_DATE_TIME);
            return dateTime.toEpochSecond(java.time.ZoneOffset.UTC) * 1000;
        } catch (Exception e) {
            return System.currentTimeMillis();
        }
    }

    private void handleSchemaChangeEvent(SchemaChangeEvent event) {
        // 处理 schema 变更事件
        // 这里可以实现动态 schema 演进逻辑
        System.out.println("Schema change detected: " + event);
    }
}