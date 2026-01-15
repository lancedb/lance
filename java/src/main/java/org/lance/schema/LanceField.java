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
package org.lance.schema;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableMap;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.DictionaryEncoding;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.stream.Collectors;

public class LanceField {
  private final int id;
  private final int parentId;
  private final String name;
  private final boolean nullable;
  private final String logicalType;
  private final ArrowType type;
  private final DictionaryEncoding dictionaryEncoding;
  private final Map<String, String> metadata;
  private final List<LanceField> children;
  private final boolean isUnenforcedPrimaryKey;
  private final int unenforcedPrimaryKeyPosition;

  LanceField(
      int id,
      int parentId,
      String name,
      boolean nullable,
      String logicalType,
      ArrowType type,
      DictionaryEncoding dictionaryEncoding,
      Map<String, String> metadata,
      List<LanceField> children,
      boolean isUnenforcedPrimaryKey,
      int unenforcedPrimaryKeyPosition) {
    this.id = id;
    this.parentId = parentId;
    this.name = name;
    this.nullable = nullable;
    this.logicalType = logicalType;
    this.type = type;
    this.dictionaryEncoding = dictionaryEncoding;
    this.metadata = metadata;
    this.children = children;
    this.isUnenforcedPrimaryKey = isUnenforcedPrimaryKey;
    this.unenforcedPrimaryKeyPosition = unenforcedPrimaryKeyPosition;
  }

  public int getId() {
    return id;
  }

  public int getParentId() {
    return parentId;
  }

  public String getName() {
    return name;
  }

  public boolean isNullable() {
    return nullable;
  }

  public String getLogicalType() {
    return logicalType;
  }

  public ArrowType getType() {
    return type;
  }

  public Optional<DictionaryEncoding> getDictionaryEncoding() {
    return Optional.ofNullable(dictionaryEncoding);
  }

  public Map<String, String> getMetadata() {
    return metadata;
  }

  public List<LanceField> getChildren() {
    return children;
  }

  public boolean isUnenforcedPrimaryKey() {
    return isUnenforcedPrimaryKey;
  }

  /**
   * Get the position of this field within a composite primary key.
   *
   * @return the 1-based position if explicitly set, or empty if using schema field id ordering
   */
  public OptionalInt getUnenforcedPrimaryKeyPosition() {
    if (unenforcedPrimaryKeyPosition > 0) {
      return OptionalInt.of(unenforcedPrimaryKeyPosition);
    }
    return OptionalInt.empty();
  }

  public Field asArrowField() {
    List<Field> arrowChildren =
        children.stream().map(LanceField::asArrowField).collect(Collectors.toList());

    if (type instanceof ArrowType.FixedSizeList) {
      arrowChildren.addAll(childrenForFixedSizeList());
    }

    return new Field(
        name, new FieldType(nullable, type, dictionaryEncoding, metadata), arrowChildren);
  }

  private List<Field> childrenForFixedSizeList() {
    if (logicalType == null || logicalType.isEmpty()) {
      return Collections.emptyList();
    }

    if (!(type instanceof ArrowType.FixedSizeList)) {
      return Collections.emptyList();
    }

    if (!logicalType.startsWith("fixed_size_list:")) {
      return Collections.emptyList();
    }

    String[] parts = logicalType.split(":");
    if (parts.length < 3) {
      throw new IllegalArgumentException("Unsupported logical type: " + logicalType);
    }

    String innerLogicalType =
        Arrays.asList(parts).subList(1, parts.length - 1).stream().collect(Collectors.joining(":"));

    Field itemField;
    switch (innerLogicalType) {
      case "lance.bfloat16":
        itemField =
            new Field(
                "item",
                new FieldType(
                    true,
                    new ArrowType.FixedSizeBinary(2),
                    null,
                    ImmutableMap.of(
                        "ARROW:extension:name", "lance.bfloat16",
                        "ARROW:extension:metadata", "")),
                Collections.emptyList());
        return Collections.singletonList(itemField);

      default:
        ArrowType elementType = arrowTypeFromLogicalType(innerLogicalType);
        itemField =
            new Field(
                "item",
                new FieldType(true, elementType, null, Collections.emptyMap()),
                Collections.emptyList());
        return Collections.singletonList(itemField);
    }
  }

  private ArrowType arrowTypeFromLogicalType(String logicalType) {
    switch (logicalType) {
      case "null":
        return ArrowType.Null.INSTANCE;
      case "bool":
        return ArrowType.Bool.INSTANCE;
      case "int8":
        return new ArrowType.Int(8, true);
      case "uint8":
        return new ArrowType.Int(8, false);
      case "int16":
        return new ArrowType.Int(16, true);
      case "uint16":
        return new ArrowType.Int(16, false);
      case "int32":
        return new ArrowType.Int(32, true);
      case "uint32":
        return new ArrowType.Int(32, false);
      case "int64":
        return new ArrowType.Int(64, true);
      case "uint64":
        return new ArrowType.Int(64, false);
      case "halffloat":
        return new ArrowType.FloatingPoint(FloatingPointPrecision.HALF);
      case "float":
        return new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE);
      case "double":
        return new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);
      case "string":
        return ArrowType.Utf8.INSTANCE;
      case "binary":
        return ArrowType.Binary.INSTANCE;
      case "large_string":
        return ArrowType.LargeUtf8.INSTANCE;
      case "large_binary":
      case "blob":
      case "json":
        return ArrowType.LargeBinary.INSTANCE;
      case "date32:day":
        return new ArrowType.Date(DateUnit.DAY);
      case "date64:ms":
        return new ArrowType.Date(DateUnit.MILLISECOND);
      case "time32:s":
        return new ArrowType.Time(TimeUnit.SECOND, 32);
      case "time32:ms":
        return new ArrowType.Time(TimeUnit.MILLISECOND, 32);
      case "time64:us":
        return new ArrowType.Time(TimeUnit.MICROSECOND, 64);
      case "time64:ns":
        return new ArrowType.Time(TimeUnit.NANOSECOND, 64);
      case "duration:s":
        return new ArrowType.Duration(TimeUnit.SECOND);
      case "duration:ms":
        return new ArrowType.Duration(TimeUnit.MILLISECOND);
      case "duration:us":
        return new ArrowType.Duration(TimeUnit.MICROSECOND);
      case "duration:ns":
        return new ArrowType.Duration(TimeUnit.NANOSECOND);
      default:
        throw new IllegalArgumentException("Unsupported logical type: " + logicalType);
    }
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("id", id)
        .add("parentId", parentId)
        .add("name", name)
        .add("nullable", nullable)
        .add("logicalType", logicalType)
        .add("type", type)
        .add("dictionaryEncoding", dictionaryEncoding)
        .add("children", children)
        .add("isUnenforcedPrimaryKey", isUnenforcedPrimaryKey)
        .add("unenforcedPrimaryKeyPosition", unenforcedPrimaryKeyPosition)
        .add("metadata", metadata)
        .toString();
  }
}
