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
package com.lancedb.lance.schema;

import com.google.common.base.MoreObjects;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.DictionaryEncoding;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

public class LanceField {
  private final int id;
  private final int parentId;
  private final String name;
  private final boolean nullable;
  private final ArrowType type;
  private final DictionaryEncoding dictionaryEncoding;
  private final Map<String, String> metadata;
  private final List<LanceField> children;
  private final boolean isUnenforcedPrimaryKey;

  LanceField(
      int id,
      int parentId,
      String name,
      boolean nullable,
      ArrowType type,
      DictionaryEncoding dictionaryEncoding,
      Map<String, String> metadata,
      List<LanceField> children,
      boolean isUnenforcedPrimaryKey) {
    this.id = id;
    this.parentId = parentId;
    this.name = name;
    this.nullable = nullable;
    this.type = type;
    this.dictionaryEncoding = dictionaryEncoding;
    this.metadata = metadata;
    this.children = children;
    this.isUnenforcedPrimaryKey = isUnenforcedPrimaryKey;
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

  public Field asArrowField() {
    List<Field> arrowChildren =
        children.stream().map(LanceField::asArrowField).collect(Collectors.toList());
    return new Field(
        name, new FieldType(nullable, type, dictionaryEncoding, metadata), arrowChildren);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("id", id)
        .add("parentId", parentId)
        .add("name", name)
        .add("nullable", nullable)
        .add("type", type)
        .add("dictionaryEncoding", dictionaryEncoding)
        .add("children", children)
        .add("isUnenforcedPrimaryKey", isUnenforcedPrimaryKey)
        .add("metadata", metadata)
        .toString();
  }
}
