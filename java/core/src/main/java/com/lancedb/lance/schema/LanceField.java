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

import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.DictionaryEncoding;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class LanceField {

  private final int id;
  private final String name;
  private final boolean nullable;
  private final ArrowType type;
  private final DictionaryEncoding dictionary;
  private final Map<String, String> metadata;
  private final List<LanceField> children;

  LanceField(
      int id,
      String name,
      boolean nullable,
      ArrowType type,
      DictionaryEncoding dictionary,
      Map<String, String> metadata,
      List<LanceField> children) {
    this.id = id;
    this.name = name;
    this.nullable = nullable;
    this.type = type;
    this.dictionary = dictionary;
    this.metadata = metadata;
    this.children = children;
  }

  public int getId() {
    return id;
  }

  public boolean isNullable() {
    return nullable;
  }

  public ArrowType getType() {
    return type;
  }

  public DictionaryEncoding getDictionary() {
    return dictionary;
  }

  public Map<String, String> getMetadata() {
    return metadata;
  }

  public Field asArrowField() {
    List<Field> arrowChildren =
        children.stream().map(LanceField::asArrowField).collect(Collectors.toList());
    return new Field(name, new FieldType(nullable, type, dictionary, metadata), arrowChildren);
  }

  public List<LanceField> getChildren() {
    return children;
  }
}
