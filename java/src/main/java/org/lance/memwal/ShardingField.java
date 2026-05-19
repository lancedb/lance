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
package org.lance.memwal;

import com.google.common.base.MoreObjects;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** One derived field used in a MemWAL shard partitioning specification. */
public class ShardingField {
  private final String fieldId;
  private final List<Integer> sourceIds;
  private final Optional<String> transform;
  private final Optional<String> expression;
  private final String resultType;
  private final Map<String, String> parameters;

  /**
   * @param fieldId identifier for the derived field
   * @param sourceIds source field IDs used as inputs
   * @param transform optional transform name applied to the source fields, may be {@code null}
   * @param expression optional expression used to derive the field value, may be {@code null}
   * @param resultType output type name for the derived field
   * @param parameters extra transform parameters
   */
  public ShardingField(
      String fieldId,
      List<Integer> sourceIds,
      String transform,
      String expression,
      String resultType,
      Map<String, String> parameters) {
    this.fieldId = fieldId;
    this.sourceIds = sourceIds == null ? Collections.emptyList() : sourceIds;
    this.transform = Optional.ofNullable(transform);
    this.expression = Optional.ofNullable(expression);
    this.resultType = resultType;
    this.parameters = parameters == null ? Collections.emptyMap() : parameters;
  }

  /** Identifier for the derived field. */
  public String fieldId() {
    return fieldId;
  }

  /** Source field IDs used as inputs. */
  public List<Integer> sourceIds() {
    return sourceIds;
  }

  /** Optional transform name applied to the source fields. */
  public Optional<String> transform() {
    return transform;
  }

  /** Optional expression used to derive the field value. */
  public Optional<String> expression() {
    return expression;
  }

  /** Output type name for the derived field. */
  public String resultType() {
    return resultType;
  }

  /** Extra transform parameters. */
  public Map<String, String> parameters() {
    return parameters;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("fieldId", fieldId)
        .add("sourceIds", sourceIds)
        .add("transform", transform.orElse(null))
        .add("expression", expression.orElse(null))
        .add("resultType", resultType)
        .add("parameters", parameters)
        .toString();
  }
}
