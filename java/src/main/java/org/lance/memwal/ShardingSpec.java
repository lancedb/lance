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

/** Partitioning specification for deriving MemWAL shard IDs. */
public class ShardingSpec {
  private final int specId;
  private final List<ShardingField> fields;

  public ShardingSpec(int specId, List<ShardingField> fields) {
    this.specId = specId;
    this.fields = fields == null ? Collections.emptyList() : fields;
  }

  /** Identifier for this shard specification. */
  public int specId() {
    return specId;
  }

  /** The derived fields that make up this specification. */
  public List<ShardingField> fields() {
    return fields;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("specId", specId).add("fields", fields).toString();
  }
}
