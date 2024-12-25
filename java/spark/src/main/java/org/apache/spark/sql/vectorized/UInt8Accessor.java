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
package org.apache.spark.sql.vectorized;

import org.apache.arrow.vector.UInt8Vector;

// UInt8Accessor can't extend the ArrowVectorAccessor since it's package private.
public class UInt8Accessor {
  private final UInt8Vector accessor;

  UInt8Accessor(UInt8Vector vector) {
    this.accessor = vector;
  }

  final long getLong(int rowId) {
    return accessor.getObjectNoOverflow(rowId).longValueExact();
  }

  final boolean isNullAt(int rowId) {
    return accessor.isNull(rowId);
  }

  final int getNullCount() {
    return accessor.getNullCount();
  }

  final void close() {
    accessor.close();
  }
}
