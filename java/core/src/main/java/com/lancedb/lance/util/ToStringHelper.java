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
package com.lancedb.lance.util;

import org.apache.commons.lang3.builder.ToStringBuilder;

import static org.apache.commons.lang3.builder.ToStringStyle.SHORT_PREFIX_STYLE;

/**
 * A helper class for toString(). In case if we want to do some customization, we can change the
 * style or use another library like guava.
 */
public class ToStringHelper {

  private final ToStringBuilder builder;

  private ToStringHelper(ToStringBuilder builder) {
    this.builder = new ToStringBuilder(builder);
  }

  public static ToStringHelper of(Object obj) {
    return new ToStringHelper(new ToStringBuilder(obj, SHORT_PREFIX_STYLE));
  }

  public ToStringHelper add(String key, Object value) {
    builder.append(key, value);
    return this;
  }

  public String toString() {
    return builder.toString();
  }
}
