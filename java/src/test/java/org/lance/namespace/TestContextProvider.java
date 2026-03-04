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
package org.lance.namespace;

import java.util.HashMap;
import java.util.Map;

/** Test implementation of DynamicContextProvider for testing class path loading. */
public class TestContextProvider implements DynamicContextProvider {
  private final String token;
  private final String prefix;

  public TestContextProvider(Map<String, String> properties) {
    this.token = properties.get("token");
    this.prefix = properties.getOrDefault("prefix", "Bearer");
  }

  @Override
  public Map<String, String> provideContext(String operation, String objectId) {
    Map<String, String> context = new HashMap<>();
    context.put("headers.Authorization", prefix + " " + token);
    context.put("headers.X-Operation", operation);
    return context;
  }
}
