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
package com.lancedb.lance.spark;

import org.apache.spark.sql.connector.catalog.Identifier;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

import java.util.HashMap;
import java.util.Map;

/**
 * LanceIdentifier is a custom implementation of {@link Identifier} for Lance. It contains the
 * dataset URI and the namespace. The namespace is an array of strings, which contains the namespace
 * of the dataset and the options. The options are key-value pairs, which are separated by "#####".
 */
public class LanceIdentifier implements Identifier {
  public static final String SEPARATOR = "#####";
  private final String[] namespace;
  private final String datasetUri;
  private final Map<String, String> options;

  public LanceIdentifier(String datasetUri) {
    this(datasetUri, "default");
  }

  public LanceIdentifier(String datasetUri, String... namespace) {
    this(datasetUri, namespace, null);
  }

  public LanceIdentifier(String datasetUri, Map<String, String> options) {
    this(datasetUri, new String[] {"default"}, options);
  }

  public LanceIdentifier(String datasetUri, String[] namespace, Map<String, String> options) {
    this.datasetUri = datasetUri;
    if (null != options && !options.isEmpty()) {
      this.namespace = new String[namespace.length + 1 + options.size()];
      System.arraycopy(namespace, 0, this.namespace, 0, namespace.length);
      this.namespace[namespace.length] = SEPARATOR;
      int i = namespace.length + 1;
      for (Map.Entry<String, String> entry : options.entrySet()) {
        this.namespace[i] = entry.getKey() + "=" + entry.getValue();
        i++;
      }
      this.options = options;
    } else {
      this.namespace = namespace;
      this.options = new HashMap<>();
    }
  }

  /**
   * Convert any identifier into LanceIdentifier.
   *
   * @param identifier may be LanceIdentifier or IdentifierImpl
   * @return LanceIdentifier
   */
  public static LanceIdentifier of(Identifier identifier) {
    if (identifier instanceof LanceIdentifier) {
      return (LanceIdentifier) identifier;
    } else {
      int index = -1;
      for (int i = 0; i < identifier.namespace().length; i++) {
        if (identifier.namespace()[i].contains(SEPARATOR)) {
          index = i;
          break;
        }
      }
      if (index > 0) {
        // when saving datasource, the IdentifierImpl will contain options in namespaces
        String[] namespace = new String[index];
        System.arraycopy(namespace, 0, identifier.namespace(), 0, index);
        Map<String, String> options = new HashMap<>();
        for (int i = index + 1; i < identifier.namespace().length; i++) {
          String keyValue = identifier.namespace()[i];
          String[] kv = keyValue.split("=");
          options.put(kv[0], kv[1]);
        }
        return new LanceIdentifier(identifier.name(), namespace, options);
      } else {
        // catalog identifier only have namespaces
        return new LanceIdentifier(identifier.name(), identifier.namespace());
      }
    }
  }

  @Override
  public String[] namespace() {
    return this.namespace;
  }

  @Override
  public String name() {
    return datasetUri;
  }

  // lance datasource options will overwrite the catalog options.
  public LanceConfig genLanceConfig(CaseInsensitiveStringMap catalogOptions) {
    Map<String, String> mergedOptions = new HashMap<>(catalogOptions);
    mergedOptions.putAll(this.options);
    return LanceConfig.from(mergedOptions, datasetUri);
  }
}
