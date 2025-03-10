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
 * dataset URI and the namespace. The key difference with IdentifierImpl is that the LanceIdentifier
 * will have the read or write options for lance. If there are some options, the `name` field will
 * change into this format "name#key1=value1&amp;key2=value2".
 */
public class LanceIdentifier implements Identifier {
  public static final String SEPARATOR = "#";
  public static final String AMPERSAND = "&";
  public static final String EQUAL = "=";
  private final String[] namespace;
  private final String datasetUri;
  private final String datasetUriWithOptions;
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
    this.namespace = namespace;
    this.datasetUri = datasetUri;
    if (null != options && !options.isEmpty()) {
      StringBuilder sb = new StringBuilder();
      sb.append(datasetUri);
      sb.append(SEPARATOR);
      for (Map.Entry<String, String> entry : options.entrySet()) {
        sb.append(entry.getKey()).append(EQUAL).append(entry.getValue());
        sb.append(AMPERSAND);
      }
      sb.deleteCharAt(sb.length() - 1);
      this.datasetUriWithOptions = sb.toString();
      this.options = options;
    } else {
      this.datasetUriWithOptions = datasetUri;
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
      if (identifier.name().contains(SEPARATOR)) {
        String nameWithOptions = identifier.name();
        String name = nameWithOptions.substring(0, nameWithOptions.indexOf(SEPARATOR));
        return new LanceIdentifier(name, identifier.namespace(), extraOptions(nameWithOptions));
      } else {
        return new LanceIdentifier(identifier.name(), identifier.namespace());
      }
    }
  }

  public static Map<String, String> extraOptions(String urlWithOptions) {
    if (urlWithOptions.contains(SEPARATOR)) {
      String optionsStr = urlWithOptions.substring(urlWithOptions.indexOf(SEPARATOR) + 1);
      Map<String, String> options = new HashMap<>();
      for (String kv : optionsStr.split(AMPERSAND)) {
        String[] keyValue = kv.split(EQUAL);
        options.put(keyValue[0], keyValue[1]);
      }
      return options;
    }
    return new HashMap<>();
  }

  @Override
  public String[] namespace() {
    return this.namespace;
  }

  @Override
  public String name() {
    return datasetUriWithOptions;
  }

  public String shortName() {
    return datasetUri;
  }

  // lance datasource options will overwrite the catalog options.
  public LanceConfig genLanceConfig(CaseInsensitiveStringMap catalogOptions) {
    Map<String, String> mergedOptions = new HashMap<>(catalogOptions);
    mergedOptions.putAll(this.options);
    return LanceConfig.from(mergedOptions, datasetUri);
  }
}
