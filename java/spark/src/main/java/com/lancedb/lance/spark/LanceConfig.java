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

import java.io.Serializable;
import java.util.Map;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * Lance Configuration.
 */
public class LanceConfig implements Serializable {
  private static final long serialVersionUID = 827364827364823764L;
  public static final String CONFIG_DATASET_URI = "path"; // Path is default spark option key
  public static final String CONFIG_PUSH_DOWN_FILTERS = "pushDownFilters";
  public static final String LANCE_FILE_SUFFIX = ".lance";

  private static final boolean DEFAULT_PUSH_DOWN_FILTERS = true;

  private final String dbPath;
  private final String datasetName;
  private final String datasetUri;
  private final boolean pushDownFilters;

  private LanceConfig(String dbPath, String datasetName,
      String datasetUri, boolean pushDownFilters) {
    this.dbPath = dbPath;
    this.datasetName = datasetName;
    this.datasetUri = datasetUri;
    this.pushDownFilters = pushDownFilters;
  }

  public static LanceConfig from(Map<String, String> properties) {
    return from(new CaseInsensitiveStringMap(properties));
  }

  public static LanceConfig from(CaseInsensitiveStringMap options) {
    if (!options.containsKey(CONFIG_DATASET_URI)) {
      throw new IllegalArgumentException("Missing required option " + CONFIG_DATASET_URI);
    }
    return from(options, options.get(CONFIG_DATASET_URI));
  }

  public static LanceConfig from(Map<String, String> properties, String datasetUri) {
    return from(new CaseInsensitiveStringMap(properties), datasetUri);
  }

  public static LanceConfig from(String datasetUri) {
    return from(CaseInsensitiveStringMap.empty(), datasetUri);
  }

  public static LanceConfig from(CaseInsensitiveStringMap options, String datasetUri) {
    boolean pushDownFilters = options.getBoolean(CONFIG_PUSH_DOWN_FILTERS,
        DEFAULT_PUSH_DOWN_FILTERS);
    String[] paths = extractDbPathAndDatasetName(datasetUri);
    return new LanceConfig(paths[0], paths[1], datasetUri, pushDownFilters);
  }

  public static String getDatasetUri(String dbPath, String datasetUri) {
    StringBuilder sb = new StringBuilder().append(dbPath);
    if (!dbPath.endsWith("/")) {
      sb.append("/");
    }
    return sb.append(datasetUri).append(LANCE_FILE_SUFFIX).toString();
  }

  private static String[] extractDbPathAndDatasetName(String datasetUri) {
    if (datasetUri == null || !datasetUri.endsWith(LANCE_FILE_SUFFIX)) {
      throw new IllegalArgumentException("Invalid dataset uri: " + datasetUri);
    }

    int lastSlashIndex = datasetUri.lastIndexOf('/');
    if (lastSlashIndex == -1) {
      throw new IllegalArgumentException("Invalid dataset uri: " + datasetUri);
    }

    String datasetNameWithSuffix = datasetUri.substring(lastSlashIndex + 1);
    return new String[]{datasetUri.substring(0, lastSlashIndex + 1),
        datasetNameWithSuffix.substring(0,
            datasetNameWithSuffix.length() - LANCE_FILE_SUFFIX.length())};
  }

  public String getDbPath() {
    return dbPath;
  }

  public String getDatasetName() {
    return datasetName;
  }

  public String getDatasetUri() {
    return datasetUri;
  }

  public boolean isPushDownFilters() {
    return pushDownFilters;
  }
}
