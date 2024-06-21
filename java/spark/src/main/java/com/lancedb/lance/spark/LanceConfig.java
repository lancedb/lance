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
  public static final String CONFIG_TABLE_PATH = "path";
  public static final String CONFIG_PUSH_DOWN_FILTERS = "pushDownFilters";
  public static final String LANCE_FILE_SUFFIX = ".lance";

  private static final boolean DEFAULT_PUSH_DOWN_FILTERS = true;

  private final String dbPath;
  private final String tableName;
  private final String tablePath;
  private final boolean pushDownFilters;

  private LanceConfig(String dbPath, String tableName, String tablePath, boolean pushDownFilters) {
    this.dbPath = dbPath;
    this.tableName = tableName;
    this.tablePath = tablePath;
    this.pushDownFilters = pushDownFilters;
  }

  public static LanceConfig from(Map<String, String> properties) {
    return from(new CaseInsensitiveStringMap(properties));
  }

  public static LanceConfig from(CaseInsensitiveStringMap options) {
    if (!options.containsKey(CONFIG_TABLE_PATH)) {
      throw new IllegalArgumentException("Missing required option " + CONFIG_TABLE_PATH);
    }
    return from(options, options.get(CONFIG_TABLE_PATH));
  }

  public static LanceConfig from(Map<String, String> properties, String tablePath) {
    return from(new CaseInsensitiveStringMap(properties), tablePath);
  }

  public static LanceConfig from(String tablePath) {
    return from(CaseInsensitiveStringMap.empty(), tablePath);
  }

  public static LanceConfig from(CaseInsensitiveStringMap options, String tablePath) {
    boolean pushDownFilters = options.getBoolean(CONFIG_PUSH_DOWN_FILTERS,
        DEFAULT_PUSH_DOWN_FILTERS);
    String[] paths = extractDbPathAndTableName(tablePath);
    return new LanceConfig(paths[0], paths[1], tablePath, pushDownFilters);
  }

  public static String getTablePath(String dbPath, String tableName) {
    StringBuilder sb = new StringBuilder().append(dbPath);
    if (!dbPath.endsWith("/")) {
      sb.append("/");
    }
    return sb.append(tableName).append(LANCE_FILE_SUFFIX).toString();
  }

  private static String[] extractDbPathAndTableName(String tablePath) {
    if (tablePath == null || !tablePath.endsWith(LANCE_FILE_SUFFIX)) {
      throw new IllegalArgumentException("Invalid table path: " + tablePath);
    }

    int lastSlashIndex = tablePath.lastIndexOf('/');
    if (lastSlashIndex == -1) {
      throw new IllegalArgumentException("Invalid table path: " + tablePath);
    }

    String tableNameWithSuffix = tablePath.substring(lastSlashIndex + 1);
    return new String[]{tablePath.substring(0, lastSlashIndex + 1),
        tableNameWithSuffix.substring(0,
            tableNameWithSuffix.length() - LANCE_FILE_SUFFIX.length())};
  }

  public String getDbPath() {
    return dbPath;
  }

  public String getTableName() {
    return tableName;
  }

  public String getTablePath() {
    return tablePath;
  }

  public boolean isPushDownFilters() {
    return pushDownFilters;
  }
}
