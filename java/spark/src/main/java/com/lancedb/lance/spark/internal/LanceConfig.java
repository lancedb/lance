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

package com.lancedb.lance.spark.internal;

import java.io.Serializable;
import java.util.Map;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * Lance Configuration.
 */
public class LanceConfig implements Serializable {
  private static final long serialVersionUID = 827364827364823764L;
  public static final String CONFIG_DB_PATH = "db";
  public static final String CONFIG_TABLE_NAME = "table";
  public static final String CONFIG_PUSH_DOWN_FILTERS = "pushDownFilters";
  private static final String LANCE_FILE_SUFFIX = ".lance";

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
    if (!options.containsKey(CONFIG_DB_PATH) || !options.containsKey(CONFIG_TABLE_NAME)) {
      throw new IllegalArgumentException("Missing required options");
    }

    String dbPath = options.get(CONFIG_DB_PATH);
    String tableName = options.get(CONFIG_TABLE_NAME);
    boolean pushDownFilters = options.getBoolean(CONFIG_PUSH_DOWN_FILTERS,
        DEFAULT_PUSH_DOWN_FILTERS);

    String tablePath = calculateTablePath(dbPath, tableName);

    return new LanceConfig(dbPath, tableName, tablePath, pushDownFilters);
  }

  private static String calculateTablePath(String dbPath, String tableName) {
    StringBuilder sb = new StringBuilder().append(dbPath);
    if (!dbPath.endsWith("/")) {
      sb.append("/");
    }
    return sb.append(tableName).append(LANCE_FILE_SUFFIX).toString();
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
