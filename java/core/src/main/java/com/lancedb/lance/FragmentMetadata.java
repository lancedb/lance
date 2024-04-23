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

package com.lancedb.lance;

import org.json.JSONObject;

/**
 * Metadata of a Fragment in the dataset. 
 * Matching to lance Fragment.
 * */
public class FragmentMetadata {
  private static final String ID_KEY = "id";
  private static final String PHYSICAL_ROWS_KEY = "physical_rows";
  private final String jsonMetadata;
  private final int id;
  private final long physicalRows;

  private FragmentMetadata(String jsonMetadata, int id, long physicalRows) {
    this.jsonMetadata = jsonMetadata;
    this.id = id;
    this.physicalRows = physicalRows;
  }

  public int getId() {
    return id;
  }
  
  public long getPhysicalRows() {
    return physicalRows;
  }

  public String getJsonMetadata() {
    return jsonMetadata;
  }

  /**
   * Creates the fragment metadata from json serialized string.
   *
   * @param jsonMetadata json metadata
   * @return created fragment metadata
   */
  public static FragmentMetadata fromJson(String jsonMetadata) {
    JSONObject metadata = new JSONObject(jsonMetadata);
    if (!metadata.has(ID_KEY) || !metadata.has(PHYSICAL_ROWS_KEY)) {
      throw new IllegalArgumentException(
          String.format("Fragment metadata must have {} and {} but is {}",
          ID_KEY, PHYSICAL_ROWS_KEY, jsonMetadata));
    }
    return new FragmentMetadata(jsonMetadata, metadata.getInt(ID_KEY),
        metadata.getLong(PHYSICAL_ROWS_KEY));
  }
}
