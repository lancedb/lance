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

import org.apache.arrow.util.Preconditions;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/** Metadata of a Fragment in the dataset. Matching to lance Fragment. */
public class FragmentMetadata implements Serializable {
  private static final long serialVersionUID = -5886811251944130460L;
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

  @Override
  public String toString() {
    return new ToStringBuilder(this)
        .append("id", id)
        .append("physicalRows", physicalRows)
        .append("jsonMetadata", jsonMetadata)
        .toString();
  }

  /**
   * Creates the fragment metadata from json serialized string.
   *
   * @param jsonMetadata json metadata
   * @return created fragment metadata
   */
  public static FragmentMetadata fromJson(String jsonMetadata) {
    Preconditions.checkNotNull(jsonMetadata);
    JSONObject metadata = new JSONObject(jsonMetadata);
    if (!metadata.has(ID_KEY) || !metadata.has(PHYSICAL_ROWS_KEY)) {
      throw new IllegalArgumentException(
          String.format(
              "Fragment metadata must have {} and {} but is {}",
              ID_KEY,
              PHYSICAL_ROWS_KEY,
              jsonMetadata));
    }
    return new FragmentMetadata(
        jsonMetadata, metadata.getInt(ID_KEY), metadata.getLong(PHYSICAL_ROWS_KEY));
  }

  /**
   * Converts a JSON array string into a list of FragmentMetadata objects.
   *
   * @param jsonMetadata A JSON array string containing fragment metadata.
   * @return A list of FragmentMetadata objects.
   */
  public static List<FragmentMetadata> fromJsonArray(String jsonMetadata) {
    Preconditions.checkNotNull(jsonMetadata);
    JSONArray metadatas = new JSONArray(jsonMetadata);
    List<FragmentMetadata> fragmentMetadataList = new ArrayList<>();
    for (Object object : metadatas) {
      JSONObject metadata = (JSONObject) object;
      if (!metadata.has(ID_KEY) || !metadata.has(PHYSICAL_ROWS_KEY)) {
        throw new IllegalArgumentException(
            String.format(
                "Fragment metadata must have {} and {} but is {}",
                ID_KEY,
                PHYSICAL_ROWS_KEY,
                jsonMetadata));
      }
      fragmentMetadataList.add(
          new FragmentMetadata(
              metadata.toString(), metadata.getInt(ID_KEY), metadata.getLong(PHYSICAL_ROWS_KEY)));
    }
    return fragmentMetadataList;
  }
}
