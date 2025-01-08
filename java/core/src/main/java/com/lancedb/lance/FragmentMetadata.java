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

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import com.google.gson.annotations.SerializedName;
import org.apache.arrow.util.Preconditions;
import org.apache.commons.lang3.builder.ToStringBuilder;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/** Metadata of a Fragment in the dataset. Matching to lance Fragment. */
public class FragmentMetadata implements Serializable {
  private static final long serialVersionUID = -5886811251944130460L;
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
    Gson gson = new Gson();
    try {
      Fragment fragment = gson.fromJson(jsonMetadata, Fragment.class);
      return new FragmentMetadata(jsonMetadata, fragment.getId(), fragment.getPhysicalRows());
    } catch (Exception e) {
      throw new IllegalArgumentException(e);
    }
  }

  /**
   * Converts a JSON array string into a list of FragmentMetadata objects.
   *
   * @param jsonMetadata A JSON array string containing fragment metadata.
   * @return A list of FragmentMetadata objects.
   */
  public static List<FragmentMetadata> fromJsonArray(String jsonMetadata) {
    Preconditions.checkNotNull(jsonMetadata);
    Gson gson = new Gson();
    JsonParser parser = new JsonParser();
    try {
      JsonArray fragments = parser.parse(jsonMetadata).getAsJsonArray();
      List<FragmentMetadata> fragmentMetadataList = new ArrayList<>();
      for (JsonElement fragmentE : fragments) {
        Fragment fragment = gson.fromJson(fragmentE, Fragment.class);
        fragmentMetadataList.add(
            new FragmentMetadata(
                fragmentE.toString(), fragment.getId(), fragment.getPhysicalRows()));
      }
      return fragmentMetadataList;
    } catch (Exception e) {
      throw new IllegalArgumentException(e);
    }
  }

  public static class Fragment {
    @SerializedName("id")
    private int id;

    @SerializedName("physical_rows")
    private long physicalRows;

    public Fragment(int id, long physicalRows) {
      this.id = id;
      this.physicalRows = physicalRows;
    }

    public int getId() {
      return id;
    }

    public void setId(int id) {
      this.id = id;
    }

    public long getPhysicalRows() {
      return physicalRows;
    }

    public void setPhysicalRows(long physicalRows) {
      this.physicalRows = physicalRows;
    }
  }
}
