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
  private final String jsonMetadata;
  private final JSONObject metadata;

  public FragmentMetadata(String jsonMetadata) {
    this.jsonMetadata = jsonMetadata;
    this.metadata = new JSONObject(jsonMetadata);
  }

  public int getId() {
    return metadata.getInt("id");
  }
  
  public long getPhysicalRows() {
    return metadata.getLong("physical_rows");
  }

  public String getJsonMetadata() {
    return jsonMetadata;
  }
}
