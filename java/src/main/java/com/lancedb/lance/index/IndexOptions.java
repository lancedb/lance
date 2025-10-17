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
package com.lancedb.lance.index;

import java.util.List;

/** Options of building indexes. */
public class IndexOptions {
  private final boolean replace;
  private final boolean train;
  private final List<Integer> fragmentIds;
  private final String fragmentUUID;

  private IndexOptions(
      boolean replace, boolean train, List<Integer> fragmentIds, String fragmentUUID) {
    this.replace = replace;
    this.train = train;
    this.fragmentIds = fragmentIds;
    this.fragmentUUID = fragmentUUID;
  }

  public String getFragmentUUID() {
    return fragmentUUID;
  }

  public List<Integer> getFragmentIds() {
    return fragmentIds;
  }

  public boolean isReplace() {
    return replace;
  }

  public boolean isTrain() {
    return train;
  }

  public static Builder builder() {
    return new Builder();
  }

  /** Builder class. */
  public static class Builder {
    private boolean replace = false;
    private boolean train = true;
    private List<Integer> fragmentIds = null;
    private String fragmentUUID = null;

    private Builder() {}

    /**
     * Replace the existing index if it exists.
     *
     * @param replace replace option
     */
    public Builder replace(boolean replace) {
      this.replace = replace;
      return this;
    }

    /**
     * If True, the index will be trained on the data to determine optimal structure. If False, an
     * empty index will be created that can be populated later.
     *
     * @param train train option
     */
    public Builder train(boolean train) {
      this.train = train;
      return this;
    }

    /**
     * If provided, the index will be created only on the specified fragments. This enables
     * distributed/fragment-level indexing. When provided, the method returns an IndexMetadata
     * object but does not commit the index to the dataset. The index can be committed later using
     * the commit API.
     *
     * @param fragmentIds fragmentIds option
     */
    public Builder withFragmentIds(List<Integer> fragmentIds) {
      this.fragmentIds = fragmentIds;
      return this;
    }

    /**
     * A UUID to use for fragment-level distributed indexing multiple fragment-level indices need to
     * share UUID for later merging. If not provided, a new UUID will be generated.
     *
     * @param fragmentUUID fragmentUUID option
     */
    public Builder withFragmentUUID(String fragmentUUID) {
      this.fragmentUUID = fragmentUUID;
      return this;
    }

    public IndexOptions build() {
      return new IndexOptions(replace, train, fragmentIds, fragmentUUID);
    }
  }
}
