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

package com.lancedb.lance.spark.read;

import com.lancedb.lance.spark.LanceConfig;
import com.lancedb.lance.spark.internal.LanceDatasetAdapter;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class LanceSplit implements Serializable {
  private static final long serialVersionUID = 2983749283749283749L;

  private final List<Integer> fragments;

  public LanceSplit(List<Integer> fragments) {
    this.fragments = fragments;
  }

  public List<Integer> getFragments() {
    return fragments;
  }

  public static List<LanceSplit> generateLanceSplits(LanceConfig config) {
    return LanceDatasetAdapter.getFragmentIds(config).stream()
        .map(id -> new LanceSplit(Collections.singletonList(id)))
        .collect(Collectors.toList());
  }
}
