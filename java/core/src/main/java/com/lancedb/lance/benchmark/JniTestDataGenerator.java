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
package com.lancedb.lance.benchmark;

import com.lancedb.lance.index.DistanceType;
import com.lancedb.lance.index.IndexParams;
import com.lancedb.lance.index.vector.VectorIndexParams;
import com.lancedb.lance.ipc.Query;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Generates test data for JNI benchmark tests. Based on existing test patterns from JniTestHelper.
 */
public class JniTestDataGenerator {
  private final Random random;

  public JniTestDataGenerator(long seed) {
    this.random = new Random(seed);
  }

  public List<Integer> generateIntegerList(int size) {
    List<Integer> list = new ArrayList<Integer>(size);
    for (int i = 0; i < size; i++) {
      list.add(random.nextInt());
    }
    return list;
  }

  public List<Long> generateLongList(int size) {
    List<Long> list = new ArrayList<Long>(size);
    for (int i = 0; i < size; i++) {
      list.add(random.nextLong());
    }
    return list;
  }

  public float[] generateFloatVector(int dimension) {
    float[] vector = new float[dimension];
    for (int i = 0; i < dimension; i++) {
      vector[i] = random.nextFloat();
    }
    return vector;
  }

  public Query generateQuery(int vectorDimension, int k) {
    return new Query.Builder()
        .setColumn("vector_column")
        .setKey(generateFloatVector(vectorDimension))
        .setK(k)
        .setNprobes(20)
        .setEf(30)
        .setRefineFactor(40)
        .setDistanceType(DistanceType.L2)
        .setUseIndex(true)
        .build();
  }

  public IndexParams generateIndexParams() {
    return new IndexParams.Builder()
        .setVectorIndexParams(VectorIndexParams.ivfFlat(10, DistanceType.L2))
        .build();
  }
}
