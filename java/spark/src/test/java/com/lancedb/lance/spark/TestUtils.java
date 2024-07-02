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

import com.lancedb.lance.spark.read.LanceInputPartition;
import com.lancedb.lance.spark.read.LanceSplit;
import com.lancedb.lance.spark.utils.Optional;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.net.URL;
import java.util.Arrays;
import java.util.List;

public class TestUtils {
  public static class TestTable1Config {
    public static final String dbPath;
    public static final String datasetName = "test_dataset1";
    public static final String datasetUri;
    public static final List<List<Long>> expectedValues = Arrays.asList(
        Arrays.asList(0L, 0L, 0L, 0L),
        Arrays.asList(1L, 2L, 3L, -1L),
        Arrays.asList(2L, 4L, 6L, -2L),
        Arrays.asList(3L, 6L, 9L, -3L)
    );
    public static final LanceConfig lanceConfig;

    public static final StructType schema = new StructType(new StructField[]{
        DataTypes.createStructField("x", DataTypes.LongType, true),
        DataTypes.createStructField("y", DataTypes.LongType, true),
        DataTypes.createStructField("b", DataTypes.LongType, true),
        DataTypes.createStructField("c", DataTypes.LongType, true),
    });
    
    public static final LanceInputPartition inputPartition;

    static {
      URL resource = TestUtils.class.getResource("/example_db");
      if (resource != null) {
        dbPath = resource.toString();
      } else {
        throw new IllegalArgumentException("example_db not found in resources directory");
      }
      datasetUri = LanceConfig.getDatasetUri(dbPath, datasetName);
      lanceConfig = LanceConfig.from(datasetUri);
      inputPartition = new LanceInputPartition(schema, 0, new LanceSplit(Arrays.asList(0, 1)), lanceConfig, Optional.empty());
    }
  }
}
