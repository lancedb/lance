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

import com.lancedb.lance.spark.TestUtils;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.stream.Collectors;

public class SparkConnectorReadWithRowId extends SparkConnectorReadTestBase {

  @Test
  public void readAllWithoutRowId() {
    validateData(data, TestUtils.TestTable1Config.expectedValues);
  }

  @Test
  public void readAllWithRowId() {
    validateData(
        data.select("x", "y", "b", "c", "_rowid"),
        TestUtils.TestTable1Config.expectedValuesWithRowId);
  }

  @Test
  public void select() {
    validateData(
        data.select("y", "b", "_rowid"),
        TestUtils.TestTable1Config.expectedValuesWithRowId.stream()
            .map(row -> Arrays.asList(row.get(1), row.get(2), row.get(4)))
            .collect(Collectors.toList()));
  }

  @Test
  public void filterSelect() {
    validateData(
        data.select("y", "b", "_rowid").filter("y > 3"),
        TestUtils.TestTable1Config.expectedValuesWithRowId.stream()
            .map(
                row ->
                    Arrays.asList(
                        row.get(1),
                        row.get(2),
                        row.get(4))) // "y" is at index 1, "b" is at index 2, "_rowid" is at index 4
            .filter(row -> row.get(0) > 3)
            .collect(Collectors.toList()));
  }

  @Test
  public void filterSelectByRowId() {
    validateData(
        data.select("y", "b", "_rowid").filter("_rowid > 3"),
        TestUtils.TestTable1Config.expectedValuesWithRowId.stream()
            .map(
                row ->
                    Arrays.asList(
                        row.get(1),
                        row.get(2),
                        row.get(4))) // "y" is at index 1, "b" is at index 2, "_rowid" is at index 4
            .filter(row -> row.get(2) > 3)
            .collect(Collectors.toList()));
  }
}
