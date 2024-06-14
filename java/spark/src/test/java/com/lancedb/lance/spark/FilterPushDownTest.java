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

import com.lancedb.lance.spark.query.FilterPushDown;
import com.lancedb.lance.spark.utils.Optional;
import org.apache.spark.sql.sources.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class FilterPushDownTest {

  @Test
  public void testCompileFiltersToSqlWhereClause() {
    // Test case 1: GreaterThan, LessThanOrEqual, IsNotNull
    Filter[] filters1 = new Filter[]{
        new GreaterThan("age", 30),
        new LessThanOrEqual("salary", 100000),
        new IsNotNull("name")
    };
    Optional<String> whereClause1 = FilterPushDown.compileFiltersToSqlWhereClause(filters1);
    assertTrue(whereClause1.isPresent());
    assertEquals("(age > 30) AND (salary <= 100000) AND (name IS NOT NULL)", whereClause1.get());

    // Test case 2: GreaterThan, StringContains, LessThan
    Filter[] filters2 = new Filter[]{
        new GreaterThan("age", 30),
        new StringContains("name", "John"),
        new LessThan("salary", 50000)
    };
    Optional<String> whereClause2 = FilterPushDown.compileFiltersToSqlWhereClause(filters2);
    assertTrue(whereClause2.isPresent());
    assertEquals("(age > 30) AND (salary < 50000)", whereClause2.get());

    // Test case 3: Empty filters array
    Filter[] filters3 = new Filter[]{};
    Optional<String> whereClause3 = FilterPushDown.compileFiltersToSqlWhereClause(filters3);
    assertFalse(whereClause3.isPresent());

    // Test case 4: Mixed supported and unsupported filters
    Filter[] filters4 = new Filter[]{
        new GreaterThan("age", 30),
        new StringContains("name", "John"),
        new IsNull("address"),
        new EqualTo("country", "USA")
    };
    Optional<String> whereClause4 = FilterPushDown.compileFiltersToSqlWhereClause(filters4);
    assertTrue(whereClause4.isPresent());
    assertEquals("(age > 30) AND (address IS NULL) AND (country == 'USA')", whereClause4.get());

    // Test case 5: Not, Or, And combinations
    Filter[] filters5 = new Filter[]{
        new Not(new GreaterThan("age", 30)),
        new Or(new IsNotNull("name"), new IsNull("address")),
        new And(new LessThan("salary", 100000), new GreaterThanOrEqual("salary", 50000))
    };
    Optional<String> whereClause5 = FilterPushDown.compileFiltersToSqlWhereClause(filters5);
    assertTrue(whereClause5.isPresent());
    assertEquals("(NOT (age > 30)) AND ((name IS NOT NULL) OR (address IS NULL)) AND ((salary < 100000) AND (salary >= 50000))", whereClause5.get());
  }

  @Test
  public void testCompileFiltersToSqlWhereClauseWithEmptyFilters() {
    Filter[] filters = new Filter[]{};

    Optional<String> whereClause = FilterPushDown.compileFiltersToSqlWhereClause(filters);
    assertFalse(whereClause.isPresent());
  }
}