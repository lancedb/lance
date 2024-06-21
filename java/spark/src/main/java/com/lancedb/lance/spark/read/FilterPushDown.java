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

import com.lancedb.lance.spark.utils.Optional;
import org.apache.spark.sql.sources.And;
import org.apache.spark.sql.sources.EqualNullSafe;
import org.apache.spark.sql.sources.EqualTo;
import org.apache.spark.sql.sources.Filter;
import org.apache.spark.sql.sources.GreaterThan;
import org.apache.spark.sql.sources.GreaterThanOrEqual;
import org.apache.spark.sql.sources.In;
import org.apache.spark.sql.sources.IsNotNull;
import org.apache.spark.sql.sources.IsNull;
import org.apache.spark.sql.sources.LessThan;
import org.apache.spark.sql.sources.LessThanOrEqual;
import org.apache.spark.sql.sources.Not;
import org.apache.spark.sql.sources.Or;
import org.apache.spark.sql.sources.StringContains;
import org.apache.spark.sql.sources.StringEndsWith;
import org.apache.spark.sql.sources.StringStartsWith;

import java.sql.Date;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class FilterPushDown {
  /**
   * Create SQL 'where clause' from Spark filters.
   *
   * @param filters Supported spark filters
   * @return where clause, or Optional.empty() if filters do not exist
   */
  public static Optional<String> compileFiltersToSqlWhereClause(Filter[] filters) {
    if (filters.length == 0) {
      return Optional.empty();
    }
    List<String> compiledFilters = new ArrayList<>();
    for (Filter filter : filters) {
      compileFilter(filter).ifPresent(compiledFilters::add);
    }
    String whereClause = compiledFilters.stream()
        .map(filter -> "(" + filter + ")")
        .collect(Collectors.joining(" AND "));
    return Optional.of(whereClause);
  }

  /**
   * @param filters filters to see if Lance supported the filter push down
   * @return the accepted push down filters (row 0) and rejected post scan filters (row 1)
   */
  public static Filter[][] processFilters(Filter[] filters) {
    List<Filter> acceptedFilters = new ArrayList<>();
    List<Filter> rejectedFilters = new ArrayList<>();

    for (Filter filter : filters) {
      if (isFilterSupported(filter)) {
        acceptedFilters.add(filter);
      } else {
        rejectedFilters.add(filter);
      }
    }

    Filter[] acceptedArray = acceptedFilters.toArray(new Filter[0]);
    Filter[] rejectedArray = rejectedFilters.toArray(new Filter[0]);

    return new Filter[][]{acceptedArray, rejectedArray};
  }

  public static boolean isFilterSupported(Filter filter) {
    if (filter instanceof EqualTo) {
      return true;
    } else if (filter instanceof EqualNullSafe) {
      return false;
    } else if (filter instanceof In) {
      return false;
    } else if (filter instanceof LessThan) {
      return true;
    } else if (filter instanceof LessThanOrEqual) {
      return true;
    } else if (filter instanceof GreaterThan) {
      return true;
    } else if (filter instanceof GreaterThanOrEqual) {
      return true;
    } else if (filter instanceof IsNull) {
      return true;
    } else if (filter instanceof IsNotNull) {
      return true;
    } else if (filter instanceof StringStartsWith) {
      return false;
    } else if (filter instanceof StringEndsWith) {
      return false;
    } else if (filter instanceof StringContains) {
      return false;
    } else if (filter instanceof Not) {
      Not f = (Not) filter;
      return isFilterSupported(f.child());
    } else if (filter instanceof Or) {
      Or f = (Or) filter;
      return isFilterSupported(f.left()) && isFilterSupported(f.right());
    } else if (filter instanceof And) {
      And f = (And) filter;
      return isFilterSupported(f.left()) && isFilterSupported(f.right());
    } else {
      return false;
    }
  }

  private static Optional<String> compileFilter(Filter filter) {
    if (filter instanceof GreaterThan) {
      GreaterThan f = (GreaterThan) filter;
      return Optional.of(f.attribute() + " > " + compileValue(f.value()));
    } else if (filter instanceof LessThan) {
      LessThan f = (LessThan) filter;
      return Optional.of(f.attribute() + " < " + compileValue(f.value()));
    } else if (filter instanceof LessThanOrEqual) {
      LessThanOrEqual f = (LessThanOrEqual) filter;
      return Optional.of(f.attribute() + " <= " + compileValue(f.value()));
    } else if (filter instanceof GreaterThanOrEqual) {
      GreaterThanOrEqual f = (GreaterThanOrEqual) filter;
      return Optional.of(f.attribute() + " >= " + compileValue(f.value()));
    } else if (filter instanceof EqualTo) {
      EqualTo f = (EqualTo) filter;
      return Optional.of(f.attribute() + " == " + compileValue(f.value()));
    } else if (filter instanceof Or) {
      Or f = (Or) filter;
      Optional<String> left = compileFilter(f.left());
      Optional<String> right = compileFilter(f.right());
      if (left.isEmpty()) return right;
      if (right.isEmpty()) return left;
      return Optional.of(String.format("(%s) OR (%s)", left.get(), right.get()));
    } else if (filter instanceof And) {
      And f = (And) filter;
      Optional<String> left = compileFilter(f.left());
      Optional<String> right = compileFilter(f.right());
      if (left.isEmpty()) return right;
      if (right.isEmpty()) return left;
      return Optional.of(String.format("(%s) AND (%s)",
          left.get(), right.get()));
    } else if (filter instanceof IsNull) {
      IsNull f = (IsNull) filter;
      return Optional.of(String.format("%s IS NULL", f.attribute()));
    } else if (filter instanceof  IsNotNull) {
      IsNotNull f = (IsNotNull) filter;
      return Optional.of(String.format("%s IS NOT NULL", f.attribute()));
    } else if (filter instanceof Not) {
      Not f = (Not) filter;
      Optional<String> child = compileFilter(f.child());
      if (child.isEmpty()) return child;
      return Optional.of(String.format("NOT (%s)", child.get()));
    }

    return Optional.empty();
  }

  private static String compileValue(Object value) {
    if (value instanceof String || value instanceof Timestamp || value instanceof Date) {
      return "'" + value + "'";
    } else if (value instanceof Object[]) {
      Object[] array = (Object[]) value;
      StringBuilder sb = new StringBuilder();
      for (Object obj : array) {
        if (sb.length() > 0) {
          sb.append(", ");
        }
        sb.append(compileValue(obj));
      }
      return sb.toString();
    } else {
      return value.toString();
    }
  }
}
