package com.lancedb.lance.spark.query;

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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
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
  
  private static Optional<String> compileFilter (Filter filter) {
    if (filter instanceof GreaterThan) {
      GreaterThan f = (GreaterThan) filter;
      return Optional.of(escapeAttr(f.attribute()) + " > " + f.value());
    }
    return Optional.empty();
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
      return false;
    } else if (filter instanceof EqualNullSafe) {
      return false;
    } else if (filter instanceof In) {
      return false;
    } else if (filter instanceof LessThan) {
      return false;
    } else if (filter instanceof LessThanOrEqual) {
      return false;
    } else if (filter instanceof GreaterThan) {
      return true;
    } else if (filter instanceof GreaterThanOrEqual) {
      return false;
    } else if (filter instanceof IsNull) {
      return false;
    } else if (filter instanceof IsNotNull) {
      return false;
    } else if (filter instanceof StringStartsWith) {
      return false;
    } else if (filter instanceof StringEndsWith) {
      return false;
    } else if (filter instanceof StringContains) {
      return false;
    } else if (filter instanceof Not) {
      return false;
    } else if (filter instanceof Or) {
      return false;
    } else if (filter instanceof And) {
      return false;
    } else {
      return false;
    }
  }

  private static String escapeAttr(String attr) {
    if (attr.contains("\"")) {
      return attr;
    } else {
      return "\"" + attr + "\"";
    }
  }
}
