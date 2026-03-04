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
package org.lance.ipc;

import com.google.common.base.MoreObjects;
import org.apache.arrow.util.Preconditions;

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/** Base type for full text search queries used by Lance scanner. */
public abstract class FullTextQuery {
  public enum Type {
    MATCH,
    MATCH_PHRASE,
    BOOST,
    MULTI_MATCH,
    BOOLEAN
  }

  public enum Operator {
    AND,
    OR
  }

  public enum Occur {
    SHOULD,
    MUST,
    MUST_NOT
  }

  public static final class BooleanClause {
    private final Occur occur;
    private final FullTextQuery query;

    public BooleanClause(Occur occur, FullTextQuery query) {
      this.occur = Objects.requireNonNull(occur, "occur must not be null");
      this.query = Objects.requireNonNull(query, "query must not be null");
    }

    public Occur getOccur() {
      return occur;
    }

    public FullTextQuery getQuery() {
      return query;
    }
  }

  public abstract Type getType();

  public static FullTextQuery match(String queryText, String column) {
    return match(queryText, column, 1.0f, Optional.empty(), 50, Operator.OR, 0);
  }

  public static FullTextQuery match(
      String queryText,
      String column,
      float boost,
      Optional<Integer> fuzziness,
      int maxExpansions,
      Operator operator,
      int prefixLength) {
    return new MatchQuery(
        queryText, column, boost, fuzziness, maxExpansions, operator, prefixLength);
  }

  public static FullTextQuery phrase(String queryText, String column) {
    return phrase(queryText, column, 0);
  }

  public static FullTextQuery phrase(String queryText, String column, int slop) {
    return new PhraseQuery(queryText, column, slop);
  }

  public static FullTextQuery multiMatch(String queryText, List<String> columns) {
    return multiMatch(queryText, columns, null, Operator.OR);
  }

  public static FullTextQuery multiMatch(
      String queryText, List<String> columns, List<Float> boosts, Operator operator) {
    return new MultiMatchQuery(queryText, columns, boosts, operator);
  }

  public static FullTextQuery boost(FullTextQuery positive, FullTextQuery negative) {
    return boost(positive, negative, 0.5f);
  }

  public static FullTextQuery boost(
      FullTextQuery positive, FullTextQuery negative, float negativeBoost) {
    return new BoostQuery(positive, negative, negativeBoost);
  }

  public static FullTextQuery booleanQuery(List<BooleanClause> clauses) {
    return new BooleanQuery(clauses);
  }

  /** Match query on a single column. */
  public static final class MatchQuery extends FullTextQuery {
    private final String queryText;
    private final String column;
    private final float boost;
    private final Optional<Integer> fuzziness;
    private final int maxExpansions;
    private final Operator operator;
    private final int prefixLength;

    MatchQuery(
        String queryText,
        String column,
        float boost,
        Optional<Integer> fuzziness,
        int maxExpansions,
        Operator operator,
        int prefixLength) {
      Preconditions.checkArgument(
          queryText != null && !queryText.isEmpty(), "queryText must not be null or empty");
      Preconditions.checkArgument(
          column != null && !column.isEmpty(), "column must not be null or empty");
      Preconditions.checkArgument(maxExpansions >= 1, "maxExpansions must be >= 1");
      Preconditions.checkArgument(prefixLength >= 0, "prefixLength must be >= 0");

      this.queryText = queryText;
      this.column = column;
      this.boost = boost;
      this.fuzziness = fuzziness;
      this.maxExpansions = maxExpansions;
      this.operator = operator == null ? Operator.OR : operator;
      this.prefixLength = prefixLength;
    }

    @Override
    public Type getType() {
      return Type.MATCH;
    }

    public String getQueryText() {
      return queryText;
    }

    public String getColumn() {
      return column;
    }

    public float getBoost() {
      return boost;
    }

    public Optional<Integer> getFuzziness() {
      return fuzziness;
    }

    public int getMaxExpansions() {
      return maxExpansions;
    }

    public Operator getOperator() {
      return operator;
    }

    public int getPrefixLength() {
      return prefixLength;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("type", getType())
          .add("queryText", queryText)
          .add("column", column)
          .add("boost", boost)
          .add("fuzziness", fuzziness)
          .add("maxExpansions", maxExpansions)
          .add("operator", operator)
          .add("prefixLength", prefixLength)
          .toString();
    }
  }

  /** Phrase query on a single column. */
  public static final class PhraseQuery extends FullTextQuery {
    private final String queryText;
    private final String column;
    private final int slop;

    PhraseQuery(String queryText, String column, int slop) {
      Preconditions.checkArgument(
          queryText != null && !queryText.isEmpty(), "queryText must not be null or empty");
      Preconditions.checkArgument(
          column != null && !column.isEmpty(), "column must not be null or empty");
      Preconditions.checkArgument(slop >= 0, "slop must be >= 0");

      this.queryText = queryText;
      this.column = column;
      this.slop = slop;
    }

    @Override
    public Type getType() {
      return Type.MATCH_PHRASE;
    }

    public String getQueryText() {
      return queryText;
    }

    public String getColumn() {
      return column;
    }

    public int getSlop() {
      return slop;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("type", getType())
          .add("queryText", queryText)
          .add("column", column)
          .add("slop", slop)
          .toString();
    }
  }

  /** Multi-match query across multiple columns. */
  public static final class MultiMatchQuery extends FullTextQuery {
    private final String queryText;
    private final List<String> columns;
    private final Optional<List<Float>> boosts;
    private final Operator operator;

    MultiMatchQuery(String queryText, List<String> columns, List<Float> boosts, Operator operator) {
      Preconditions.checkArgument(
          queryText != null && !queryText.isEmpty(), "queryText must not be null or empty");
      Preconditions.checkArgument(
          columns != null && !columns.isEmpty(), "columns must not be null or empty");

      this.queryText = queryText;
      this.columns =
          Collections.unmodifiableList(new java.util.ArrayList<>(Objects.requireNonNull(columns)));
      this.boosts = boosts == null ? Optional.empty() : Optional.of(boosts);
      this.operator = operator == null ? Operator.OR : operator;
    }

    @Override
    public Type getType() {
      return Type.MULTI_MATCH;
    }

    public String getQueryText() {
      return queryText;
    }

    public List<String> getColumns() {
      return columns;
    }

    public Optional<List<Float>> getBoosts() {
      return boosts;
    }

    public Operator getOperator() {
      return operator;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("type", getType())
          .add("queryText", queryText)
          .add("columns", columns)
          .add("boosts", boosts)
          .add("operator", operator)
          .toString();
    }
  }

  /** Boost query combining positive and negative queries. */
  public static final class BoostQuery extends FullTextQuery {
    private final FullTextQuery positive;
    private final FullTextQuery negative;
    private final Float negativeBoost;

    BoostQuery(FullTextQuery positive, FullTextQuery negative, float negativeBoost) {
      this.positive = Objects.requireNonNull(positive, "positive must not be null");
      this.negative = Objects.requireNonNull(negative, "negative must not be null");
      this.negativeBoost = negativeBoost;
    }

    @Override
    public Type getType() {
      return Type.BOOST;
    }

    public FullTextQuery getPositive() {
      return positive;
    }

    public FullTextQuery getNegative() {
      return negative;
    }

    public float getNegativeBoost() {
      return negativeBoost;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("type", getType())
          .add("positive", positive)
          .add("negative", negative)
          .add("negativeBoost", negativeBoost)
          .toString();
    }
  }

  /** Boolean query composed of multiple clauses. */
  public static final class BooleanQuery extends FullTextQuery {
    private final List<BooleanClause> clauses;

    BooleanQuery(List<BooleanClause> clauses) {
      Preconditions.checkArgument(
          clauses != null && !clauses.isEmpty(), "clauses must not be null or empty");
      this.clauses =
          Collections.unmodifiableList(new java.util.ArrayList<>(Objects.requireNonNull(clauses)));
    }

    @Override
    public Type getType() {
      return Type.BOOLEAN;
    }

    public List<BooleanClause> getClauses() {
      return clauses;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("type", getType())
          .add("clauses", clauses)
          .toString();
    }
  }
}
