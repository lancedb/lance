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
package com.lancedb.lance.schema;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents a list of SQL expressions. Each expression has a name and an expression string. Name:
 * is used to refer to the new column name. Expression: SQL expression strings. These strings can
 * reference existing columns in the dataset. The expression would be calculated as the value of new
 * column.
 */
public class SqlExpressions {

  private final List<SqlExpression> sqlExpressions;

  private SqlExpressions(List<SqlExpression> sqlExpressions) {
    this.sqlExpressions = sqlExpressions;
  }

  public List<SqlExpression> getSqlExpressions() {
    return sqlExpressions;
  }

  public static class SqlExpression {

    private String name;
    private String expression;

    public SqlExpression() {}

    public String getName() {
      return name;
    }

    public void setName(String name) {
      this.name = name;
    }

    public String getExpression() {
      return expression;
    }

    public void setExpression(String expression) {
      this.expression = expression;
    }
  }

  public static class Builder {

    private final SqlExpressions sqlExpressions;

    public Builder() {
      this.sqlExpressions = new SqlExpressions(new ArrayList<>());
    }

    public Builder withExpression(String name, String expr) {
      SqlExpression expression = new SqlExpression();
      expression.setName(name);
      expression.setExpression(expr);
      this.sqlExpressions.getSqlExpressions().add(expression);
      return this;
    }

    public SqlExpressions build() {
      return this.sqlExpressions;
    }
  }
}
