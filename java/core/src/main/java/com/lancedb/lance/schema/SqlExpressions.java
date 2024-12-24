package com.lancedb.lance.schema;

import java.util.ArrayList;
import java.util.List;

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
