package com.lancedb.lance.schema;

import java.util.List;

public class SqlExpressions {

  private List<SqlExpression> sqlExpressions;

  public SqlExpressions() {}

  public List<SqlExpression> getSqlExpressions() {
    return sqlExpressions;
  }

  public void setSqlExpressions(List<SqlExpression> sqlExpressions) {
    this.sqlExpressions = sqlExpressions;
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
}
