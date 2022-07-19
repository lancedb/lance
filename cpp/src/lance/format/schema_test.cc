#include "lance/format/schema.h"

#include <arrow/type.h>
#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>

const auto arrow_schema = ::arrow::schema(
    {::arrow::field("pk", ::arrow::utf8()),
     ::arrow::field("split", ::arrow::utf8()),
     ::arrow::field("annotations",
                    ::arrow::struct_({::arrow::field("label", ::arrow::utf8()),
                                      ::arrow::field("box",
                                                     ::arrow::struct_({
                                                         ::arrow::field("xmin", ::arrow::float32()),
                                                         ::arrow::field("ymin", ::arrow::float32()),
                                                         ::arrow::field("xmax", ::arrow::float32()),
                                                         ::arrow::field("ymax", ::arrow::float32()),
                                                     }))}))});

TEST_CASE("Get field by name") {
  auto schema = lance::format::Schema(arrow_schema);

  auto field = schema.GetField("pk");
  CHECK(field);
  CHECK(field->name() == "pk");
  CHECK(field->type()->Equals(::arrow::utf8()));

  field = schema.GetField("annotations.box");
  CHECK(field->name() == "box");
  CHECK(field->type()->Equals(::arrow::struct_({::arrow::field("xmin", ::arrow::float32()),
                                                ::arrow::field("ymin", ::arrow::float32()),
                                                ::arrow::field("xmax", ::arrow::float32()),
                                                ::arrow::field("ymax", ::arrow::float32())})));

  field = schema.GetField("non.exist.path");
  CHECK(!field);
}

TEST_CASE("Schema equal") {
  auto schema1 = lance::format::Schema(arrow_schema);
  auto schema2 = lance::format::Schema(arrow_schema);

  INFO("Schema 1 == " << schema1.ToString() << "\nschema 2 = " << schema2.ToString());
  CHECK(schema1 == schema2);
}

TEST_CASE("Get schema view") {
  auto original = lance::format::Schema(arrow_schema);
  auto view = original.Project({"split", "annotations.box.xmin"});
  INFO("Create view status: " << view.status());
  CHECK(view.ok());
  CHECK((*view)->GetField("split"));
  CHECK((*view)->GetField("annotations.box.xmin"));
  CHECK(!(*view)->GetField("pk"));
  CHECK(!(*view)->GetField("annotations.label"));

  view = original.Project({"annotations.label"});
  CHECK(view.ok());
}

TEST_CASE("Get projection via arrow schema") {
  auto schema = lance::format::Schema(arrow_schema);
  auto projected_schema = ::arrow::schema(
      {::arrow::field("pk", ::arrow::utf8()),
       ::arrow::field("annotations",
                      ::arrow::struct_({::arrow::field("label", ::arrow::utf8())}))});
  auto projection = schema.Project(*projected_schema).ValueOrDie();

  auto expect_schema = lance::format::Schema(projected_schema);
  INFO("Expect schema: " << expect_schema.ToString()
                         << "\n Actual schema: " << projection->ToString());
  CHECK(expect_schema.Equals(projection, false));
}

TEST_CASE("Exclude schema") {
  auto original = lance::format::Schema(arrow_schema);
  auto projected = original.Project({"split", "annotations.box"}).ValueOrDie();
  INFO("Projected schema: " << projected->ToString());
  auto excluded = original.Exclude(projected).ValueOrDie();

  auto excluded_arrow_schema = ::arrow::schema(
      {::arrow::field("pk", ::arrow::utf8()),
       ::arrow::field("annotations",
                      ::arrow::struct_({::arrow::field("label", ::arrow::utf8())}))});
  auto expected = original.Project(*excluded_arrow_schema).ValueOrDie();

  INFO("Expected: " << expected->ToString() << "\nActual: " << excluded->ToString());
  CHECK(excluded->Equals(expected));
}