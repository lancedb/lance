//  Copyright 2022 Lance Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "lance/format/schema.h"

#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>
#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>

#include "lance/arrow/stl.h"
#include "lance/testing/extension_types.h"
#include "lance/testing/io.h"
#include "lance/testing/json.h"

using lance::arrow::ToArray;
using lance::testing::MakeDataset;
using lance::testing::TableFromJSON;

const auto arrow_schema = ::arrow::schema(
    {::arrow::field("pk", ::arrow::utf8()),
     ::arrow::field("split", ::arrow::utf8()),
     ::arrow::field("annotations",
                    ::arrow::list(::arrow::struct_(
                        {::arrow::field("label", ::arrow::utf8()),
                         ::arrow::field("box",
                                        ::arrow::struct_({
                                            ::arrow::field("xmin", ::arrow::float32()),
                                            ::arrow::field("ymin", ::arrow::float32()),
                                            ::arrow::field("xmax", ::arrow::float32()),
                                            ::arrow::field("ymax", ::arrow::float32()),
                                        }))})))});

std::shared_ptr<::arrow::DataType> image_type = std::make_shared<::lance::testing::ImageType>();
const auto ext_schema =
    ::arrow::schema({::arrow::field("pk", ::arrow::utf8()), ::arrow::field("image", image_type)});

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

TEST_CASE("Project nested fields") {
  auto original = lance::format::Schema(arrow_schema);
  auto projection = original.Project({"annotations.box.xmin"}).ValueOrDie();

  auto expected_schema = ::arrow::schema({::arrow::field(
      "annotations",
      ::arrow::list(::arrow::struct_({::arrow::field(
          "box", ::arrow::struct_({::arrow::field("xmin", ::arrow::float32())}))})))});
  auto expected = lance::format::Schema(expected_schema);
  INFO("Expected: " << expected.ToString());
  INFO("Actual: " << projection->ToString());
  CHECK(projection->Equals(expected, false));
}

TEST_CASE("Get schema view") {
  auto original = lance::format::Schema(arrow_schema);
  auto view = original.Project({"split", "annotations.box.xmin"});
  INFO("Create view status: " << view.status());
  CHECK(view.ok());
  CHECK((*view)->GetField("split"));
  INFO("Annotations xmin: " << (*view)->GetField("annotations.box.xmin"));
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
       ::arrow::field(
           "annotations",
           ::arrow::list(::arrow::struct_({::arrow::field("label", ::arrow::utf8())})))});
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
  auto excluded = original.Exclude(*projected).ValueOrDie();

  auto excluded_arrow_schema = ::arrow::schema(
      {::arrow::field("pk", ::arrow::utf8()),
       ::arrow::field(
           "annotations",
           ::arrow::list(::arrow::struct_({::arrow::field("label", ::arrow::utf8())})))});
  auto expected = original.Project(*excluded_arrow_schema).ValueOrDie();

  INFO("Expected: " << expected->ToString() << "\nActual: " << excluded->ToString());
  CHECK(excluded->Equals(expected));
}

TEST_CASE("Get Field Counts") {
  auto schema = lance::format::Schema(arrow_schema);
  CHECK(schema.GetFieldsCount() == 10);
}

TEST_CASE("Get Field Counts extension") {
  auto schema = lance::format::Schema(ext_schema);
  CHECK(schema.GetFieldsCount() == 4);
}

TEST_CASE("Project nested extension type") {
  auto original = lance::format::Schema(ext_schema);
  auto projection = original.Project({"image.uri"}).ValueOrDie();

  auto expected_schema = ::arrow::schema(
      {::arrow::field("image", ::arrow::struct_({::arrow::field("uri", ::arrow::utf8())}))});
  auto expected = lance::format::Schema(expected_schema);
  INFO("Expected: " << expected.ToString());
  INFO("Actual: " << projection->ToString());
  CHECK(projection->Equals(expected, false));
}

TEST_CASE("Fixed size list") {
  auto arrow_field =
      ::arrow::field("fixed_size_list", ::arrow::fixed_size_list(::arrow::int32(), 4));
  auto field = ::lance::format::Field(arrow_field);
  CHECK(field.encoding() == ::lance::encodings::PLAIN);
  CHECK(field.logical_type() == "fixed_size_list:int32:4");
}

TEST_CASE("Fixed size binary") {
  auto arrow_field = ::arrow::field("fs_binary", ::arrow::fixed_size_binary(100));
  auto field = ::lance::format::Field(arrow_field);
  CHECK(field.encoding() == ::lance::encodings::PLAIN);
  CHECK(field.logical_type() == "fixed_size_binary:100");
}

TEST_CASE("Test storage type") {
  auto image_type = std::make_shared<lance::testing::ImageType>();
  CHECK(::arrow::RegisterExtensionType(image_type).ok());
  auto arrow_field = ::arrow::field("image", image_type);
  auto field = ::lance::format::Field(arrow_field);

  CHECK(field.is_extension_type());
  INFO("Field data type: " << field.type()->ToString());
  CHECK(field.type()->Equals(image_type));
  INFO("Field storage type: " << field.storage_type()->ToString());
  CHECK(field.storage_type()->Equals(image_type->storage_type()));
}

TEST_CASE("Test nested storage type") {
  auto annotation_type = std::make_shared<lance::testing::AnnotationType>();
  CHECK(::arrow::RegisterExtensionType(annotation_type).ok());
  auto arrow_field = ::arrow::field("annotation", annotation_type);
  auto field = ::lance::format::Field(arrow_field);

  CHECK(field.is_extension_type());
  INFO("Field data type: " << field.type()->ToString());
  CHECK(field.type()->Equals(annotation_type));
  INFO("Field storage type: " << field.storage_type()->ToString());
  CHECK(field.storage_type()->Equals(::arrow::struct_({
      ::arrow::field("class", ::arrow::int32()),
      ::arrow::field("box",
                     ::arrow::struct_({
                         ::arrow::field("xmin", ::arrow::float64()),
                         ::arrow::field("ymin", ::arrow::float64()),
                         ::arrow::field("xmax", ::arrow::float64()),
                         ::arrow::field("ymax", ::arrow::float64()),
                     })),
  })));
}

TEST_CASE("Test schema metadata") {
  auto schema = ::arrow::schema({::arrow::field("val", ::arrow::int32())},
                                ::arrow::KeyValueMetadata::Make({"k1", "k2"}, {"v1", "v2"}));

  auto table = ::arrow::Table::Make(schema, {ToArray({1, 2, 3}).ValueOrDie()});

  auto dataset = MakeDataset(table).ValueOrDie();
  CHECK(dataset->schema()->metadata());
  CHECK(dataset->schema()->metadata()->Get("k1").ValueOrDie() == "v1");
  CHECK(dataset->schema()->metadata()->Get("k1").ValueOrDie() == "v1");
}