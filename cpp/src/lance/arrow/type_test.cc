#include "lance/arrow/type.h"

#include <arrow/type.h>
#include <fmt/format.h>
#include <google/protobuf/util/message_differencer.h>

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <string>
#include <vector>

#include "lance/format/format.pb.h"

using google::protobuf::util::MessageDifferencer;
using std::string;
using std::vector;

namespace pb = lance::format::pb;
using lance::arrow::FromArrowSchema;

TEST_CASE("To arrow schema") {
  // COCO dataset
  auto annotationType = arrow::struct_(
      {arrow::field("label", arrow::utf8()), arrow::field("score", arrow::float32())});
  auto schema =
      arrow::schema({arrow::field("filename", arrow::utf8()),
                     arrow::field("split", arrow::utf8()),
                     arrow::field("width", arrow::int32()),
                     arrow::field("image", arrow::struct_({arrow::field("uri", arrow::utf8())})),
                     arrow::field("annotations", arrow::list(annotationType))});

  auto result = FromArrowSchema(schema);
  CHECK(result.ok());
  CHECK(result->size() == 9);

  auto actual_schema = lance::arrow::ToArrowSchema(result.ValueOrDie());
  CHECK(actual_schema.ok());
  INFO("Expect schema: " << schema->ToString() << "\n, Actual: " << (*actual_schema)->ToString());
  CHECK((*actual_schema)->Equals(schema));
};

TEST_CASE("Arrow Schema (simple)") {
  auto schema =
      arrow::schema({arrow::field("pk", arrow::utf8()), arrow::field("value", arrow::int64())});

  auto fields = lance::arrow::FromArrowSchema(schema).ValueOrDie();
  CHECK(fields.size() == 2);

  pb::Field expected;
  expected.set_id(0);
  expected.set_type(pb::Field::LEAF);
  expected.set_data_type(pb::BYTES);
  expected.set_logical_type("string");
  expected.set_name("pk");
  expected.set_parent_id(-1);
  expected.set_encoding(pb::Encoding::VAR_BINARY);

  CHECK(MessageDifferencer::Equals(expected, fields.at(0)));

  expected.set_id(1);
  expected.set_type(pb::Field::LEAF);
  expected.set_data_type(pb::INT64);
  expected.set_logical_type("int64");
  expected.set_name("value");
  expected.set_parent_id(-1);
  expected.set_encoding(pb::Encoding::PLAIN);

  INFO("Actual fields: " << fields[1].DebugString() << " expect: " << expected.DebugString());
  CHECK(MessageDifferencer::Equals(expected, fields.at(1)));

  auto actual_schema = lance::arrow::ToArrowSchema(fields).ValueOrDie();
  CHECK(actual_schema->Equals(schema));
}

TEST_CASE("List schema") {
  auto schema = arrow::schema({arrow::field("ints", arrow::list(arrow::int32()))});

  //  auto fields = lance::arrow::FromArrowSchema(schema).ValueOrDie();
  auto result = lance::arrow::FromArrowSchema(schema);
  INFO("Convert to fields: " << result.status());
  CHECK(result.ok());

  auto actual_schema = lance::arrow::ToArrowSchema(*result);
  CHECK(result.ok());
  CHECK((*actual_schema)->Equals(schema));
}
