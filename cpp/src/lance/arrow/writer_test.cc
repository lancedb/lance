//  Copyright 2022 Lance Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <arrow/builder.h>
#include <arrow/io/api.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>
#include <map>
#include <string>
#include <vector>

#include "lance/arrow/file_lance.h"
#include "lance/arrow/type.h"
#include "lance/format/schema.h"
#include "lance/io/reader.h"
#include "lance/testing/extension_types.h"
#include "lance/testing/io.h"

using arrow::ArrayBuilder;
using arrow::BooleanBuilder;
using arrow::Int32Builder;
using arrow::LargeBinaryBuilder;
using arrow::ListBuilder;
using arrow::StringBuilder;
using arrow::StructBuilder;
using arrow::Table;
using lance::format::Schema;
using lance::io::FileReader;

using std::make_shared;
using std::map;
using std::shared_ptr;
using std::string;
using std::vector;

/** Build coco dataset for testing. */
auto CocoDataset() {
  auto box_type = arrow::struct_(
      {arrow::field("xmin", arrow::float32()), arrow::field("ymin", arrow::float32())});
  auto annotationType = arrow::struct_({arrow::field("label", arrow::utf8()),
                                        arrow::field("score", arrow::float32()),
                                        arrow::field("box", box_type)});
  auto image_type = arrow::struct_({arrow::field("uri", arrow::utf8())});
  auto schema = arrow::schema({arrow::field("filename", arrow::utf8()),
                               arrow::field("split", arrow::utf8()),
                               arrow::field("width", arrow::int32()),
                               arrow::field("valid", arrow::boolean()),
                               arrow::field("image", image_type),
                               arrow::field("annotations", arrow::list(annotationType))});

  StringBuilder stringBuilder;
  Int32Builder intBuilder;
  BooleanBuilder booleanBuilder;

  CHECK(stringBuilder.AppendValues({"1.jpg", "2.jpg", "3.jpg", "4.jpg"}).ok());
  auto filenameArr = stringBuilder.Finish().ValueOrDie();
  stringBuilder.Reset();

  CHECK(stringBuilder.AppendValues({"train", "train", "split", "train"}).ok());
  auto split = stringBuilder.Finish().ValueOrDie();
  stringBuilder.Reset();

  CHECK(intBuilder.AppendValues({1, 2, 3, 4}).ok());
  auto width = intBuilder.Finish().ValueOrDie();
  intBuilder.Reset();

  CHECK(booleanBuilder.AppendValues(std::vector<bool>({true, true, false, true})).ok());
  auto valid = booleanBuilder.Finish().ValueOrDie();
  booleanBuilder.Reset();

  auto uriBuilder = std::make_shared<StringBuilder>();
  auto imageBuilder = std::make_shared<StructBuilder>(
      image_type, arrow::default_memory_pool(), vector<shared_ptr<ArrayBuilder>>({uriBuilder}));
  for (int i = 0; i < 4; i++) {
    CHECK(imageBuilder->Append().ok());
    CHECK(uriBuilder->Append(fmt::format("s3://{}", i)).ok());
  }
  auto images = imageBuilder->Finish().ValueOrDie();

  auto labelBuilder = std::make_shared<StringBuilder>();
  auto scoreBuilder = std::make_shared<arrow::FloatBuilder>();

  auto xminBuilder = std::make_shared<arrow::FloatBuilder>();
  auto yminBuilder = std::make_shared<arrow::FloatBuilder>();
  auto boxBuilder =
      std::make_shared<StructBuilder>(box_type,
                                      arrow::default_memory_pool(),
                                      vector<shared_ptr<ArrayBuilder>>({xminBuilder, yminBuilder}));
  auto structBuilder = std::make_shared<StructBuilder>(annotationType,
                                                       arrow::default_memory_pool(),
                                                       vector<shared_ptr<ArrayBuilder>>({
                                                           labelBuilder,
                                                           scoreBuilder,
                                                           boxBuilder,
                                                       }));
  ListBuilder listBuilder(arrow::default_memory_pool(), structBuilder, arrow::list(annotationType));
  for (int i = 0; i < 4; i++) {
    CHECK(listBuilder.Append().ok());
    CHECK(labelBuilder->AppendValues({"cat", "dog"}).ok());
    CHECK(scoreBuilder->AppendValues({static_cast<float>(0.5 + i), static_cast<float>(0.7 + i)})
              .ok());
    CHECK(xminBuilder->AppendValues({float(0.1 + i), float(0.2 + i)}).ok());
    CHECK(yminBuilder->AppendValues({float(0.1 + i), float(0.2 + i)}).ok());
    CHECK(boxBuilder->AppendValues(2, nullptr).ok());
    CHECK(structBuilder->AppendValues(2, nullptr).ok());
  }
  auto annotations = listBuilder.Finish().ValueOrDie();
  listBuilder.Reset();

  auto table = arrow::Table::Make(schema, {filenameArr, split, width, valid, images, annotations});
  return table;
}

std::shared_ptr<::arrow::Table> ReadTable(std::shared_ptr<arrow::io::BufferOutputStream> sink) {
  auto infile = make_shared<arrow::io::BufferReader>(sink->Finish().ValueOrDie());
  INFO(::lance::io::FileReader::Make(infile).status());
  auto reader = FileReader::Make(infile).ValueOrDie();
  return reader->ReadTable().ValueOrDie();
}

TEST_CASE("Write COCO Dataset") {
  auto coco = CocoDataset();
  auto reader = lance::testing::MakeReader(coco).ValueOrDie();
  CHECK(reader->num_batches() == 1);
  CHECK(reader->length() == 4);

  auto result = reader->ReadTable();
  INFO("ReadTable: " << result.status().message());
  CHECK(result.ok());
  auto table = result.ValueUnsafe();
  CHECK(coco->num_columns() == table->num_columns());

  INFO("expect " << coco->ToString() << "\nactual " << table->ToString());
  CHECK(coco->Equals(*table));

  auto row = reader->Get(2);
  CHECK(row.ok());
  auto anno_scalar = std::static_pointer_cast<::arrow::ListScalar>((*row)[row->size() - 1]);
  CHECK(anno_scalar->value->length() == 2);
}

TEST_CASE("Write dictionary type") {
  auto label_type = ::arrow::dictionary(arrow::int8(), arrow::utf8());

  ::arrow::DictionaryBuilder<arrow::StringType> builder;
  CHECK(builder.Append("cat").ok());
  CHECK(builder.Append("dog").ok());
  CHECK(builder.Append("cat").ok());
  CHECK(builder.Append("person").ok());

  auto arr = builder.Finish().ValueOrDie();
  INFO("array is " << arr->ToString());

  auto schema = arrow::schema({arrow::field("label", label_type)});
  auto table = arrow::Table::Make(schema, {arr});

  auto reader = lance::testing::MakeReader(table).ValueOrDie();
  CHECK(reader->num_batches() == 1);
  CHECK(reader->length() == 4);

  auto actual_table = reader->ReadTable().ValueOrDie();
  CHECK(table->Equals(*actual_table));
}

TEST_CASE("Large binary field") {
  auto field_type = ::arrow::large_binary();
  auto schema = ::arrow::schema({arrow::field("f1", field_type)});

  auto builder = ::arrow::LargeBinaryBuilder();
  for (int i = 0; i < 4; i++) {
    CHECK(builder.Append(fmt::format("{}", i)).ok());
  }
  auto arr = builder.Finish().ValueOrDie();
  auto table = ::arrow::Table::Make(std::move(schema), {arr});

  auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
  lance::testing::MakeReader(table).ValueOrDie();
}

TEST_CASE("Binary field") {
  auto field_type = ::arrow::binary();
  auto schema = ::arrow::schema({arrow::field("f1", field_type)});

  auto builder = ::arrow::BinaryBuilder();
  for (int i = 0; i < 4; i++) {
    CHECK(builder.Append(fmt::format("{}", i)).ok());
  }
  auto arr = builder.Finish().ValueOrDie();
  auto table = ::arrow::Table::Make(std::move(schema), {arr});

  CHECK(lance::testing::MakeReader(table).ok());
}

TEST_CASE("Write timestamp") {
  for (auto& type : {
           ::arrow::date32(),
           ::arrow::date64(),
           ::arrow::time32(::arrow::TimeUnit::SECOND),
           ::arrow::time64(::arrow::TimeUnit::NANO),
           ::arrow::timestamp(::arrow::TimeUnit::NANO),
           ::arrow::timestamp(::arrow::TimeUnit::SECOND),
       }) {
    auto schema = ::arrow::schema({arrow::field("ts", type)});
    auto builder = ::arrow::TimestampBuilder(type, ::arrow::default_memory_pool());
    for (int i = 0; i < 4; i++) {
      CHECK(builder.Append(i * 1000).ok());
    }
    auto arr = builder.Finish().ValueOrDie();
    auto table = ::arrow::Table::Make(std::move(schema), {arr});

    auto reader = lance::testing::MakeReader(table).ValueOrDie();

    auto actual_table = reader->ReadTable().ValueOrDie();
    INFO("Actual table: " << actual_table->ToString());
    INFO("Expected table: " << table->ToString());
    CHECK(table->Equals(*actual_table));
  }
}

TEST_CASE("Write with batch size") {
  auto options = lance::arrow::FileWriteOptions();
  options.batch_size = 5;  // use an odd number;
  auto int_builder = ::arrow::Int32Builder();
  for (int i = 0; i < 100; i++) {
    CHECK(int_builder.Append(i).ok());
  }
  auto arr = int_builder.Finish().ValueOrDie();
  auto table =
      ::arrow::Table::Make(::arrow::schema({::arrow::field("v", ::arrow::int32())}), {arr});
  auto reader = lance::testing::MakeReader(table, 5).ValueOrDie();

  auto actual_table = reader->ReadTable().ValueOrDie();
  auto batch_reader = ::arrow::TableBatchReader(actual_table);
  std::shared_ptr<::arrow::RecordBatch> batch;
  int batch_count = 0;
  while (true) {
    CHECK(batch_reader.ReadNext(&batch).ok());
    if (!batch) {
      break;
    }
    batch_count++;
    CHECK(batch->num_rows() == 5);
  }
  CHECK(batch_count == 20);
}

TEST_CASE("Write fixed size binary") {
  auto dtype = ::arrow::fixed_size_binary(10);
  auto builder = ::arrow::FixedSizeBinaryBuilder(dtype);
  for (int i = 0; i < 10; i++) {
    CHECK(builder.Append("1234567890").ok());
  }
  auto arr = builder.Finish().ValueOrDie();
  auto schema = ::arrow::schema({::arrow::field("bins", dtype)});
  auto t = ::arrow::Table::Make(schema, {arr});

  auto reader = lance::testing::MakeReader(t).ValueOrDie();
  auto actual = reader->ReadTable().ValueOrDie();
  CHECK(t->Equals(*actual));
}

TEST_CASE("Write fixed size list") {
  auto list_size = 4;
  auto dtype = ::arrow::fixed_size_list(::arrow::int32(), list_size);
  auto int_builder = std::make_shared<::arrow::Int32Builder>();
  auto builder = ::arrow::FixedSizeListBuilder(::arrow::default_memory_pool(), int_builder, dtype);

  for (int i = 0; i < 10; i++) {
    CHECK(builder.Append().ok());
    for (int j = 0; j < list_size; j++) {
      CHECK(int_builder->Append(i * list_size + j).ok());
    }
  }
  auto arr = builder.Finish().ValueOrDie();
  auto schema = ::arrow::schema({::arrow::field("list", dtype)});
  auto t = ::arrow::Table::Make(schema, {arr});

  auto reader = lance::testing::MakeReader(t).ValueOrDie();
  auto actual = reader->ReadTable().ValueOrDie();
  CHECK(t->Equals(*actual));
}

std::shared_ptr<::arrow::Table> MakeTable() {
  auto ext_type = std::make_shared<::lance::testing::ImageType>();
  auto uri_builder = std::make_shared<::arrow::StringBuilder>();
  auto data_builder = std::make_shared<::arrow::Int32Builder>();
  auto image_builder = std::make_shared<::arrow::StructBuilder>(
      ext_type->storage_type(),
      arrow::default_memory_pool(),
      std::vector<std::shared_ptr<::arrow::ArrayBuilder>>({uri_builder, data_builder}));

  auto box_type = std::make_shared<::lance::testing::Box2dType>();
  /// Value builders for xmin, ymin, xmax, ymax
  std::vector<std::shared_ptr<::arrow::ArrayBuilder>> coor_builders;
  for (int i = 0; i < 4; i++) {
    coor_builders.emplace_back(std::make_shared<::arrow::DoubleBuilder>());
  }
  auto box_builder = std::make_shared<::arrow::StructBuilder>(
      box_type->storage_type(), ::arrow::default_memory_pool(), coor_builders);
  auto boxes_list_type = std::make_shared<::arrow::ListType>(box_type);
  auto boxes_builder =
      std::make_shared<::arrow::ListBuilder>(::arrow::default_memory_pool(), box_builder);
  auto annotations_type = ::arrow::struct_({::arrow::field("boxes", boxes_list_type)});
  auto annotations_builder = std::make_shared<::arrow::StructBuilder>(
      annotations_type,
      ::arrow::default_memory_pool(),
      std::vector<std::shared_ptr<::arrow::ArrayBuilder>>({boxes_builder}));

  for (int i = 0; i < 4; i++) {
    CHECK(image_builder->Append().ok());
    CHECK(uri_builder->Append(fmt::format("s3://{}", i)).ok());
    CHECK(data_builder->Append(i).ok());

    CHECK(annotations_builder->Append().ok());
    CHECK(boxes_builder->Append().ok());
    // Add five boxes per image
    for (int j = 0; j < 5; j++) {
      CHECK(box_builder->Append().ok());
      for (auto& cood_builder : coor_builders) {
        CHECK(std::dynamic_pointer_cast<::arrow::DoubleBuilder>(cood_builder)->Append(0.5).ok());
      }
    }
  }
  auto image_arr = image_builder->Finish().ValueOrDie();
  INFO("array is " << image_arr->ToString());

  auto ann_arr =
      std::dynamic_pointer_cast<::arrow::StructArray>(annotations_builder->Finish().ValueOrDie());
  auto list_arr = std::dynamic_pointer_cast<::arrow::ListArray>(ann_arr->GetFieldByName("boxes"));

  auto ext_box_arr =
      ::arrow::ListArray::FromArrays(
          *list_arr->offsets(), *::arrow::ExtensionType::WrapArray(box_type, list_arr->values()))
          .ValueOrDie();
  ann_arr = ::arrow::StructArray::Make({ext_box_arr}, {"boxes"}).ValueOrDie();

  auto schema = ::arrow::schema(
      {arrow::field("image_ext", ext_type), ::arrow::field("annotations", annotations_type)});
  std::vector<std::shared_ptr<::arrow::Array>> cols;
  cols.emplace_back(::arrow::ExtensionType::WrapArray(ext_type, image_arr));
  cols.emplace_back(ann_arr);
  return ::arrow::Table::Make(std::move(schema), std::move(cols));
}

TEST_CASE("Write extension but read storage if not registered") {
  auto table = MakeTable();
  auto arr = std::static_pointer_cast<::arrow::ExtensionArray>(
                 table->GetColumnByName("image_ext")->chunk(0))
                 ->storage();
  auto reader = lance::testing::MakeReader(table).ValueOrDie();

  // We can read it back without the extension
  auto actual_table = reader->ReadTable().ValueOrDie();
  CHECK(arr->Equals(actual_table->GetColumnByName("image_ext")->chunk(0)));
  auto lance_schema = Schema(actual_table->schema());
  auto image_field = lance_schema.GetField("image_ext");
  CHECK(image_field->logical_type() == "struct");
  CHECK(image_field->extension_name() == "");
  CHECK(!(image_field->is_extension_type()));
  CHECK(lance_schema.GetFieldsCount() == 10);
}

TEST_CASE("Extension type round-trip") {
  auto ext_type = std::make_shared<::lance::testing::ImageType>();
  CHECK(arrow::RegisterExtensionType(ext_type).ok());
  auto box_type = std::make_shared<::lance::testing::Box2dType>();
  CHECK(arrow::RegisterExtensionType(box_type).ok());
  auto table = MakeTable();

  auto reader = lance::testing::MakeReader(table).ValueOrDie();
  // We can read it back without the extension
  auto actual_table = reader->ReadTable().ValueOrDie();

  INFO("Actual table: " << actual_table->ToString() << "\nExpected table: " << table->ToString());
  CHECK(table->Equals(*actual_table));

  CHECK(actual_table->GetColumnByName("annotations")
            ->type()
            ->Equals(::arrow::struct_({::arrow::field("boxes", ::arrow::list(box_type))})));
  CHECK(actual_table->GetColumnByName("image_ext")->type()->Equals(ext_type));
}

TEST_CASE("Write empty arrays") {
  auto int_builder = std::make_shared<::arrow::Int32Builder>();
  auto boolean_builder = std::make_shared<::arrow::BooleanBuilder>();
  auto fixed_size_builder = std::make_shared<::arrow::FixedSizeListBuilder>(
      ::arrow::default_memory_pool(), std::make_shared<::arrow::FloatBuilder>(), 4);

  auto struct_field = ::arrow::struct_({
      ::arrow::field("ints", ::arrow::int32()),
      ::arrow::field("bools", ::arrow::boolean()),
      ::arrow::field("fixed_size_list", ::arrow::fixed_size_list(::arrow::float32(), 4)),
  });
  auto struct_builder = std::make_shared<::arrow::StructBuilder>(
      struct_field,
      ::arrow::default_memory_pool(),
      std::vector<std::shared_ptr<::arrow::ArrayBuilder>>(
          {int_builder, boolean_builder, fixed_size_builder}));
  auto list_builder = ::arrow::ListBuilder(::arrow::default_memory_pool(), struct_builder);

  CHECK(list_builder.AppendNulls(5).ok());
  auto arr = list_builder.Finish().ValueOrDie();
  auto schema = ::arrow::schema({::arrow::field("values", ::arrow::list(struct_field))});
  auto table = ::arrow::Table::Make(schema, {arr});

  auto reader = lance::testing::MakeReader(table).ValueOrDie();
  auto actual_table = reader->ReadTable().ValueOrDie();
  CHECK(table->Equals(*actual_table));
}
