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

#include "lance/arrow/writer.h"

#include <arrow/builder.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <catch2/catch_test_macros.hpp>
#include <map>
#include <string>
#include <vector>

#include "lance/arrow/reader.h"
#include "lance/arrow/type.h"
#include "lance/io/reader.h"

using arrow::ArrayBuilder;
using arrow::Int32Builder;
using arrow::ListBuilder;
using arrow::StringBuilder;
using arrow::StructBuilder;
using arrow::Table;
using lance::arrow::FileReader;

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
                               arrow::field("image", image_type),
                               arrow::field("annotations", arrow::list(annotationType))});

  StringBuilder stringBuilder;
  Int32Builder intBuilder;

  CHECK(stringBuilder.AppendValues({"1.jpg", "2.jpg", "3.jpg", "4.jpg"}).ok());
  auto filenameArr = stringBuilder.Finish().ValueOrDie();
  stringBuilder.Reset();

  CHECK(stringBuilder.AppendValues({"train", "train", "split", "train"}).ok());
  auto split = stringBuilder.Finish().ValueOrDie();
  stringBuilder.Reset();

  CHECK(intBuilder.AppendValues({1, 2, 3, 4}).ok());
  auto width = intBuilder.Finish().ValueOrDie();
  intBuilder.Reset();

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

  auto table = arrow::Table::Make(schema, {filenameArr, split, width, images, annotations});
  return table;
}

TEST_CASE("Write COCO Dataset") {
  auto coco = CocoDataset();
  auto sink = arrow::io::BufferOutputStream::Create();
  auto status = lance::arrow::WriteTable(*coco, sink.ValueOrDie());
  INFO("Write file: " << status);
  CHECK(status.ok());
  auto buf = sink.ValueOrDie()->Finish().ValueOrDie();

  auto infile = make_shared<arrow::io::BufferReader>(buf);
  INFO(FileReader::Make(infile).status());
  auto reader = FileReader::Make(infile).ValueOrDie();
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

  auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
  CHECK(lance::arrow::WriteTable(*table, sink).ok());

  auto infile = make_shared<arrow::io::BufferReader>(sink->Finish().ValueOrDie());
  INFO(FileReader::Make(infile).status());
  auto reader = FileReader::Make(infile).ValueOrDie();
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
  auto result = lance::arrow::WriteTable(*table, sink);
  INFO("Write table: " << result.message());
  CHECK(result.ok());
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

  auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
  CHECK(lance::arrow::WriteTable(*table, sink).ok());
}


class ImageType : public ::arrow::ExtensionType {
 public:
  ImageType() : ExtensionType(::arrow::struct_({
                    ::arrow::field("uri", ::arrow::utf8()),
                })) {}

  std::string extension_name() const override { return "image"; }

  bool ExtensionEquals(const ExtensionType& other) const override {
    return other.extension_name() != extension_name();
  }

  std::shared_ptr<::arrow::Array> MakeArray(std::shared_ptr<::arrow::ArrayData> data)
      const override {
    return std::make_shared<::arrow::ExtensionArray>(data);
  }

  ::arrow::Result<std::shared_ptr<DataType>> Deserialize(
      std::shared_ptr<DataType> storage_type,
      const std::string& serialized) const override {
    if (serialized != "ext-struct-type-unique-code") {
      return ::arrow::Status::Invalid("Type identifier did not match");
    }
    return std::make_shared<ImageType>();
  }
  std::string Serialize() const override { return "image-ext"; }
};


std::shared_ptr<::arrow::Table> MakeTable() {
  auto ext_type = std::make_shared<ImageType>();
  auto uriBuilder = std::make_shared<StringBuilder>();
  auto imageBuilder = std::make_shared<StructBuilder>(
      ext_type->storage_type(),
      arrow::default_memory_pool(),
      vector<shared_ptr<ArrayBuilder>>({uriBuilder}));
  for (int i = 0; i < 4; i++) {
    CHECK(imageBuilder->Append().ok());
    CHECK(uriBuilder->Append(fmt::format("s3://{}", i)).ok());
  }
  auto arr = imageBuilder->Finish().ValueOrDie();
  INFO("array is " << arr->ToString());

  auto schema = ::arrow::schema({arrow::field("image", ext_type)});
  std::vector<std::shared_ptr<::arrow::Array>> cols;
  cols.push_back(::arrow::ExtensionType::WrapArray(ext_type, arr));
  return ::arrow::Table::Make(std::move(schema), std::move(cols));
}

std::shared_ptr<::arrow::Table> ReadTable(std::shared_ptr<arrow::io::BufferOutputStream> sink) {
  auto infile = make_shared<arrow::io::BufferReader>(sink->Finish().ValueOrDie());
  INFO(FileReader::Make(infile).status());
  auto reader = FileReader::Make(infile).ValueOrDie();
  CHECK(reader->num_batches() == 1);
  CHECK(reader->length() == 4);
  return reader->ReadTable().ValueOrDie();
}

TEST_CASE("Extension type round-trip unregistered") {
  auto table = MakeTable();
  auto arr = std::static_pointer_cast<::arrow::ExtensionArray>(
      table->GetColumnByName("image")->chunk(0))->storage();

  auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
  CHECK(lance::arrow::WriteTable(*table, sink).ok());

  // We can read it back without the extension
  auto actual_table = ReadTable(sink);
  CHECK(arr->Equals(actual_table->GetColumnByName("image")->chunk(0)));
}
