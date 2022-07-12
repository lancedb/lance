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
  auto status = lance::arrow::WriteTable(*coco, sink.ValueOrDie(), "filename");
  INFO("Write file: " << status);
  CHECK(status.ok());
  auto buf = sink.ValueOrDie()->Finish().ValueOrDie();

  auto infile = make_shared<arrow::io::BufferReader>(buf);
  INFO(FileReader::Make(infile).status());
  auto reader = FileReader::Make(infile).ValueOrDie();
  CHECK(reader->primary_key() == "filename");
  CHECK(reader->num_chunks() == 1);
  CHECK(reader->length() == 4);

  auto table = reader->ReadTable().ValueOrDie();
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
  CHECK(lance::arrow::WriteTable(*table, sink, "label").ok());

  auto infile = make_shared<arrow::io::BufferReader>(sink->Finish().ValueOrDie());
  INFO(FileReader::Make(infile).status());
  auto reader = FileReader::Make(infile).ValueOrDie();
  CHECK(reader->primary_key() == "label");
  CHECK(reader->num_chunks() == 1);
  CHECK(reader->length() == 4);

  auto actual_table = reader->ReadTable().ValueOrDie();
  CHECK(table->Equals(*actual_table));
}