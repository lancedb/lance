#include "lance/format/lookup_table.h"

#include <arrow/io/api.h>

#include <catch2/catch_test_macros.hpp>

#include "lance/format/format.pb.h"

using lance::format::LookupTable;

TEST_CASE("Serialize Chunk length") {
  lance::format::LookupTable lt;

  int num_columns = 3;
  int num_chunks = 5;
  for (int col = 0; col < num_columns; col++) {
    for (int chk = 0; chk < num_chunks; chk++) {
      lt.AddOffset(col, chk, col * 10 + chk);
      lt.AddPageLength(col, chk, col * 10 + chk);
    }
  }
  lance::format::pb::Metadata metadata;

  auto out_buf = arrow::io::BufferOutputStream::Create().ValueOrDie();
  lt.Write(out_buf).ValueOrDie();
  lt.WritePageLengthTo(&metadata);

  auto in_buf = std::make_shared<arrow::io::BufferReader>(out_buf->Finish().ValueOrDie());
  auto actual = LookupTable::Read(in_buf, 0, metadata).ValueOrDie();

  for (int col = 0; col < num_columns; col++) {
    for (int chk = 0; chk < num_chunks; chk++) {
      INFO("Getting offset col=" << col << " chk=" << chk);
      CHECK(actual->GetOffset(col, chk) == col * 10 + chk);
      CHECK(actual->GetPageLength(col, chk).ValueOrDie() == col * 10 + chk);
    }
  }
}