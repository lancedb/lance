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

#pragma once

#include <arrow/status.h>
#include <arrow/type_fwd.h>

#include <memory>

namespace lance::format {

/// Forward Declaration.
class Field;

/// Visitor over Field.
class FieldVisitor {
 public:
  virtual ::arrow::Status Visit(std::shared_ptr<Field> field) = 0;
};

/// A Visitor to convert Field / Schema to an arrow::Schema
class ToArrowVisitor : public FieldVisitor {
 public:
  ::arrow::Status Visit(std::shared_ptr<Field> root) override;

  std::shared_ptr<::arrow::Schema> Finish();

 private:
  ::arrow::Result<::std::shared_ptr<::arrow::Field>> DoVisit(std::shared_ptr<Field> node);

  std::vector<::std::shared_ptr<::arrow::Field>> arrow_fields_;
};

}  // namespace lance::format