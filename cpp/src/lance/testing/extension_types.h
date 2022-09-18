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

#include <arrow/extension_type.h>
#include <arrow/type.h>

#include <string>

#include "fmt/format.h"

namespace lance {
namespace testing {

class ImageType : public ::arrow::ExtensionType {
 public:
  ImageType()
      : ::arrow::ExtensionType(::arrow::struct_({
            ::arrow::field("uri", ::arrow::utf8()),
            ::arrow::field("data", ::arrow::int32()),
        })) {}

  std::string extension_name() const override { return "image"; }

  bool ExtensionEquals(const ::arrow::ExtensionType& other) const override {
    return other.extension_name() == extension_name();
  }

  std::shared_ptr<::arrow::Array> MakeArray(
      std::shared_ptr<::arrow::ArrayData> data) const override {
    return std::make_shared<::arrow::ExtensionArray>(data);
  }

  ::arrow::Result<std::shared_ptr<::arrow::DataType>> Deserialize(
      [[maybe_unused]] std::shared_ptr<::arrow::DataType> storage_type,
      const std::string& serialized) const override {
    if (serialized != "ext-struct-type-unique-code") {
      return ::arrow::Status::Invalid("Type identifier did not match");
    }
    return std::make_shared<ImageType>();
  }

  std::string Serialize() const override { return "image-ext"; }
};

class Box2dType : public ::arrow::ExtensionType {
 public:
  Box2dType()
      : ::arrow::ExtensionType(::arrow::struct_({
            ::arrow::field("xmin", ::arrow::float64()),
            ::arrow::field("ymin", ::arrow::float64()),
            ::arrow::field("xmax", ::arrow::float64()),
            ::arrow::field("ymax", ::arrow::float64()),
        })){};

  std::string extension_name() const override { return "box2d"; }

  bool ExtensionEquals(const ::arrow::ExtensionType& other) const override {
    return other.extension_name() == extension_name();
  }

  std::shared_ptr<::arrow::Array> MakeArray(
      std::shared_ptr<::arrow::ArrayData> data) const override {
    return std::make_shared<::arrow::ExtensionArray>(data);
  }

  ::arrow::Result<std::shared_ptr<::arrow::DataType>> Deserialize(
      [[maybe_unused]] std::shared_ptr<::arrow::DataType> storage_type,
      const std::string& serialized) const override {
    if (serialized != "ext-struct-type-unique-code") {
      return ::arrow::Status::Invalid("Type identifier did not match");
    }
    return std::make_shared<ImageType>();
  }

  std::string Serialize() const override { return "box2d-ext"; }
};

// A parametric type where the extension_name() is always the same
class ParametricType : public ::arrow::ExtensionType {
 public:
  explicit ParametricType(int32_t parameter)
      : ::arrow::ExtensionType(::arrow::int32()), parameter_(parameter) {}

  int32_t parameter() const { return parameter_; }

  std::string extension_name() const override { return "parametric-type"; }

  bool ExtensionEquals(const ::arrow::ExtensionType& other) const override {
    const auto& other_ext = static_cast<const ::arrow::ExtensionType&>(other);
    if (other_ext.extension_name() != this->extension_name()) {
      return false;
    }
    return this->parameter() == static_cast<const ParametricType&>(other).parameter();
  }

  std::shared_ptr<::arrow::Array> MakeArray(
      std::shared_ptr<::arrow::ArrayData> data) const override {
    return std::make_shared<::arrow::ExtensionArray>(data);
  }

  ::arrow::Result<std::shared_ptr<::arrow::DataType>> Deserialize(
      [[maybe_unused]] std::shared_ptr<::arrow::DataType> storage_type,
      const std::string& serialized) const override {
    const int32_t parameter = *reinterpret_cast<const int32_t*>(serialized.data());
    return std::make_shared<ParametricType>(parameter);
  }

  std::string Serialize() const override {
    std::string result("    ");
    memcpy(&result[0], &parameter_, sizeof(int32_t));
    return result;
  }

 private:
  int32_t parameter_;
};

class AnnotationType : public ::arrow::ExtensionType {
 public:
  AnnotationType()
      : ::arrow::ExtensionType(::arrow::struct_({
            ::arrow::field("class", ::arrow::int32()),
            ::arrow::field("box", std::make_shared<Box2dType>()),
        })){};

  std::string extension_name() const override { return "annotation"; }

  bool ExtensionEquals(const ::arrow::ExtensionType& other) const override {
    return other.extension_name() == extension_name();
  }

  std::shared_ptr<::arrow::Array> MakeArray(
      std::shared_ptr<::arrow::ArrayData> data) const override {
    return std::make_shared<::arrow::ExtensionArray>(data);
  }

  ::arrow::Result<std::shared_ptr<::arrow::DataType>> Deserialize(
      [[maybe_unused]] std::shared_ptr<::arrow::DataType> storage_type,
      const std::string& serialized) const override {
    if (serialized != "ext-struct-type-unique-code") {
      return ::arrow::Status::Invalid("Type identifier did not match");
    }
    return std::make_shared<ImageType>();
  }

  std::string Serialize() const override { return "annotations-ext"; }
};

}  // namespace testing
}  // namespace lance