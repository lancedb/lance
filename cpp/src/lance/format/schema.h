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

#include <arrow/compute/api.h>
#include <arrow/type.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "lance/encodings/encoder.h"
#include "lance/format/format.pb.h"
#include "lance/format/visitors.h"

namespace lance::format {

class Field;

/// Schema is a tree representation of on-disk columns.
///
class Schema final {
 public:
  /// Field ID type.
  using FieldIdType = int32_t;

  Schema() = default;

  /// Construct Lance Schema from Arrow Schema.
  explicit Schema(const std::shared_ptr<::arrow::Schema>& schema);

  /// Construct Lance Schema from Protobuf.
  ///
  /// \param pb_fields the fields described in protobuf.
  /// \param metadata the metadata pairs.
  explicit Schema(const google::protobuf::RepeatedPtrField<::lance::format::pb::Field>& pb_fields,
                  const google::protobuf::Map<std::string, std::string>& metadata = {});

  ~Schema() = default;

  /// Convert to arrow schema.
  std::shared_ptr<::arrow::Schema> ToArrow() const;

  /// Serialize Schema into a list of protobuf Fields.
  std::vector<lance::format::pb::Field> ToProto() const;

  /// Create a Projection of this schema that only contains the specified columns.
  ///
  /// \param column_names a list of fully qualified column names.
  /// \return a view of schema that only contains the specified columns. Returns failures if such
  ///         view can not be built.
  [[nodiscard]] ::arrow::Result<std::shared_ptr<Schema>> Project(
      const std::vector<std::string>& column_names) const;

  /// Use `arrow::Schema` to create a project over the current Lance schema.
  ::arrow::Result<std::shared_ptr<Schema>> Project(const ::arrow::Schema& arrow_schema) const;

  /// Use `arrow::compute::Expression` to create a projected schema.
  ///
  /// \param expr the expression to compute projection.
  /// \return the projected schema. Or nullptr if the expression does not contain any field.
  ::arrow::Result<std::shared_ptr<Schema>> Project(const ::arrow::compute::Expression& expr) const;

  /// Use a vector of field ids to create a project schema.
  ///
  /// \param field_ids a vector of field IDs
  /// \return the view of schema that only contains the column specified by the field IDs.
  ::arrow::Result<std::shared_ptr<Schema>> Project(const std::vector<FieldIdType>& field_ids) const;

  /// Exclude (subtract) the fields from the given schema.
  ///
  /// \param other the schema to be excluded. It must to be a strict subset of this schema.
  /// \return The newly created schema, excluding any column in "other".
  ::arrow::Result<std::shared_ptr<Schema>> Exclude(const Schema& other) const;

  /// Merge with new fields.
  ///
  /// \param arrow_schema the schema to be merged.
  /// \return A newly merged schema.
  ::arrow::Result<std::shared_ptr<Schema>> Merge(const ::arrow::Schema& arrow_schema) const;

  /// Intersection between two Schemas
  ///
  /// \param other the other schema to run intersection with.
  /// \return the intersection of two schemas.
  ::arrow::Result<std::shared_ptr<Schema>> Intersection(const Schema& other) const;

  /// Add a new parent field.
  void AddField(std::shared_ptr<Field> f);

  /// Get nested field by it. It can access any field in the schema tree.
  std::shared_ptr<Field> GetField(FieldIdType id) const;

  /// Top level fields;
  const std::vector<std::shared_ptr<Field>>& fields() const { return fields_; }

  /// Count the number of all fields, including nested fields.
  int32_t GetFieldsCount() const;

  /// Get all field ids in this schema.
  std::vector<FieldIdType> GetFieldIds() const;

  /// Get the field by fully qualified field name.
  ///
  /// \param name the fully qualified name, i.e., "annotations.box.xmin".
  /// \return the field if found. Return nullptr if not found.
  std::shared_ptr<Field> GetField(const std::string& name) const;

  /// Schema metadata, k/v pairs.
  const std::unordered_map<std::string, std::string>& metadata() const { return metadata_; }

  /// Set metadata key/value pairs.
  void SetMetadata(const std::unordered_map<std::string, std::string>& metadata);

  std::string ToString() const;

  bool Equals(const std::shared_ptr<Schema>& other, bool check_id = true) const;

  bool Equals(const Schema& other, bool check_id = true) const;

  /// Compare two schemas are the same.
  bool operator==(const Schema& other) const;

 private:
  /// (Re-)Assign Field IDs to all the fields.
  void AssignIds();

  /// Get the max assigned ID.
  int32_t GetMaxId() const;

  /// Make a full copy of the schema.
  std::shared_ptr<Schema> Copy() const;

  bool RemoveField(int32_t id);

  std::vector<std::shared_ptr<Field>> fields_;

  /// Schema metadata
  std::unordered_map<std::string, std::string> metadata_;
};

/// Pretty print Lance Schema.
void Print(const Schema& schema);

/// \brief Field is the metadata of a column on disk.
class Field final {
 public:
  /// Default constructor
  Field();

  /// Convert an arrow Field to Field.
  explicit Field(const std::shared_ptr<::arrow::Field>& field);

  explicit Field(const pb::Field& pb);

  /// Add a subfield.
  ::arrow::Status Add(const pb::Field& pb);

  void AddChild(std::shared_ptr<Field> child);

  std::shared_ptr<Field> Get(int32_t id);

  /// Convert to Apache Arrow Field.
  std::shared_ptr<::arrow::Field> ToArrow() const;

  /// Serialize to Protobuf.
  std::vector<lance::format::pb::Field> ToProto() const;

  /// Returns Field ID.
  int32_t id() const;

  int32_t parent_id() const { return parent_; }

  std::shared_ptr<::arrow::DataType> type() const;

  /// Returns the storage type of the field.
  /// If this is extension type, return the type of the underneath array.
  /// Otherwise, it returns the same as type().
  std::shared_ptr<::arrow::DataType> storage_type() const;

  std::string name() const;

  const std::string& logical_type() const { return logical_type_; };

  const std::string& extension_name() { return extension_name_; }

  bool is_extension_type() const { return !extension_name_.empty(); }

  const std::shared_ptr<::arrow::Array>& dictionary() const;

  /// Set the directory values for a dictionary field.
  ///
  /// \param dict_arr dictionary values
  /// \return `status::OK()` if success. Fails if the dictionary is already set.
  ::arrow::Status set_dictionary(std::shared_ptr<::arrow::Array> dict_arr);

  lance::encodings::Encoding encoding() const { return encoding_; };

  ::arrow::Result<std::shared_ptr<lance::encodings::Decoder>> GetDecoder(
      std::shared_ptr<::arrow::io::RandomAccessFile> infile);

  std::shared_ptr<lance::encodings::Encoder> GetEncoder(
      std::shared_ptr<::arrow::io::OutputStream> sink);

  /// Debug String
  std::string ToString() const;

  /// All subfields;
  const std::vector<std::shared_ptr<Field>>& fields() const { return children_; };

  const std::shared_ptr<Field>& field(int i) const { return children_[i]; };

  /// Returns the direct child with the name. Returns nullptr if such field does not exist.
  std::shared_ptr<Field> Get(const std::string_view& name) const;

  /// Check if two fields are equal.
  ///
  /// \param other the point to the other field to check against to.
  /// \param check_id if true, check Id are equal as well. Can be set to false in unit test.
  /// \return true if they are equal.
  bool Equals(const std::shared_ptr<Field>& other, bool check_id = true) const;

  bool Equals(const Field& other, bool check_id = true) const;

  bool operator==(const Field& other) const;

 private:
  explicit Field(const Field& field) = delete;

  std::shared_ptr<Field> Get(const std::vector<std::string>& field_path,
                             std::size_t start_idx = 0) const;

  /// Make a new copy of the field.
  std::shared_ptr<Field> Copy(bool include_children = false) const;

  /// Project an arrow field to this field.
  std::shared_ptr<Field> Project(const std::shared_ptr<::arrow::Field>& arrow_field) const;

  /// Merge an arrow field with this field.
  ::arrow::Result<std::shared_ptr<Field>> Merge(const ::arrow::Field& arrow_field) const;

  /// Load dictionary array from disk.
  ::arrow::Status LoadDictionary(std::shared_ptr<::arrow::io::RandomAccessFile> infile);

  void SetId(int32_t parent_id, int32_t* current_id);

  bool RemoveChild(int32_t id);

  /// Get the fields count recursively.
  ///
  /// It counts all the fields (node) from the schema tree, including the parent nodes.
  int32_t GetFieldsCount() const;

  // TODO: use enum to replace protobuf enum.
  pb::Field::Type GetNodeType() const;

  void Init(std::shared_ptr<::arrow::DataType> dtype);

  int32_t id_ = -1;
  int32_t parent_ = -1;
  std::string name_;
  std::string logical_type_;
  std::string extension_name_;
  lance::encodings::Encoding encoding_ = lance::encodings::NONE;

  // Dictionary type
  int64_t dictionary_offset_ = -1;
  int64_t dictionary_page_length_ = 0;
  std::shared_ptr<::arrow::Array> dictionary_;

  std::mutex lock_;

  friend class FieldVisitor;
  friend class ReadDictionaryVisitor;
  friend class ToArrowVisitor;
  friend class WriteDictionaryVisitor;
  friend class Schema;
  friend ::arrow::Status CopyField(std::shared_ptr<Field> new_field,
                                   std::shared_ptr<Field> field,
                                   std::vector<std::string> components,
                                   std::size_t comp_idx);
  friend void Print(const Field& field, const std::string& path, int indent);
  friend ::arrow::Result<std::shared_ptr<Field>> Intersection(const Field& lhs, const Field& rhs);

  std::vector<std::shared_ptr<Field>> children_;
};

}  // namespace lance::format
