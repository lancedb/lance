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

#include <arrow/array.h>
#include <arrow/result.h>
#include <arrow/type_fwd.h>
#include <arrow/type_traits.h>

#include <initializer_list>
#include <memory>
#include <vector>

namespace lance::arrow {

/// \file STL adaptors between Arrow and STL.

/// Convert a vector to Apache Arrow Array.
template <typename T>
::arrow::Result<std::shared_ptr<
    typename ::arrow::TypeTraits<typename ::arrow::CTypeTraits<T>::ArrowType>::ArrayType>>
ToArray(const std::vector<T>& vec, ::arrow::MemoryPool* pool = ::arrow::default_memory_pool()) {
  using ArrowType = typename ::arrow::CTypeTraits<T>::ArrowType;
  using ArrayType = typename ::arrow::TypeTraits<ArrowType>::ArrayType;
  typename ::arrow::TypeTraits<ArrowType>::BuilderType builder(pool);
  ARROW_RETURN_NOT_OK(builder.Reserve(vec.size()));
  for (auto& v : vec) {
    ARROW_RETURN_NOT_OK(builder.Append(v));
  }
  auto result = builder.Finish();
  if (!result.ok()) {
    return result.status();
  }
  return std::static_pointer_cast<ArrayType>(*result);
}

template <typename T>
::arrow::Result<std::shared_ptr<
    typename ::arrow::TypeTraits<typename ::arrow::CTypeTraits<T>::ArrowType>::ArrayType>>
ToArray(std::initializer_list<T> values,
        ::arrow::MemoryPool* pool = ::arrow::default_memory_pool()) {
  return ToArray(std::vector<T>(values), pool);
}

}  // namespace lance::arrow
