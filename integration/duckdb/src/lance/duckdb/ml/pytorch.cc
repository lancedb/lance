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

#include "lance/duckdb/ml/pytorch.h"

#include <torch/script.h>

#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

namespace lance::duckdb::ml {

/// https://stackoverflow.com/questions/65617755/how-to-replicate-pytorch-normalization-in-opencv-or-numpy
torch::Tensor normalize(const torch::Tensor& input,
                        std::vector<float> means = {0.485, 0.456, 0.406},
                        std::vector<float> std = {0.229, 0.224, 0.225}) {
  auto opts = torch::TensorOptions().dtype(torch::kFloat);
  at::Tensor mean_tensor =
      torch::from_blob(means.data(), {static_cast<int64_t>(means.size())}, opts)
          .reshape({-1, 1, 1});
  at::Tensor std_tensor =
      torch::from_blob(std.data(), {static_cast<int64_t>(std.size())}, opts).reshape({-1, 1, 1});
  return torch::sub(input / 255, mean_tensor).div(std_tensor);
}

std::string ShapeString(const torch::Tensor& tensor) {
  std::stringstream ss;
  ss << "(" << tensor.size(0);
  for (int i = 1; i < tensor.dim(); i++) {
    ss << ", " << tensor.size(i);
  }
  ss << ")";
  return ss.str();
}

/// Convert OpenCV image to PyTorch Tensor
torch::Tensor ToTensor(const cv::Mat& image) {
  at::TensorOptions options(at::kFloat);
  auto tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, options);
  return tensor.permute({0, 3, 1, 2});
}

std::unique_ptr<ModelEntry> PyTorchModelEntry::Make(const std::string& name,
                                                    const std::string& uri) {
  // Let's just support local model for now
  torch::jit::script::Module module;
  try {
    module = torch::jit::load(uri);
  } catch (const c10::Error& e) {
    throw ::duckdb::IOException("Error loading the model: " + e.msg());
  }
  return std::unique_ptr<ModelEntry>(new PyTorchModelEntry{name, uri, module});
}

void PyTorchModelEntry::Execute(::duckdb::DataChunk& args,
                                ::duckdb::ExpressionState& state,
                                ::duckdb::Vector& result) {
  result.SetVectorType(::duckdb::VectorType::FLAT_VECTOR);
  for (int i = 0; i < args.size(); i++) {
    auto img_bytes = ::duckdb::StringValue::Get(args.data[1].GetValue(i));
    cv::Mat1b buf(img_bytes.size(), 1, (unsigned char*)(img_bytes.data()));
    auto mat = cv::imdecode(buf, cv::IMREAD_UNCHANGED);
    if (mat.data == nullptr) {
      std::cerr << "Failed to parse image: image size=" << img_bytes.size()
                << " buf=" << buf.size().width << "\n";
      continue;
    }
    cv::Mat rgb_mat;
    cv::cvtColor(mat, rgb_mat, cv::COLOR_BGR2RGB);
    // Convert to float matrix.
    cv::Mat fmat;
    rgb_mat.convertTo(fmat, cv::DataType<float>::type);

    // TODO: support batch mode
    auto input_tensor = ToTensor(fmat);
    input_tensor = normalize(input_tensor);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    auto output = module_.forward(inputs).toTensor();

    // OMG this copying is painfully slow.
    std::vector<::duckdb::Value> values;
    auto softmax = output[0].softmax(0);
    for (int i = 0; i < softmax.size(0); i++) {
      values.emplace_back(::duckdb::Value::FLOAT(*softmax[i].data_ptr<float>()));
    }
    result.SetValue(i, ::duckdb::Value::LIST(values));
  }
}

void Predict(::duckdb::DataChunk& args,
             ::duckdb::ExpressionState& state,
             ::duckdb::Vector& result) {
  auto ml_catalog = ModelCatalog::Get();
  assert(ml_catalog);
  assert(args.ColumnCount() == 2);

  assert(args.size() > 0);

  auto model_name = args.data[0].GetValue(0).GetValue<std::string>();
  auto model = ml_catalog->Get(model_name);
  if (model == nullptr) {
    throw ::duckdb::InvalidInputException("Model " + model_name + " is not found");
  }
  model->Execute(args, state, result);
}

std::vector<std::unique_ptr<::duckdb::CreateFunctionInfo>> GetPyTorchFunctions() {
  /// Initialize singleton
  auto catalog = ModelCatalog::Get();
  assert(catalog != nullptr);

  std::vector<std::unique_ptr<::duckdb::CreateFunctionInfo>> functions;
  // Predict
  ::duckdb::ScalarFunctionSet predict("predict");
  predict.AddFunction(
      ::duckdb::ScalarFunction({::duckdb::LogicalType::VARCHAR, ::duckdb::LogicalType::BLOB},
                               ::duckdb::LogicalType::LIST(::duckdb::LogicalType::FLOAT),
                               Predict));
  functions.emplace_back(std::make_unique<::duckdb::CreateScalarFunctionInfo>(predict));

  // Release pytorch model.
  return functions;
}

}  // namespace lance::duckdb::ml