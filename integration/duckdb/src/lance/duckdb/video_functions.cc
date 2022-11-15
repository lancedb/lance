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

#include "lance/duckdb/video_functions.h"

#include <cstdint>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <vector>


namespace lance::duckdb {

::std::unique_ptr<::duckdb::TableFunctionRef> VideoScanReplacement(
    ::duckdb::ClientContext &context,
    const ::std::string &table_name,
    ::duckdb::ReplacementScanData *data) {
  auto lower_name = ::duckdb::StringUtil::Lower(table_name);
  if (!::duckdb::StringUtil::EndsWith(lower_name, ".mp4") && !::duckdb::StringUtil::Contains(lower_name, ".mp4?")) {
    return nullptr;
  }
  auto table_function = ::duckdb::make_unique<::duckdb::TableFunctionRef>();
  ::std::vector<::std::unique_ptr<::duckdb::ParsedExpression>> children;
  children.push_back(::duckdb::make_unique<::duckdb::ConstantExpression>(::duckdb::Value(table_name)));
  table_function->function = ::duckdb::make_unique<::duckdb::FunctionExpression>("video_scan", move(children));
  return table_function;
}

struct VideoFunctionData : public ::duckdb::TableFunctionData {
  VideoFunctionData() = default;

  // video uri
  std::string uri;
  bool finished = false;
};

std::unique_ptr<::duckdb::FunctionData> VideoFunctionBind(
    ::duckdb::ClientContext& context,
    ::duckdb::TableFunctionBindInput& input,
    std::vector<::duckdb::LogicalType>& return_types,
    std::vector<std::string>& names) {
  auto result = std::make_unique<VideoFunctionData>();
  result->uri = input.inputs[0].GetValue<std::string>();

  auto children = {
      ::std::pair<::std::string, ::duckdb::LogicalType>("data", ::duckdb::LogicalType::BLOB),
      ::std::pair<::std::string, ::duckdb::LogicalType>("width", ::duckdb::LogicalType::INTEGER),
      ::std::pair<::std::string, ::duckdb::LogicalType>("height", ::duckdb::LogicalType::INTEGER),
      ::std::pair<::std::string, ::duckdb::LogicalType>("channels", ::duckdb::LogicalType::INTEGER)
  };
  return_types.push_back(::duckdb::LogicalType::STRUCT(children));
  names.emplace_back("frame");
  return std::move(result);
}

void VideoScanner(::duckdb::ClientContext& context,
                  ::duckdb::TableFunctionInput& data_p,
                  ::duckdb::DataChunk& output) {
  auto data = const_cast<VideoFunctionData*>(
      dynamic_cast<const VideoFunctionData*>(data_p.bind_data));
  if (data->finished) {
    return;
  }
  auto uri = data->uri;

  // initialize a video capture object
  ::cv::VideoCapture vid_capture(uri);

  // Print error message if the stream is invalid
  if (!vid_capture.isOpened()) {
    ::std::cout << "Error opening video stream or file" << ::std::endl;
  } else {
    // Obtain fps and frame count by get() method and print
    // You can replace 5 with CAP_PROP_FPS as well, they are enumerations
    int fps = vid_capture.get(5);
    ::std::cout << "Frames per second :" << fps;

    // Obtain frame_count using opencv built in frame count reading method
    // You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
    int frame_count = vid_capture.get(7);
    ::std::cout << "  Frame count :" << frame_count;
  }

  auto frame_Number = vid_capture.get(::cv::CAP_PROP_FRAME_COUNT);
  output.SetCardinality(100);
  int i = 0;
  // Read the frames to the last frame
  while (vid_capture.isOpened()) {
    // Initialise frame matrix
    ::cv::Mat frame;
    // If frames are present, show it
    if(vid_capture.read(frame)) {
      const uchar* frame_data = (frame.isContinuous() ? frame.data : frame.clone().data);
      auto size = frame.cols * frame.rows * frame.channels();
      auto children = {
          ::std::pair<::std::string, ::duckdb::Value>("data", ::duckdb::Value::BLOB(frame_data,size)),
          ::std::pair<::std::string, ::duckdb::Value>("width", frame.cols),
          ::std::pair<::std::string, ::duckdb::Value>("height", frame.rows),
          ::std::pair<::std::string, ::duckdb::Value>("channels", frame.channels())
      };
      auto frame_struct = ::duckdb::Value::STRUCT(children);
      output.SetValue(0, i, frame_struct);
    }
    i++;
    if (i >= 100) {
      break;
    }
  }
  // Release the video capture object
  vid_capture.release();
  data->finished = true;
}


std::vector<std::unique_ptr<::duckdb::CreateTableFunctionInfo>> GetVideoTableFunctions() {
  std::vector<std::unique_ptr<::duckdb::CreateTableFunctionInfo>> functions;

  ::duckdb::TableFunction video_scanner("video_scan", {::duckdb::LogicalType::VARCHAR}, VideoScanner, VideoFunctionBind);
  functions.emplace_back(std::make_unique<::duckdb::CreateTableFunctionInfo>(video_scanner));

  return functions;
}

}  // namespace lance::duckdb::ml