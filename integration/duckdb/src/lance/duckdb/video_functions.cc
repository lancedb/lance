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

#include <duckdb/parser/parsed_data/create_table_function_info.hpp>
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include <iostream>
#include <memory>
#include <opencv2/videoio/videoio.hpp>
#include <vector>


namespace lance::duckdb {

/**
 * If we see SELECT * FROM '*.mp4', interpret this as a video_scan call
 */
::std::unique_ptr<::duckdb::TableFunctionRef> VideoScanReplacement(
    ::duckdb::ClientContext &context,
    const ::std::string &table_name,
    ::duckdb::ReplacementScanData *data) {
  auto lower_name = ::duckdb::StringUtil::Lower(table_name);
  if (!::duckdb::StringUtil::EndsWith(lower_name, ".mp4") &&
      !::duckdb::StringUtil::Contains(lower_name, ".mp4?")) {
    return nullptr;
  }
  auto table_function = ::duckdb::make_unique<::duckdb::TableFunctionRef>();
  ::std::vector<::std::unique_ptr<::duckdb::ParsedExpression>> children;
  children.emplace_back(::std::make_unique<::duckdb::ConstantExpression>(::duckdb::Value(table_name)));
  table_function->function = ::std::make_unique<::duckdb::FunctionExpression>("video_scan", ::std::move(children));
  return table_function;
}

struct VideoFunctionData : public ::duckdb::TableFunctionData {
  VideoFunctionData() = default;

  // video uri
  std::string uri;

  // output columns;
  std::vector<std::string> names = {
      "duration_ms",
      "frame_num",
      "frame"
  };

  // output data types
  std::vector<::duckdb::LogicalType> types = {
      ::duckdb::LogicalType::DOUBLE,
      ::duckdb::LogicalType::INTEGER,
      // we don't have an image type so we need this for reconstruction
      ::duckdb::LogicalType::STRUCT({
          ::std::pair<::std::string, ::duckdb::LogicalType>("data", ::duckdb::LogicalType::BLOB),
          ::std::pair<::std::string, ::duckdb::LogicalType>("width", ::duckdb::LogicalType::INTEGER),
          ::std::pair<::std::string, ::duckdb::LogicalType>("height", ::duckdb::LogicalType::INTEGER),
          ::std::pair<::std::string, ::duckdb::LogicalType>("channels", ::duckdb::LogicalType::INTEGER)
      })
  };

  ::duckdb::idx_t tot_frames = 0; // total number of names as reported by opencv
  ::duckdb::idx_t frames_per_task = 1024; // standard vector size

public:
  ::std::unique_ptr<FunctionData> Copy() const override {
    throw ::duckdb::NotImplementedException("");
  }
  bool Equals(const FunctionData &other) const override {
    throw ::duckdb::NotImplementedException("");
  }
};

// compute the total number of frames
std::unique_ptr<::duckdb::FunctionData> VideoFunctionBind(
    ::duckdb::ClientContext& context,
    ::duckdb::TableFunctionBindInput& input,
    std::vector<::duckdb::LogicalType>& return_types,
    std::vector<std::string>& names) {
  auto bind_data = std::make_unique<VideoFunctionData>();
  bind_data->uri = input.inputs[0].GetValue<std::string>();
  // TOOD make it so we only need to open it once per thread
  auto cap = ::cv::VideoCapture(bind_data->uri);
  if (!cap.isOpened()) {
    throw ::duckdb::IOException("Cannot open video stream or file");
  } else {
    double frame_count = cap.get(::cv::CAP_PROP_FRAME_COUNT);
    bind_data->tot_frames = (::duckdb::idx_t)frame_count;
  }
  names = bind_data->names;
  return_types = bind_data->types;
  return std::move(bind_data);
}

struct VideoScanLocalState : public ::duckdb::LocalTableFunctionState {
  // TODO projection pushdown
  ::std::vector<::duckdb::column_t> column_ids;
  ::duckdb::idx_t frame_start = -1; // start scanning on this frame
  ::duckdb::idx_t frame_end = -1; // end scanning once you reach this frame
  bool done = false;
  // TODO predicate pushdown
  ::duckdb::TableFilterSet *filters = nullptr;
};

// Maintain global state for parallel scans
struct VideoScanGlobalState : public ::duckdb::GlobalTableFunctionState {
  explicit VideoScanGlobalState(::duckdb::idx_t max_threads) : frame_idx(0), max_threads(max_threads) {
  }

  ::std::mutex lock;
  ::duckdb::idx_t frame_idx;  // max frame scanned
  ::duckdb::idx_t max_threads;

  ::duckdb::idx_t MaxThreads() const override {
    return max_threads;
  }
};

// Compute the maximum number of threads to use
static ::duckdb::idx_t VideoScanMaxThreads(
    ::duckdb::ClientContext &context,
    const ::duckdb::FunctionData *bind_data_p) {
  D_ASSERT(bind_data_p);

  auto bind_data = (const VideoFunctionData *)bind_data_p;
  return bind_data->tot_frames / bind_data->frames_per_task;
}

static std::unique_ptr<::duckdb::GlobalTableFunctionState> VideoScanInitGlobalState(
    ::duckdb::ClientContext &context,
    ::duckdb::TableFunctionInitInput &input) {
  auto max_threads = VideoScanMaxThreads(context, input.bind_data);
  return ::std::make_unique<VideoScanGlobalState>(max_threads);
}

// Get the next parallel scan parameters
// return true if there's more data to scan still
static bool VideoScanParallelStateNext(
    ::duckdb::ClientContext &context,
    const ::duckdb::FunctionData *bind_data_p,
    VideoScanLocalState &lstate,
    VideoScanGlobalState &gstate) {
  D_ASSERT(bind_data_p);
  auto bind_data = (const VideoFunctionData *)bind_data_p;

  ::std::lock_guard<::std::mutex> parallel_lock(gstate.lock);

  if (gstate.frame_idx < bind_data->tot_frames) {
    auto frame_idx = gstate.frame_idx + bind_data->frames_per_task;
    frame_idx = ::std::min(frame_idx, bind_data->tot_frames);

    lstate.frame_start = gstate.frame_idx;
    lstate.frame_end = frame_idx;
    lstate.done = false;

    gstate.frame_idx += bind_data->frames_per_task;
    return true;
  }
  lstate.done = true;
  return false;
}

static ::std::unique_ptr<::duckdb::LocalTableFunctionState> VideoScanInitLocalState(
    ::duckdb::ExecutionContext &context,
    ::duckdb::TableFunctionInitInput &input,
    ::duckdb::GlobalTableFunctionState *global_state) {
  auto &gstate = (VideoScanGlobalState &)*global_state;
  auto local_state = ::std::make_unique<VideoScanLocalState>();
  local_state->column_ids = input.column_ids;
  local_state->filters = input.filters;
  if (!VideoScanParallelStateNext(context.client, input.bind_data, *local_state, gstate)) {
    local_state->done = true;
  }
  return ::std::move(local_state);
}

// The main function to scan the video and return frames
void VideoScanner(::duckdb::ClientContext& context,
                  ::duckdb::TableFunctionInput& data_p,
                  ::duckdb::DataChunk& output) {
  auto bind_data = (const VideoFunctionData *)data_p.bind_data;
  auto local_state = (VideoScanLocalState *)data_p.local_state;
  auto gstate = (VideoScanGlobalState *)data_p.global_state;

  auto cap = ::cv::VideoCapture(bind_data->uri);
  cap.set(::cv::CAP_PROP_POS_FRAMES, local_state->frame_start);

  if (local_state->done && !VideoScanParallelStateNext(
      context, data_p.bind_data,
      *local_state, *gstate)) {
    if(cap.isOpened()) {
      cap.release();
    }
    return;
  }

  int n_frames = local_state->frame_end - local_state->frame_start;
  for (int i=0; i<n_frames; i++) {
    if (!cap.isOpened()) {
      return;
    }
    int curr_frame = local_state->frame_start + i + 1; // TODO INT_MAX?
    output.SetValue(1, i, curr_frame );

    // Initialise frame matrix
    ::cv::Mat frame;
    // If frames are present, show it
    if(cap.read(frame)) {
      const uchar* frame_data = (frame.isContinuous() ? frame.data : frame.clone().data);
      auto size = frame.cols * frame.rows * frame.channels();
      auto children = {
          ::std::pair<::std::string, ::duckdb::Value>("data", ::duckdb::Value::BLOB(frame_data,size)),
          ::std::pair<::std::string, ::duckdb::Value>("width", frame.cols),
          ::std::pair<::std::string, ::duckdb::Value>("height", frame.rows),
          ::std::pair<::std::string, ::duckdb::Value>("channels", frame.channels())
      };
      auto frame_struct = ::duckdb::Value::STRUCT(children);
      auto ts = cap.get(::cv::CAP_PROP_POS_MSEC);
      output.SetValue(0, i, ts);
      output.SetValue(2, i, frame_struct);
    }
    output.SetCardinality(i + 1);
  }
  cap.release();
  local_state->done = true;
}


class VideoScannerFunction : public ::duckdb::TableFunction {
 public:
  VideoScannerFunction()
      : ::duckdb::TableFunction("video_scan", {::duckdb::LogicalType::VARCHAR},
                                VideoScanner, VideoFunctionBind,
                                VideoScanInitGlobalState,
                                VideoScanInitLocalState) {
  }
};

std::vector<std::unique_ptr<::duckdb::CreateTableFunctionInfo>> GetVideoTableFunctions() {
  std::vector<std::unique_ptr<::duckdb::CreateTableFunctionInfo>> functions;
  auto video_scanner = VideoScannerFunction();
  functions.emplace_back(std::make_unique<::duckdb::CreateTableFunctionInfo>(video_scanner));

  return functions;
}

}  // namespace lance::duckdb::ml