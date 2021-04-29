/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file broadcast_to.cc
 * \brief
 */
#include <sstream>
#include <cctype>
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "eletwise.h"
#include "vector_tiling.h"
#include <iostream>

namespace optiling {
bool BroadcastToTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                       OpRunInfo& run_info) {
  CHECK(op_paras.inputs.size() == 2, "op [%s] : length of op_paras.inputs should be 2", op_type.c_str());
  CHECK(!op_paras.inputs[0].tensor.empty(), "op [%s] : op_paras.inputs[0].tensor cannot be empty", op_type.c_str());

  std::vector<int64_t> x_runtime_shape = op_paras.inputs[0].tensor[0].shape;
  std::vector<int64_t> shape_value;

  std::string shape_dtype = op_paras.inputs[1].tensor[0].dtype;
  auto pointer = std::get<0>(op_paras.const_inputs.at("shape"));
  auto size = std::get<1>(op_paras.const_inputs.at("shape"));

  uint32_t count = (shape_dtype == "int64")   ? size / sizeof(int64_t)
                   : (shape_dtype == "int32") ? size / sizeof(int32_t)
                                              : 0;
  CHECK(count != 0, "op [%s]: input shape shape cannot be empty", op_type.c_str());

  if (shape_dtype == "int64") {
    auto* data = (int64_t*)pointer;
    while (count--) {
      shape_value.push_back(*data++);
    }
  }
  if (shape_dtype == "int32") {
    auto* data = (int32_t*)pointer;
    while (count--) {
      shape_value.push_back(*data++);
    }
  }

  std::vector<int64_t> broadcast_shape = {};
  std::vector<int64_t> output_shape = {};

  // align shape for shape and input shapes
  uint64_t len_diff = shape_value.size() - x_runtime_shape.size();
  CHECK((len_diff >= 0), "op [%s]: length of shape should not be less than input_x's dimension", op_type.c_str());
  int64_t const_value_front =
      std::accumulate(shape_value.begin(), shape_value.begin() + len_diff, 1, std::multiplies<int>());
  broadcast_shape.push_back(const_value_front);
  for (uint64_t i = 0; i < x_runtime_shape.size(); i++) {
    broadcast_shape.push_back(shape_value[len_diff + i]);
  }

  TeOpParas op_paras_tmp = op_paras;
  // update new shape
  op_paras_tmp.inputs[1].tensor[0].shape = std::move(broadcast_shape);
  op_paras_tmp.outputs[0].tensor[0].shape = std::move(broadcast_shape);

  bool ret = EletwiseTiling(op_type, const_cast<TeOpParas&>(op_paras_tmp), op_info, run_info);
  return ret;
}

// register tiling interface of the BroadcastTo op.
REGISTER_OP_TILING_FUNC_BUFFERED(BroadcastTo, BroadcastToTiling);
}  // namespace optiling
