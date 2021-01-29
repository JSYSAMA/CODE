/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file pad_ops.cpp
 * \brief
 */
#include "inc/pad_ops.h"

#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

#include "graph/utils/node_utils.h"

#include "util/util.h"
#include "util/common_shape_fns.h"
#include "util/error_util.h"
#include "op_log.h"

namespace ge {
// ----------------PadD Op Begin-------------------
static graphStatus PadDInferShapeAndType(ge::Operator& op, std::vector<std::vector<int64_t>>& paddings) {
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  Shape shape_x = op.GetInputDesc("x").GetShape();
  op.GetInputDesc("x").GetShapeRange(shape_range);
  // shape_x is UNKNOWN_RANK
  if (shape_x.GetDims() == UNKNOWN_RANK) {
    DataType input_dtype = op.GetInputDesc("x").GetDataType();
    vector<int64_t> shape(1, -2);
    Shape out_shape(shape);
    TensorDesc tensordesc_output = op.GetOutputDesc("y");
    tensordesc_output.SetShape(out_shape);
    tensordesc_output.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("y", tensordesc_output);
    return GRAPH_SUCCESS;
  }
  // adapter net
  vector<int64_t> shape;
  int64_t dim_cur = 0;
  if (shape_x.GetDimNum() != paddings.size()) {
    OpsInputShapeErrReport(op.GetName(), "Paddings and shape should be the same length", "x",
                           ConcatString(shape_x.GetDimNum()));
    OP_LOGE(op.GetName().c_str(),
            "Paddings and shape"
            "are not the same length.");
    return GRAPH_FAILED;
  }
  for (size_t dim = 0; dim < shape_x.GetDimNum(); dim++) {
    if (paddings[dim].size() != 2) {
      OpsInputShapeErrReport(op.GetName(), "Paddings's shape should be in the form of (n,2)", "paddings",
                             ConcatString(paddings[dim].size()));
      OP_LOGE(op.GetName().c_str(),
              "Paddings's shape"
              "is not in the form of (n,2)");
      return GRAPH_FAILED;
    }
  }
  for (size_t dim = 0; dim < shape_x.GetDimNum(); dim++) {
    dim_cur = shape_x.GetDim(dim) + paddings[dim][0] + paddings[dim][1];
    OP_LOGD("OP[PadD]", "output_shape[%d] is [%d].", dim, dim_cur);
    shape.push_back(dim_cur);
  }

  // Dynamic Set Range
  std::vector<std::pair<int64_t, int64_t>> out_range = shape_range;
  if (shape_range.size() > 0) {
    for (size_t dim = 0; dim < shape_range.size(); dim++) {
      OP_LOGD("OP[PadD]", "Bout_range[%d].first is [%d].", dim, out_range[dim].first);
      OP_LOGD("OP[PadD]", "Bout_range[%d].second is [%d].", dim, out_range[dim].second);
      out_range[dim].first += (out_range[dim].first >= 0) ? paddings[dim][0] + paddings[dim][1] : 0;
      out_range[dim].second += (out_range[dim].second >= 0) ? paddings[dim][0] + paddings[dim][1] : 0;
      OP_LOGD("OP[PadD]", "Aout_range[%d].first is [%d].", dim, out_range[dim].first);
      OP_LOGD("OP[PadD]", "Aout_range[%d].second is [%d].", dim, out_range[dim].second);
    }
  }

  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  Shape out_shape(shape);
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(out_shape);
  tensordesc_output.SetDataType(input_dtype);
  if (out_range.size() > 0) {
    tensordesc_output.SetShapeRange(out_range);
  }
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  OP_LOGD("OP[PadD]", "PadDInferShape END.");
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(PadDInferShape) {
  OP_LOGD("OP[PadD]", "PadDInferShape Begin.");
  const vector<string> depends;
  PREPARE_DYNAMIC_SHAPE(depends);
  std::vector<std::vector<int64_t>> paddings;
  if (ge::GRAPH_SUCCESS != op.GetAttr("paddings", paddings)) {
    OpsGetAttrErrReport(op.GetName(), "paddings");
    return GRAPH_FAILED;
  }

  return PadDInferShapeAndType(op, paddings);
}

COMMON_INFER_FUNC_REG(PadD, PadDInferShape);
// ----------------PadD Op End-------------------

// ----------------Pad Op Begin-------------------
static graphStatus PadInferShapeAndType(ge::Operator& op, std::vector<int64_t>& paddings) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_shape = input_desc->MutableShape().GetDims();
  auto input_dtype = input_desc->GetDataType();
  auto output_desc = op_info->MutableOutputDesc("y");
  output_desc->SetDataType(input_dtype);

  if (!IsUnknown(input_shape)) {
    // not dynamic shape, will output shape and dtype
    if (input_shape.empty()) {
      input_shape.push_back(1);
    }
    if (input_shape.size() * 2 != paddings.size()) {
      OP_LOGE("OP[Pad]", "the num of paddings must be double the input dim size");
      return GRAPH_FAILED;
    }

    // calce the output shape
    vector<int64_t> output_shape;
    for (size_t dim = 0; dim < input_shape.size(); dim++) {
      output_shape.push_back(input_shape[dim] + paddings[dim * 2] + paddings[dim * 2 + 1]);
    }
    output_desc->SetShape(GeShape(output_shape));
  
    return GRAPH_SUCCESS;
  }

  // input shape is -2, output is -2
  if (IsUnknownRankShape(input_shape)) {
    output_desc->SetShape(GeShape(input_shape));
  
    return GRAPH_SUCCESS;
  }

  // input shape is -1, will get the shape and range
  // calcu the output shape
  vector<int64_t> output_shape;
  for (size_t dim = 0; dim < input_shape.size(); dim++) {
    if (input_shape[dim] == -1) {
      output_shape.push_back(input_shape[dim]);
    } else {
      output_shape.push_back(input_shape[dim] + paddings[dim * 2] + paddings[dim * 2 + 1]);
    }
  }
  output_desc->SetShape(GeShape(output_shape));

  // calcu the output range
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc->GetShapeRange(input_range);
  MakeUpShapeRange(input_shape, input_range);
  std::vector<std::pair<int64_t, int64_t>> output_range;
  for (size_t dim = 0; dim < input_shape.size(); dim++) {
    auto range_min = input_range[dim].first + paddings[dim * 2] + paddings[dim * 2 + 1];
    auto range_max = input_range[dim].second == -1 ?
                     -1 : input_range[dim].second + paddings[dim * 2] + paddings[dim * 2 + 1];
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  }
  output_desc->SetShapeRange(output_range);

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(PadInferShape) {
  OP_LOGD("OP[Pad]", "PadInferShape Begin.");
  const vector<string> depend_names = {"paddings"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  // first get the padding const
  auto node = NodeUtils::GetNodeFromOperator(op);
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorPtr paddings_tensor = nullptr;
  if (GRAPH_SUCCESS != NodeUtils::GetInputConstData(node, "paddings", paddings_tensor)) {
    OP_LOGW("OP[Pad]", "the node paddings is not const node, will set the output dynamic");
    auto input_desc = op_info->MutableInputDesc("x");
    auto input_shape = input_desc->MutableShape().GetDims();
    DataType input_dtype = input_desc->GetDataType();
    auto output_desc = op_info->MutableOutputDesc("y");
  
    // shape_x is UNKNOWN_RANK
    if (IsUnknownRankShape(input_shape)) {
      OP_LOGW("OP[Pad]", "shape_x is UNKNOWN_RANK. Set output UNKNOWN_RANK");
      output_desc->SetShape(GeShape(input_shape));
      output_desc->SetDataType(input_dtype);
      return GRAPH_SUCCESS;
    }
    // shape_x is UNKNOWN_DIM
    if (input_shape.empty()) {
      input_shape.push_back(-1);
    }
    vector<int64_t> out_shape;
    for (size_t dim = 0; dim < input_shape.size(); dim++) {
      out_shape.push_back(-1);
    }
    std::vector<std::pair<int64_t, int64_t>> output_range;
    MakeUpShapeRange(out_shape, output_range);
    output_desc->SetShape(GeShape(out_shape));
    output_desc->SetDataType(input_dtype);
    output_desc->SetShapeRange(output_range);
    return GRAPH_SUCCESS;
  }

  // get const paddings data
  auto const_desc = op_info->MutableInputDesc("paddings");
  auto const_dtype = const_desc->GetDataType();
  std::vector<int64_t> paddings;
  if (!GetConstValue(op, paddings_tensor, const_dtype, paddings)) {
    OP_LOGE(op.GetName().c_str(), "Get Const paddings value failed, infershape failed");
    return GRAPH_FAILED;
  }

  return PadInferShapeAndType(op, paddings);
}

COMMON_INFER_FUNC_REG(Pad, PadInferShape);
// ----------------Pad Op End-------------------

// ----------------PadV2D Op Begin-------------------
static graphStatus PadV2DInferShapeAndType(ge::Operator& op, std::vector<std::vector<int64_t>>& paddings) {
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  Shape shape_x = op.GetInputDesc("x").GetShape();
  op.GetInputDesc("x").GetShapeRange(shape_range);
  // shape_x is UNKNOWN_RANK
  if (shape_x.GetDims() == UNKNOWN_RANK) {
    DataType input_dtype = op.GetInputDesc("x").GetDataType();
    vector<int64_t> shape(1, -2);
    Shape out_shape(shape);
    TensorDesc tensordesc_output = op.GetOutputDesc("y");
    tensordesc_output.SetShape(out_shape);
    tensordesc_output.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("y", tensordesc_output);
    return GRAPH_SUCCESS;
  }
  // adapter net
  vector<int64_t> shape;
  int64_t dim_cur = 0;
  if (shape_x.GetDimNum() != paddings.size()) {
    OpsInputShapeErrReport(op.GetName(), "Paddings and shape should be the same length", "x",
                           ConcatString(shape_x.GetDimNum()));
    OP_LOGE(op.GetName().c_str(),
            "Paddings and shape are not the same length.");
    return GRAPH_FAILED;
  }
  for (size_t dim = 0; dim < shape_x.GetDimNum(); dim++) {
    if (paddings[dim].size() != 2) {
      OpsInputShapeErrReport(op.GetName(), "Paddings's shape should be in the form of (n,2)", "paddings",
                             ConcatString(paddings[dim].size()));
      OP_LOGE(op.GetName().c_str(),
              "Paddings's shape"
              "is not in the form of (n,2)");
      return GRAPH_FAILED;
    }
  }
  for (size_t dim = 0; dim < shape_x.GetDimNum(); dim++) {
    dim_cur = shape_x.GetDim(dim) + paddings[dim][0] + paddings[dim][1];
    OP_LOGD("OP[PadV2D]", "output_shape[%d] is [%d].", dim, dim_cur);
    shape.push_back(dim_cur);
  }

  // Dynamic Set Range
  std::vector<std::pair<int64_t, int64_t>> out_range = shape_range;
  if (shape_range.size() > 0) {
    for (size_t dim = 0; dim < shape_range.size(); dim++) {
      OP_LOGD("OP[PadV2D]", "Bout_range[%d].first is [%d].", dim, out_range[dim].first);
      OP_LOGD("OP[PadV2D]", "Bout_range[%d].second is [%d].", dim, out_range[dim].second);
      out_range[dim].first += (out_range[dim].first >= 0) ? paddings[dim][0] + paddings[dim][1] : 0;
      out_range[dim].second += (out_range[dim].second >= 0) ? paddings[dim][0] + paddings[dim][1] : 0;
      OP_LOGD("OP[PadV2D]", "Aout_range[%d].first is [%d].", dim, out_range[dim].first);
      OP_LOGD("OP[PadV2D]", "Aout_range[%d].second is [%d].", dim, out_range[dim].second);
    }
  }

  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  Shape out_shape(shape);
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(out_shape);
  tensordesc_output.SetDataType(input_dtype);
  if (out_range.size() > 0) {
    tensordesc_output.SetShapeRange(out_range);
  }
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  OP_LOGD("OP[PadV2D]", "PadV2DInferShape END.");
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(PadV2DInferShape) {
  OP_LOGD("OP[PadV2D]", "PadV2DInferShape Begin.");
  std::vector<std::vector<int64_t>> paddings;
  if (ge::GRAPH_SUCCESS != op.GetAttr("paddings", paddings)) {
    OpsGetAttrErrReport(op.GetName(), "paddings");
    return GRAPH_FAILED;
  }

  return PadV2DInferShapeAndType(op, paddings);
}

COMMON_INFER_FUNC_REG(PadV2D, PadV2DInferShape);
// ----------------PadV2D Op End-------------------

// ----------------PadV2 Op Begin-------------------
static graphStatus PadV2InferShapeAndType(ge::Operator& op, std::vector<int64_t>& paddings) {
  Shape shape_x = op.GetInputDesc("x").GetShape();
  // shape_x is UNKNOWN_RANK
  if (shape_x.GetDims() == UNKNOWN_RANK) {
    OP_LOGD("OP[PadV2]", "shape_x is UNKNOWN_RANK. Couldn't set shape_range");
    DataType input_dtype = op.GetInputDesc("x").GetDataType();
    vector<int64_t> shape(1, -2);
    Shape out_shape(shape);
    TensorDesc tensordesc_output = op.GetOutputDesc("y");
    tensordesc_output.SetShape(out_shape);
    tensordesc_output.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("y", tensordesc_output);
    return GRAPH_SUCCESS;
  }

  // adapter net
  vector<int64_t> shape;
  int64_t dim_cur = 0;
  if (shape_x.GetDimNum() * 2 != paddings.size()) {
    return GRAPH_FAILED;
  }

  for (size_t dim = 0; dim < shape_x.GetDimNum(); dim++) {
    dim_cur = shape_x.GetDim(dim) + paddings[dim * 2] + paddings[dim * 2 + 1];
    shape.push_back(dim_cur);
  }

  for (size_t dim = 0; dim < shape_x.GetDimNum(); dim++) {
    if (shape_x.GetDim(dim) == UNKNOWN_DIM) {
      shape[dim] = UNKNOWN_DIM;
    }
  }

  std::vector<std::pair<int64_t, int64_t>> shape_range;
  op.GetInputDesc("x").GetShapeRange(shape_range);
  std::vector<std::pair<int64_t, int64_t>> out_range = shape_range;
  // Dynamic Set Range
  if (shape_range.size() > 0) {
    for (size_t dim = 0; dim < shape_range.size(); dim++) {
      OP_LOGD("OP[PadV2]", "Bout_range[%d].first is [%d].", dim, out_range[dim].first);
      OP_LOGD("OP[PadV2]", "Bout_range[%d].second is [%d].", dim, out_range[dim].second);
      out_range[dim].first += (out_range[dim].first >= 0) ? paddings[dim * 2] + paddings[dim * 2 + 1] : 0;
      out_range[dim].second += (out_range[dim].second >= 0) ? paddings[dim * 2] + paddings[dim * 2 + 1] : 0;
      OP_LOGD("OP[PadV2]", "Aout_range[%d].first is [%d].", dim, out_range[dim].first);
      OP_LOGD("OP[PadV2]", "Aout_range[%d].second is [%d].", dim, out_range[dim].second);
    }
  }

  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  Shape out_shape(shape);
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(out_shape);
  tensordesc_output.SetDataType(input_dtype);
  if (shape_range.size() > 0) {
    tensordesc_output.SetShapeRange(out_range);
  }
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  OP_LOGD("OP[PadV2]", "PadV2InferShape Branch1 END.");
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(PadV2InferShape) {
  OP_LOGD("OP[PadV2]", "PadV2InferShape Begin.");
  Tensor paddings_tensor;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  //op_desc->SetOpInferDepends({"paddings"});

  if (ge::GRAPH_SUCCESS != op.GetInputConstData("paddings", paddings_tensor)) {
    Shape shape_x = op.GetInputDesc("x").GetShape();

    // shape_x is UNKNOWN_RANK
    if (shape_x.GetDims() == UNKNOWN_RANK) {
      OP_LOGD("OP[PadV2]", "shape_x is UNKNOWN_RANK. Couldn't set shape_range");
      DataType input_dtype = op.GetInputDesc("x").GetDataType();
      vector<int64_t> shape(1, -2);
      Shape out_shape(shape);
      TensorDesc tensordesc_output = op.GetOutputDesc("y");
      tensordesc_output.SetShape(out_shape);
      tensordesc_output.SetDataType(input_dtype);
      (void)op.UpdateOutputDesc("y", tensordesc_output);
      return GRAPH_SUCCESS;
    }

    // shape_x is UNKNOWN_DIM
    vector<int64_t> shape;
    for (size_t dim = 0; dim < shape_x.GetDimNum(); dim++) {
      shape.push_back(UNKNOWN_DIM);
    }
    DataType input_dtype = op.GetInputDesc("x").GetDataType();
    TensorDesc tensordesc_output = op.GetOutputDesc("y");
    Shape out_shape(shape);
    tensordesc_output.SetShape(out_shape);
    tensordesc_output.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("y", tensordesc_output);
    OP_LOGD("OP[PadV2]", "PadV2InferShape Branch0 END.");
    return GRAPH_SUCCESS;
  }
  DataType dtype = op.GetInputDesc("paddings").GetDataType();
  std::vector<int64_t> paddings;
  if (!GetConstValue(op, paddings_tensor, dtype, paddings)) {
    OP_LOGE(op.GetName().c_str(), "Get Const Value failed ");
    return GRAPH_FAILED;
  }

  return PadV2InferShapeAndType(op, paddings);
}

COMMON_INFER_FUNC_REG(PadV2, PadV2InferShape);
// ----------------PadV2 Op End-------------------

// ----------------PadV3D Op Begin-------------------
IMPLEMT_COMMON_INFERFUNC(PadV3DInferShape) {
  std::vector<std::vector<int64_t>> paddings;
  if (ge::GRAPH_SUCCESS != op.GetAttr("paddings", paddings)) {
    OpsGetAttrErrReport(op.GetName(), "paddings");
    return GRAPH_FAILED;
  }

  bool paddings_contiguous = true;
  if (op.GetAttr("paddings_contiguous", paddings_contiguous) == GRAPH_FAILED) {
    OP_LOGI(op.GetName().c_str(), "Get attr [paddings_contiguous] failed");
  }

  if (!paddings_contiguous) {
    std::vector<int64_t> pads;
    int64_t rank = paddings.size() / 2;
    for (int64_t i = 0; i < rank; i++) {
      pads.push_back(paddings[i][0]);
      pads.push_back(paddings[i][1]);
    }
    paddings.clear();
    for (int64_t i = 0; i < rank; i++) {
      paddings.push_back({pads[i], pads[i + rank]});
    }
    OP_LOGI(op.GetName().c_str(), "Get attr paddings_contiguous = false");
  } else {
    OP_LOGI(op.GetName().c_str(), "Get attr paddings_contiguous = true[default]");
  }

  return PadDInferShapeAndType(op, paddings);
}

COMMON_INFER_FUNC_REG(PadV3D, PadV3DInferShape);
// ----------------PadV3D Op End-------------------

// ----------------PadV3 Op Begin-------------------
IMPLEMT_COMMON_INFERFUNC(PadV3InferShape) {
  Tensor paddings_tensor;
  if (ge::GRAPH_SUCCESS != op.GetInputConstData("paddings", paddings_tensor)) {
    OP_LOGE(op.GetName().c_str(), "Get Const Value [paddings] failed, Setting shape to UNKNOWN_DIM");
    Shape shape_x = op.GetInputDesc("x").GetShape();
    vector<int64_t> shape;
    for (size_t dim = 0; dim < shape_x.GetDimNum(); dim++) {
      shape.push_back(UNKNOWN_DIM);
    }
    DataType input_dtype = op.GetInputDesc("x").GetDataType();
    TensorDesc tensordesc_output = op.GetOutputDesc("y");
    Shape out_shape(shape);
    tensordesc_output.SetShape(out_shape);
    tensordesc_output.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("y", tensordesc_output);
    return GRAPH_SUCCESS;
  }

  DataType dtype = op.GetInputDesc("paddings").GetDataType();

  std::vector<int64_t> paddings;
  if (!GetConstValue(op, paddings_tensor, dtype, paddings)) {
    OP_LOGE(op.GetName().c_str(), "Get Const Value [paddings] failed ");
    return GRAPH_FAILED;
  }

  bool paddings_contiguous = true;
  if (op.GetAttr("paddings_contiguous", paddings_contiguous) == GRAPH_FAILED) {
    OP_LOGI(op.GetName().c_str(), "Get attr [paddings_contiguous] failed");
  }

  if (!paddings_contiguous) {
    std::vector<int64_t> pads;
    int64_t rank = paddings.size() / 2;
    for (int i = 0; i < rank; i++) {
      pads.push_back(paddings[i]);
      pads.push_back(paddings[i + rank]);
    }
    paddings = pads;
    OP_LOGI(op.GetName().c_str(), "Get attr paddings_contiguous = false");
  } else {
    OP_LOGI(op.GetName().c_str(), "Get attr paddings_contiguous = true[default]");
  }

  return PadInferShapeAndType(op, paddings);
}

COMMON_INFER_FUNC_REG(PadV3, PadV3InferShape);
// ----------------PadV3 Op End-------------------

// ----------------Fill Op Begin-------------------
template <typename T>
static void CaclDims(const GeTensorPtr& data, std::vector<int64_t>& vec_dim) {
  int32_t size = data->GetData().GetSize() / sizeof(T);
  for (int32_t i = 0; i < size; i++) {
    void* data_ptr = (void*)data->GetData().GetData();
    if (data_ptr == nullptr) {
      return;
    }
    T dim = *((T*)data_ptr + i);
    vec_dim.push_back(dim);
  }
}

template <typename T>
static void CaclDims(const Tensor& data, std::vector<int64_t>& vec_dim) {
  int32_t size = data.GetSize() / sizeof(T);
  for (int32_t i = 0; i < size; i++) {
    T dim = *((T*)data.GetData() + i);
    vec_dim.push_back(dim);
  }
}

IMPLEMT_COMMON_INFERFUNC(FillInferShape) {
  GeTensorPtr data;
  std::vector<int64_t> vec_dim;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends({"dims"});

  auto node = NodeUtils::GetNodeFromOperator(op);
  if (node == nullptr) {
    OP_LOGE(op.GetName().c_str(), "get null node ptr");
    return GRAPH_PARAM_INVALID;
  }

  TensorDesc td = op.GetOutputDesc("y");

  if (NodeUtils::GetInputConstData(node, "dims", data) != GRAPH_SUCCESS) {
    GE_OP_LOGW(op.GetName().c_str(), "Get constValue failed of [dims]");
    auto shape = op.GetInputDesc("dims").GetShape();
    int64_t dim_value;
    dim_value = shape.GetDim(0);
    std::vector<std::pair<int64_t, int64_t>> range_output;
    std::vector<std::pair<int64_t, int64_t>> range_input;
    range_input.push_back(std::make_pair(1, -1));
    op.GetInputDesc("dims").SetShapeRange(range_input);
    for (int64_t m = 0; m < dim_value; m++) {
      vec_dim.push_back(-1);
      range_output.push_back(std::make_pair(1, -1));
    }
    if (vec_dim.empty()) {
      vec_dim.push_back(-2);
    }
    for (uint64_t i = 0; i < vec_dim.size(); i++) {
      OP_LOGD(op.GetName().c_str(), "fill no const infershape dims value [%d] is [%d]", i, vec_dim[i]);
    }
    OP_LOGD(op.GetName().c_str(), "fill no const infershape dims value done");
    td.SetShape(Shape(vec_dim));
    td.SetDataType(op.GetInputDesc("value").GetDataType());
    td.SetShapeRange(range_output);

    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
  } else {
    NodeUtils::GetInputConstData(node, "dims", data);
    DataType data_type = data->GetTensorDesc().GetDataType();
    std::vector<int64_t> vec_dim;
    if (data_type == DT_INT32) {
      CaclDims<int32_t>(data, vec_dim);
    } else if (data_type == DT_INT64) {
      CaclDims<int64_t>(data, vec_dim);
    } else {
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), "const dtype", "it must DT_INT32 or DT_INT64");
      GE_OP_LOGE(op.GetName().c_str(), "Get constValue failed of [dims], the dtype must DT_INT32 or DT_INT64");
      return GRAPH_PARAM_INVALID;
    }

    int64_t fused_output = std::accumulate(vec_dim.begin(), vec_dim.end(), 1, std::multiplies<int64_t>());
    OP_LOGD(op.GetName().c_str(), "fused_output dims value done [%d]", fused_output);
    std::vector<std::pair<int64_t, int64_t>> range_input;
    std::vector<std::pair<int64_t, int64_t>> range_output;
    range_input.push_back(std::make_pair(fused_output, fused_output));

    td.SetShape(Shape(vec_dim));
    td.SetDataType(op.GetInputDesc("value").GetDataType());
    auto status = op.GetInputDesc("dims").SetShapeRange(range_input);

    for (auto& dim_val : vec_dim) {
      range_output.push_back(std::make_pair(dim_val, dim_val));
    }

    if (status != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
    td.SetShapeRange(range_output);

    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
  }
}

COMMON_INFER_FUNC_REG(Fill, FillInferShape);
// ----------------Fill Op End-------------------

// ----------------FillD Op Begin-------------------
IMPLEMT_COMMON_INFERFUNC(FillDInferShape) {
  std::vector<int64_t> vec_dim;
  if (ge::GRAPH_SUCCESS != op.GetAttr("dims", vec_dim)) {
    OpsGetAttrErrReport(op.GetName(), "dims");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr failed of FillD!");
    return GRAPH_FAILED;
  }

  if (vec_dim.size() < DIM_SIZE1 || vec_dim.size() > DIM_SIZE8) {
    OpsInputShapeDimErrReport(op.GetName(), "dims", ConcatString(DIM_SIZE8), ConcatString(DIM_SIZE1),
                              ConcatString(vec_dim.size()));
    OP_LOGE(op.GetName().c_str(), "dims must be between 1 and 8.");
    return GRAPH_FAILED;
  }

  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(Shape(vec_dim));
  td.SetDataType(op.GetInputDesc("value").GetDataType());

  (void)op.UpdateOutputDesc("y", td);
  OP_LOGI(op.GetName().c_str(), "infershape success");
  return GRAPH_SUCCESS;
 }

COMMON_INFER_FUNC_REG(FillD, FillDInferShape);
// ----------------FillD Op End-------------------

// -------------------BroadcastTo-----------------------
IMPLEMT_INFERFUNC(BroadcastTo, BroadcastToInferShape) {
  Tensor data;
  if (op.GetInputConstData("shape", data) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "Get constValue failed of [shape]");
    Shape ashape = op.GetInputDesc("shape").GetShape();
    std::vector<int64_t> shapedims = ashape.GetDims();
    size_t dim_num = ashape.GetDimNum();

    DataType input_dtype = op.GetInputDesc("x").GetDataType();

    if (dim_num > 1) {
      OP_LOGE(op.GetName().c_str(), "The dim numbles of constnode are less than one.");
      return GRAPH_FAILED;
    }

    std::vector<int64_t> shape_vector;
    std::vector<std::pair<int64_t, int64_t>> range_vector;
    for (int64_t item = 0; item < shapedims[0]; ++item) {
      shape_vector.push_back(-1);
      range_vector.push_back(std::make_pair(1, -1));
    }
    Shape input_shape(shape_vector);

    TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetShape(input_shape);
    output_desc.SetShapeRange(range_vector);
    output_desc.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("y", output_desc);

    return GRAPH_SUCCESS;
  }

  DataType data_type = data.GetTensorDesc().GetDataType();
  std::vector<int64_t> vec_dim;
  if (data_type == DT_INT32) {
    CaclDims<int32_t>(data, vec_dim);
  } else if (data_type == DT_INT64) {
    CaclDims<int64_t>(data, vec_dim);
  } else {
    return GRAPH_PARAM_INVALID;
  }
  std::vector<std::pair<int64_t, int64_t>> range_output;
  for (auto& dim_val : vec_dim){
    range_output.push_back(std::make_pair(dim_val, dim_val));
  }
  OP_LOGI(op.GetName().c_str(), "the op infer shape and dtype");
  DataType input_dtype = op.GetInputDesc("x").GetDataType();

  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(Shape(vec_dim));
  td.SetShapeRange(range_output);
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BroadcastTo, BroadcastToInferShape);
// --------------------BroadcastTo END-----------------------

// ------------------BroadcastToD------------------------
IMPLEMT_INFERFUNC(BroadcastToD, BroadcastToDInferShape) {
  OP_LOGI(op.GetName().c_str(), "the op infer shape and dtype");
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  std::vector<int64_t> shape_out;
  if (ge::GRAPH_SUCCESS != op.GetAttr("shape", shape_out)) {
    OpsGetAttrErrReport(op.GetName(), "shape");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr failed of BroadcastToD!");
    return GRAPH_FAILED;
  }
  if (shape_out.size() < DIM_SIZE1 || shape_out.size() > DIM_SIZE8) {
    OpsInputShapeDimErrReport(op.GetName(), "shape", ConcatString(DIM_SIZE8), ConcatString(DIM_SIZE1),
                              ConcatString(shape_out.size()));
    OP_LOGE(op.GetName().c_str(), "shape must be between 1 and 8.");
    return GRAPH_FAILED;
  }
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(shape_out));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(BroadcastToD, BroadcastToDInferShape);
// ----------------BroadcastToD END-------------------

// ---------------------DiagD-------------------------
COMMON_INFER_FUNC_REG(DiagD, ELMTWISE_INFER_SHAPEANDTYPE("assist", "y"));
// ---------------------DiagD_End---------------------

// ---------------------Diag--------------------------
IMPLEMT_COMMON_INFERFUNC(DiagInferShape) {
  Shape shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  vector<int64_t> dimInfo = shape.GetDims();
  vector<int64_t> assitDimInfo;
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < dimInfo.size(); ++j) {
      assitDimInfo.push_back(dimInfo[j]);
    }
  }

  shape = Shape(assitDimInfo);
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Diag, DiagInferShape);
// ---------------------Diag END-------------------------------------

// ---------------------AscendPadding-------------------------------------
IMPLEMT_COMMON_INFERFUNC(AscendPaddingInferShape) {
  Shape x_shape;
  auto x_desc = op.GetInputDesc(0);
  if (WithRankAtLeast(x_desc, 2, x_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input x rank must be at least 2, real rank is %lld.", x_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  auto x_rank = x_shape.GetDimNum();
  auto x_dims = x_shape.GetDims();

  if (x_dims[x_rank - 1] != 1) {
    OP_LOGE(op.GetName().c_str(), "the last dim of x must be 1, real dim is %lld.", x_dims[x_rank - 1]);
    return GRAPH_FAILED;
  }

  int32_t pad_dim_size;
  if (op.GetAttr("pad_dim_size", pad_dim_size) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr pad_dim_size error.");
    return GRAPH_FAILED;
  }
  if (pad_dim_size < 1) {
    OP_LOGE(op.GetName().c_str(), "pad_dim_size should be a positive value, real value is %d.", pad_dim_size);
    return GRAPH_PARAM_INVALID;
  }

  if (ReplaceDim(x_shape, x_rank - 1, pad_dim_size, x_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "failed to create y shape.");
    return GRAPH_FAILED;
  }
  auto y_desc = op.GetOutputDesc(0);
  y_desc.SetShape(x_shape);
  y_desc.SetDataType(x_desc.GetDataType());
  (void)op.UpdateOutputDesc("y", y_desc);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AscendPadding, AscendPaddingInferShape);
// ---------------------AscendPadding END-------------------------------------

// ---------------------EmbdingRankId-------------------------------------
IMPLEMT_COMMON_INFERFUNC(EmbeddingRankIdInferShape) {
  Shape addr_shape;
  auto addr_desc = op.GetInputDesc(0);
  if (WithRank(addr_desc, 2, addr_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input addr_table rank must be at least 2, real rank is %lld.",
            addr_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }
  auto addr_rank = addr_shape.GetDimNum();
  auto addr_dims = addr_shape.GetDims();

  if (addr_dims[addr_rank - 1] != 3) {
    OP_LOGE(op.GetName().c_str(), "the last dim of addr_table must be 3, real dim is %lld.", addr_dims[addr_rank - 1]);
    return GRAPH_FAILED;
  }
  if (addr_dims[0] <= 0) {
    OP_LOGE(op.GetName().c_str(), "the first dim of addr_table must be >0, real dim is %lld.", addr_dims[0]);
    return GRAPH_FAILED;
  }
  Shape index_shape;
  auto index_desc = op.GetInputDesc(1);
  if (WithRank(index_desc, 1, index_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input index rank must be at least 1, real rank is %lld.",
            index_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  int32_t row_memory;
  if (op.GetAttr("row_memory", row_memory) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr row_memory error.");
    return GRAPH_FAILED;
  }
  if (row_memory <= 0) {
    OP_LOGE(op.GetName().c_str(), "row_memory should be >0 , real value is %d.", row_memory);
    return GRAPH_PARAM_INVALID;
  }

  Shape out_shape;
  std::vector<int64_t> dims = index_shape.GetDims();
  if (ReplaceDim(addr_shape, 0, dims[0], out_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "failed to create rank_id shape.");
    return GRAPH_FAILED;
  }

  auto rankid_desc = op.GetOutputDesc(0);
  rankid_desc.SetShape(out_shape);
  rankid_desc.SetDataType(DT_UINT64);
  (void)op.UpdateOutputDesc("rank_id", rankid_desc);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(EmbeddingRankId, EmbeddingRankIdInferShape);
// ---------------------EmbeddingRankId END-------------------------------------

// ----------------FillV2 Begin-------------------
IMPLEMT_INFERFUNC(FillV2, FillV2InferShape) {
  Tensor data;
  std::vector<int64_t> vec_dim;
  TensorDesc td = op.GetOutputDesc("y");
  if (op.GetInputConstData("dims", data) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get constValue failed of [dims]");
    auto shape = op.GetInputDesc("dims").GetShape();
    int64_t dimValue;
    dimValue = shape.GetDim(0);
    for (int64_t m = 0; m < dimValue; m++) {
      vec_dim.push_back(-1);
    }
    td.SetShape(Shape(vec_dim));
    td.SetDataType(op.GetInputDesc("value").GetDataType());
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
  } else {
    DataType dataType = data.GetTensorDesc().GetDataType();
    std::vector<int64_t> vec_dim;
    if (dataType == DT_INT32) {
      CaclDims<int32_t>(data, vec_dim);
    } else if (dataType == DT_INT64) {
      CaclDims<int64_t>(data, vec_dim);
    } else {
      OP_LOGE(op.GetName().c_str(), "data type not supported");
      return GRAPH_PARAM_INVALID;
    }
    td.SetShape(Shape(vec_dim));
    td.SetDataType(op.GetInputDesc("value").GetDataType());
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
  }
}

// Registered inferfunction
INFER_FUNC_REG(FillV2, FillV2InferShape);
// ----------------FillV2 END---------------------

// ----------------FillV2D Begin-------------------
IMPLEMT_INFERFUNC(FillV2D, FillV2DInferShape) {
  const int DIM_SIZE1 = 1;
  const int DIM_SIZE8 = 8;
  std::vector<int64_t> vec_dim;
  if (ge::GRAPH_SUCCESS != op.GetAttr("dims", vec_dim)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr failed of FillD!");
    return GRAPH_FAILED;
  }

  OP_LOGI(op.GetName().c_str(), "start infershape");

  if (vec_dim.size() < DIM_SIZE1 || vec_dim.size() > DIM_SIZE8) {
    OP_LOGE(op.GetName().c_str(), "dims must between 1 and 8.");
    return GRAPH_FAILED;
  }

  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(Shape(vec_dim));
  td.SetDataType(DT_FLOAT);

  op.UpdateOutputDesc("y", td);
  OP_LOGI(op.GetName().c_str(), "infershape success!");
  return GRAPH_SUCCESS;
}

// Registered inferfunction
INFER_FUNC_REG(FillV2D, FillV2DInferShape);
// ----------------FillV2D END---------------------

}  // namespace ge
