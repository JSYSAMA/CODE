# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
masked_fill
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# pylint: disable=invalid-name,unused-argument,unused-variable,too-many-locals
@register_operator_compute("MaskedFill", op_mode="dynamic", support_fusion=True)
def masked_fill_compute(x, mask, value, y, kernel_name="masked_fill"):
    """
    calculating masked_fill
    :param x: TVM tensor
                   the output of previous layer
    :param mask: TVM tensor
                    mask dtype is bool
    :param value: scalar or TVM tensor
                    the value to fill in with
    :param kernel_name: str
                    kernel name, default value is "masked_fill"
    :return:y
            TVM tensor
    """
    ori_dtype = x.dtype
    if x.dtype in ('int8',):
        x = tbe.cast_to(x, 'float16')
    target_dtype = x.dtype

    if x.dtype == 'int32':
        mask = tbe.cast_to(mask, 'float16')
    mask = tbe.cast_to(mask, x.dtype)

    if value.dtype != x.dtype:
        value = tbe.cast_to(value, x.dtype)
    
    x_shape = shape_util.shape_to_list(x.shape)
    mask_shape = shape_util.shape_to_list(mask.shape)
    value_shape = shape_util.shape_to_list(value.shape)
    # computer output shape
    x_shape, mask_shape, value_shape, target_shape = shape_util.unify_broadcast_shapes(
        [x_shape, mask_shape, value_shape])
    mask = tbe.broadcast(mask, target_shape)
    x = tbe.broadcast(x, target_shape)
    value = tbe.broadcast(value, target_shape)

    tensor_ones = tbe.broadcast(tvm.const(1, target_dtype), target_shape)

    if x.dtype == 'int32':
        tensor_mask_value = tbe.vmul(mask, value)
        tensor_mask_sub = tbe.vsub(tensor_ones, mask)
        tensor_x_mul = tbe.vmul(x, tensor_mask_sub)
        y = tbe.vadd(tensor_x_mul, tensor_mask_value)
        return y

    y = tbe.vcmpsel(mask, tensor_ones, 'eq', value, x)

    if y.dtype != ori_dtype:
        y = tbe.cast_to(y, ori_dtype)

    return y


@register_operator("MaskedFill")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def masked_fill(x, mask, value, y, kernel_name="masked_fill"):
    """
    :param x: dict
                    shape and dtype of tensor x input
    :param mask: dict
                    shape and dtype of tensor mask,
                    can be boardcast as shape as x
    :param value: dict
                    shape and dtype of value
    :param y: dict
                    the output of masked _fill
    :param kernel_name: str
                      kernel name, default value is "masked _fill"
    :return: none
    """

    x_shape = x.get("shape")
    x_dtype = x.get("dtype")
    x_dtype_lower = x_dtype.lower()
    x_range = list(x.get("range"))

    mask_shape = mask.get("shape")
    mask_dtype = mask.get("dtype")
    mask_dtype_lower = mask_dtype.lower()
    mask_range = list(mask.get("range"))

    value_shape = value.get("shape")
    value_dtype = value.get("dtype")
    value_dtype_lower = value_dtype.lower()
    value_range = list(value.get("range"))

    # check dtype
    x_dtype_list = ("float16", "float32", "int8", "int32")
    para_check.check_dtype(x_dtype, x_dtype_list)

    mask_dtype_list = ("bool", "int8")
    para_check.check_dtype(mask_dtype, mask_dtype_list)

    if mask_dtype == "bool":
        mask_dtype = "int8"

    value_dtype_list = ("float16", "float32", "int8", "int32")
    para_check.check_dtype(value_dtype, value_dtype_list)

    # check kernel_name
    para_check.check_kernel_name(kernel_name)

    pos_mask_shape = tuple([1] * (len(x_shape) - len(mask_shape))) + tuple(mask_shape)
    pos_mask_range = [[1, 1]] * (len(x_shape) - len(mask_shape)) + mask_range
    mask["shape"] = pos_mask_shape
    mask["range"] = pos_mask_range
    pos_value = tuple([1] * (len(x_shape) - len(value_shape))) + tuple(value_shape)
    pos_range = [[1, 1]] * (len(x_shape) - len(value_shape)) + value_range
    value["shape"] = pos_value
    value["range"] = pos_range

    ins = classify([x, mask, value], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_x, _mask, _value) in ins:
        with tbe.compute():
            shape_x, shape_mask, shape_value = shape_util.variable_shape([_x, _mask, _value])
            data_x = tvm.placeholder(shape_x, dtype=x_dtype_lower, name="data_x")
            data_mask = tvm.placeholder(shape_mask, dtype=mask_dtype_lower, name="data_mask")
            data_value = tvm.placeholder(shape_value, dtype=value_dtype_lower, name="data_value")
            res = masked_fill_compute(data_x, data_mask, data_value, kernel_name)
            tensors.append([data_x, data_mask, data_value, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors,
    }
    tbe.build(schedules, config)
