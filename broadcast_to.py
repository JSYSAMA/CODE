# Copyright 2020 Huawei Technologies Co., Ltd
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
dynamic broadcast_to
"""
import te.lang.cce as tbe
import te.lang.base as tbe_base
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector
from te import tvm
from impl.util.platform_adapter import register_operator


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
def broadcast_to_compute(x, shape, y, kernel_name="broadcast_to"):
    """
    TVM calculation process, used for fusion operation.

    Parameters
    ----------
    x: list of placeholders.
        Input data.
    shape : list or tuple.
        Number of the axis replicates.
    y: dict.
        dict of output.

    kernel_name : str.
        Cce kernel name, default value is "broadcast_to_d".

    Returns
    -------
    res
    """
    res = tbe.broadcast(x, shape)

    return res


# pylint: disable=too-many-locals,too-many-statements
@register_operator("BroadcastTo")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def broadcast_to(x, shape, y, kernel_name="broadcast_to"):
    """algorithm: broadcast_to.
    The broadcast_to in tensorflow can multiple the shape of the given tensor.
    For example, tiling [a b c d] by [2] produces [a b c d a b c d].
    The broadcast_to op in TBE is compatible with the tensorflow operator BroadcastTo
    Abnormal condition:
    1. The length of shape must be equal to or less than the shape of multiples.

    Parameters
    ----------
    x : dict
        shape and dtype of input
    shape : dict
        shape and dtype of multiples
    y: dict
        dict of output.
    kernel_name : str.
        kernel name, default value is "broadcast_to".

    Returns
    -------
    None
    """

    input_x_dtype = x.get("dtype").lower()
    input_shape_dtype = shape.get("dtype").lower()

    input_x_shape = list(x.get("shape"))
    input_shape_shape = list(shape.get("shape"))

    check_list = ('float16', 'float32', 'int8', 'uint8', 'int32')
    para_check.check_dtype(input_x_dtype, check_list, param_name = "x")
    check_list = ('int32', 'int64')
    para_check.check_dtype(input_shape_dtype, check_list, param_name = "shape")

    if len(input_shape_shape) > 1:
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "shape", "shape should be 1D")
    
    input_x_range = list(x.get("range"))
    dims_value = input_shape_shape[0]

    if dims_value < -1:
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "shape", "shape[0] should be more than -1")
    
    if dims_value == -1:
        dims_value = len(input_x_shape)
    
    if len(input_x_shape) > dims_value:
        len_diff = dims_value - len(input_x_shape)
        input_x_shape = [1] * len_diff + input_x_shape
        input_x_range = [(1, 1)] * len_diff + input_x_range
    
    x_shape_adapt = []
    shape_shape_adapt = []
    x_range_adapt = []
    shape_range_adapt = []

    for shape_i, range_i in zip(input_x_shape, input_x_range):
        if shape_i == 1:
            x_shape_adapt.append(1)
            x_range_adapt.append((1, 1))
            shape_shape_adapt.append(-1)
            shape_range_adapt.append((1, None))
        else:
            x_shape_adapt.append(shape_i)
            x_range_adapt.append(range_i)
            shape_shape_adapt.append(shape_i)
            shape_range_adapt.append(range_i)
    
    x["shape"] = x_shape_adapt
    x["range"] = x_range_adapt
    shape["shape"] = shape_shape_adapt
    shape["range"] = shape_range_adapt

    schedules, tensors = [], []

    with tbe_base.compute():
        shape_shape, shape_x = shape_util.variable_shape([shape, x], support_broadcast=True)
        shape_input = tvm.placeholder(shape_shape, name="shape_input", dtype=input_shape_dtype)
        x_input = tvm.placeholder(shape_x, name="x_input", dtype=input_x_dtype)
        res = broadcast_to_compute(x_input, shape_shape, y, kernel_name=kernel_name)
        tensors.append([x_input, shape_input, res])
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)

    