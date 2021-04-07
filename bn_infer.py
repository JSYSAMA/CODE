# Copyright 2019 Huawei Technologies Co., Ltd
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
bn_infer
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


def _check_shape(shape_x, shape_scale):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_scale: list or tuple
        scale's data shape

    Returns
    -------
    None
    """
    para_check.check_shape(shape_x, param_name="x")

    para_check.check_shape(shape_scale, param_name="scale")

    if len(shape_x) != 5 or len(shape_scale) != 5:
        error_detail = "The data format is 5HD, but x or scale's shape length is not 5"
        error_manager_vector.raise_err_two_input_shape_invalid("bn_infer", "x", \
                                                               "scale", error_detail)

    dim_c1 = shape_x[1]
    dim_c0 = shape_x[4]

    if shape_scale[1] != dim_c1 or shape_scale[4] != dim_c0:
        error_detail = "Dimension C must be equal"
        error_manager_vector.raise_err_input_shape_invalid("bn_infer", "scale", \
                                                           error_detail)


def _check_dtype(dtype_x, dtype_scale, dtype_offset,
                 dtype_mean, dtype_variance):
    """
    Function to check if the dtype is in line with norms.

    Parameters
    ----------
    dtype_x: str
        x's data type
    dtype_scale: str
        scale's data type
    dtype_offset: str
        offset's data type
    dtype_mean: str
        mean's data type
    dtype_variance: str
        variance's data type

    Returns
    -------
    None
    """
    para_check.check_dtype(dtype_x.lower(), ("float16", "float32"), param_name="x")
    para_check.check_dtype(dtype_scale.lower(), ("float32", "float16"), param_name="scale")
    para_check.check_dtype(dtype_offset.lower(), ("float32", "float16"), param_name="offset")
    para_check.check_dtype(dtype_mean.lower(), ("float32", "float16"), param_name="mean")
    para_check.check_dtype(dtype_variance.lower(), ("float32", "float16"), param_name="variance")


# pylint: disable=locally-disabled,invalid-name,too-many-arguments
# pylint: disable=locally-disabled,too-many-locals,unused-argument
@register_operator_compute("BNInfer", op_mode="dynamic", support_fusion=True)
def bn_infer_compute(x, scale, offset, mean, variance,
                     y, epsilon, kernel_name="bn_inf"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization for inference

    Parameters
    ----------
    x: TVM tensor
        contains x data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    mean: TVM tensor
        contains mean data
    variance: TVM tensor
        contains variance data
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v2"

    Returns
    -------
    res: TVM tensor
        the result of bn_training_update_v2 compute for inference
    """
    shape_x = shape_util.shape_to_list(x.shape)
    shape_scale = shape_util.shape_to_list(scale.shape)
    shape_offset = shape_util.shape_to_list(offset.shape)
    shape_mean = shape_util.shape_to_list(mean.shape)
    shape_variance = shape_util.shape_to_list(variance.shape)
    
    shape_x, shape_scale, shape_offset, shape_mean, shape_variance, shape_max = shape_util.unify_broadcast_shapes(
        [shape_x, shape_scale, shape_offset, shape_mean, shape_variance])

    # compute the oefficient of y
    x = tbe.broadcast(x, shape_max)
    scale = tbe.broadcast(scale, shape_max)
    offset = tbe.broadcast(offset, shape_max)
    mean = tbe.broadcast(mean, shape_max)
    variance = tbe.broadcast(variance,shape_max)

    multiplier_add = tbe.vadds(variance, epsilon)
    multiplier_sqrt = tbe.vsqrt(multiplier_add)
    multiplier_div = tbe.vdiv(scale, multiplier_sqrt)

    addend_mul = tbe.vmul(multiplier_div, mean)
    addend_sub = tbe.vsub(offset, addend_mul)

    # compute the batch normalization of x
    is_cast = False
    if x.dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vmul", "float32"):
        is_cast = True
        x = tbe.cast_to(x, "float32")
        multiplier_div = tbe.cast_to(multiplier_div, "float32")
        addend_sub = tbe.cast_to(addend_sub, "float32")

    res = tbe.vadd(tbe.vmul(multiplier_div, x), addend_sub)
    if is_cast:
        res = tbe.cast_to(res, "float16")

    return res


@register_operator("BNInfer")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.KERNEL_NAME)
def bn_infer(x, scale, offset, mean, variance, y,
             epsilon, kernel_name="bn_infer"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization for inference

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    scale: dict
        dict of scale, A 5HD Tensor for scale.
    offset: dict
        dict of offset, A 5HD Tensor for offset.
    mean: dict
        dict of mean, A `Tensor`.
        dict of scale, A 5HD Tensor for mean.
    variance: dict
        dict of batch_variance, A `Tensor`.
        dict of offset, A 5HD Tensor for variance.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_infer"

    Returns
    -------
    None
    """

    dtype_x = x.get("dtype").lower()
    dtype_scale = scale.get("dtype").lower()
    dtype_offset = offset.get("dtype").lower()
    dtype_mean = mean.get("dtype").lower()
    dtype_variance = variance.get("dtype").lower()

    data_format = x.get("format").upper()

    if data_format not in ("NC1HWC0",):
        format_rule = "Format only support 5HD"
        error_manager_vector.raise_err_check_params_rules("bn_infer", format_rule, "x", data_format)
    _check_dtype(dtype_x, dtype_scale, dtype_offset, dtype_mean, dtype_variance)

    ins = classify([x, scale, offset, mean, variance], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_x, _scale, _offset, _mean, _variance) in ins:
        with tbe.compute():
            shape_list = shape_util.variable_shape([_x, _scale, _offset, _mean, _variance])
            x_input = tvm.placeholder(shape_list[0], name="x_input", dtype=dtype_x.lower())
            scale_input = tvm.placeholder(shape_list[1], name="scale_input", dtype=dtype_scale.lower())
            offset_input = tvm.placeholder(shape_list[2], name="offset_input", dtype=dtype_offset.lower())
            mean_input = tvm.placeholder(shape_list[3], name="mean_input", dtype=dtype_mean.lower())
            variance_input = tvm.placeholder(shape_list[4], name="variance_input", dtype=dtype_variance.lower())
            res = bn_infer_compute(x_input, scale_input, offset_input, mean_input, variance_input,
                                   y, epsilon, kernel_name=kernel_name)
            tensors.append([x_input, scale_input, offset_input, mean_input, variance_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
