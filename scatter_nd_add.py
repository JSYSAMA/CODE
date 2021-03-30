"""
scatter_nd_add
"""
from te import tik
from te import platform as tbe_platform
import te.lang.dynamic
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from te.utils.error_manager import register_operator

# max int64 value
MAX_INT64_VALUE = 2 ** 64 - 1
# tiling param num
TILING_ARG_NUM = 24
# reserved ub size
RESERVED_UB_SIZE = 8 * 1024
# 8 bit
EIGHT_BIT = 8
# bytes of one block
BLOCK_BYTES = 32

# pylint: disable=too-many-arguments,too-many-instance-attributes,unused-argument,invalid-name
class ScatterNdAdd():
    """
    Function: use to store scatter_nd_add base parameters
    Modify: 2020-10-29
    """
    def __init__(self, var, indices, adds, var_out, use_locking, kernel_name):
        """
        Init ScatterNdAdd parameters

        Paramters
        _________
        var: dict
            the dict of input tensor.
        indices: dict
            the dict of input tensor.
        add: dict
            the dict of input tensor.
        var_out: dict
            the dict of output tensor.
        use_locking: bool
            not used in this compute, default value is "False".
        kernel_name: str
            cce kernel name, default value is "scatter_nd_add"

        Returns
        _______
        None.
        enable_const_fold
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        self.var_dtype = var.get("dtype").lower()
        self.indice_dtype = indices.get("dtype").lower()
        self.adds_dtype = adds.get("dtype").lower()
        self.out_dtype = var_out.get("dtype").lower()

        self.check_input_params()

        self.ai_core_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        self.ub_size_bytes = (tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.UB_SIZE) - RESERVED_UB_SIZE)
        self.var_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(
            self.var_dtype) // EIGHT_BIT
        self.indices_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(
            self.indice_dtype) // EIGHT_BIT
        self.var_data_each_block = BLOCK_BYTES // self.var_dtype_bytes_size
        self.indices_data_each_block = BLOCK_BYTES // self.indices_dtype_bytes_size

        self.adds_ub_num = self.ub_size_bytes // 96 * 32 // self.var_dtype_bytes_size
        self.indices_ub_num = self.ub_size_bytes // 96 * 32 // self.indices_dtype_bytes_size
        self.tiling_gm = self.tik_instance.Tensor("int64", (TILING_ARG_NUM, ), name="tiling_gm", scope=tik.scope_gm)
        self.var_gm = self.tik_instance.Tensor(self.var_dtype, (MAX_INT64_VALUE, ), name="var_gm", scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor("int32", (MAX_INT64_VALUE, ), name="indices_gm", scope=tik.scope_gm)
        self.adds_gm = self.tik_instance.Tensor(self.var_dtype, (MAX_INT64_VALUE, ), name="adds_gm", scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.var_dtype, (MAX_INT64_VALUE, ), name="out_gm", scope=tik.scope_gm)

        self.adds_ub = None
        self.var_ub = None
        self.indices_ub = None
        self.var_tile_ub = None
        self.adds_tile_ub = None
        self.var_vconv_ub = None
        self.adds_vconv_ub = None
        self.tiling_ub = None

        self.var_read_index = None
        self.core_loop_index = None

        self.update_value = None
        self.indices_burst_len = None
        self.adds_burst_len = None
        self.tiling_mode = None
        self.indice_step = None
        self.core_num = None
        self.update_data_num = None

        self.indices_loop_num = None
        self.indices_last_num = None

        self.adds_num = None
        self.adds_loop_num = None
        self.adds_last_num = None
    
    def tiling_args(self):
        """
        get runtime params from tiling

        Parameters
        __________
        tiling_ub: tensor, runtime params from scatter_nd_add tiling

        Returns
        _______
        None
        """
        self.tiling_mode = self.tik_instance.Scalar("int64", name="tiling_mode")
        self.indices_step = self.tik_instance.Scalar("int64", name="indice_step")
        self.core_num = self.tik_instance.Scalar("int64", name="core_num")
        self.update_data_num = self.tik_instance.Scalar("int64", name="update_data_num")
        self.indices_loop_num = self.tik_instance.Scalar("int64", name="indices_loop_num")
        self.indices_last_num = self.tik_instance.Scalar("int64", name="indices_last_num")
        self.adds_num = self.tik_instance.Scalar("int64", name="adds_num")
        self.adds_loop_num = self.tik_instance.Scalar("int64", name="adds_loop_num")
        self.adds_last_num = self.tik_instance.Scalar("int64", name="adds_last_num")
        self.indices_last_dim = self.tik_instance.Scalar("int64", name="indices_last_dim")
        self.indicesFrontDim = self.tik_instance.Scalar("int64", name="indicesFrontDim")

        self.tiling_mode.set_as(self.tiling_ub[0])
        self.indice_step.set_as(self.tiling_ub[1])
        self.core_num.set_as(self.tiling_ub[2])
        self.update_data_num.set_as(self.tiling_ub[3])
        self.indices_loop_num.set_as(self.tiling_ub[4])
        self.indices_last_num.set_as(self.tiling_ub[5])
        self.adds_num.set_as(self.tiling_ub[6])
        self.adds_loop_num.set_as(self.tiling_ub[7])
        self.adds_last_num.set_as(self.tiling_ub[8])
        self.indices_last_num.set_as(self.tiling_ub[16])
        self.indicesFrontDim.set_as(self.tiling_ub[17])

    def init_ub_tensor(self):
        """
        Compute the ub size of tensors

        Parameters
        __________
        None

        Returns
        _______
        None
        """
        self.adds_ub = self.tik_instance.Tensor(self.var_dtype, (self.adds_ub_num,),
                                                name="adds_ub", scope=tik.scope_ubuf)
        self.var_ub = self.tik_instance.Tensor(self.var_dtype, (self.adds_ub_num,),
                                                name="adds_ub", scope=tik.scope_ubuf)
        self.indices_ub = self.tik_instance.Tensor("int32", (self.indices_ub_num,),
                                                name="indices_ub", scope=tik.scope_ubuf)
        self.var_tile_ub = self.tik_instance.Tensor(self.var_dtype, (self.var_data_each_block,),
                                                name="var_tile_ub", scope=tik.scope_ubuf)
        self.adds_tile_ub = self.tik_instance.Tensor(self.var_dtype, (self.var_data_each_block,),
                                                name="adds_tile_ub", scope=tik.scope_ubuf)
        self.var_vconv_ub = self.tik_instance.Tensor("float16", (32,),
                                                name="var_vconv_ub", scope=tik.scope_ubuf)
        self.adds_vconv_ub = self.tik_instance.Tensor("float16", (32,),
                                                name="adds_vconv_ub", scope=tik.scope_ubuf)

        self.var_read_index = self.tik_instance.Scalar("int64", name="var_read_index")
        self.var_read_index.set_as(0)
        self.core_loop_index = self.tik_instance.Scalar("int64", name="core_loop_index")
        self.core_loop_index.set_as(0)
        self.update_value = self.tik_instance.Scalar(self.var_dtype, name="update_value")
        self.update_value.set_as(0)
        self.indices_burst_len = self.tik_instance.Scalar("int64", name="indices_burst_len")
        self.indices_burst_len.set_as(0)
        self.adds_burst_len = self.tik_instance.Scalar("int64", name="adds_burst_len")
        self.adds_burst_len.set_as(0)
        

    def check_input_params(self):
        """
        Check whether the input parameters is valid or not
        """
        indices_support_dtype_list = ("int32",)
        var_support_dtype_list = ("float32", "float16", "int8", "uint8", "int32")
        para_check.check_dtype(self.indices_dtype, indices_support_dtype_list, param_name="indices")
        para_check.check_dtype(self.var_dtype, var_support_dtype_list, param_name="var")
        if self.var_dtype != self.adds_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(self.kernel_name, "add", "var",
                                                                  self.adds_dtype, self.var_dtype)
        if self.var_dtype != self.out_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(self.kernel_name, "out", "var",
                                                                  self.out_dtype, self.var_dtype)
    
    def calc_indices(self, indices_ub_index):
        """
        calculate indices on gm
        """
        self.var_read_index.set_as(0)
        with self.tik_instance.for_range(0, self.indices_last_num) as i:
            tiling_offset = self.tik_instance.Scalar("int64", name="tmp")
            adds_sum_index = self.tik_instance.Scalar("int32", name="add_summ_index")
            indices_sum_index = self.tik_instance.Scalar("int64", name="indices_sum_index")
            indices_sum_index.set_as(indices_ub_index * self.indices_last_dim + i)
            adds_sum_index.set_as(self.indices_ub[indices_sum_index])
            tiling_offset.set_as(self.tiling_ub[9 + i])
            self.var_read_index.set_as(self.var_read_index + adds_sum_index * tiling_offset)
    
    def move_indices(self, indices_in_index, indice_num):
        """
        Move indices, choose branch

        Parameters
        __________
        indices_in_index: int64
            indices index on GM
        indice_num: int64
            the number of indexes in the indices on UB
        
        Returns
        _______
        None
        """
        with self.tik_instance.if_scope(indice_num % self.indices_data_each_block == 0):
            self.indices_burst_len.set_as(indice_num // self.indices_data_each_block)
        with self.tik_instance.else_scope():
            self.indices_burst_len.set_as((indice_num // self.indices_data_each_block)+1)
        self.tik_instance.data_move(self.indices_ub, self.indices_gm[indices_in_index], 0, 1,
                                    self.indices_burst_len, 0, 0)

        with self.tik_instance.if_scope(self.tiling_mode == 1):
            self.traversing_adds_32b_aligned_and_ub_enough(indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 2):
            self.circulate_indices(indices_in_index, indice_num, 2)
        with self.tik_instance.if_scope(self.tiling_mode == 3):
            self.traversing_adds_32b_aligned_and_ub_enough(indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 4):
            self.traversing_adds_32b_aligned_and_ub_enough(indices_in_index, indice_num)
        with self.tik_instance.if_scope(self.tiling_mode == 5):
            self.circulate_indices(indices_in_index, indice_num, 5)
        
    def circulate_indices(self, indices_in_index, indice_num, mode):
        """
        Circulate the index in the indices

        Parameters
        __________
        indices_in_index: int64
            Indices index on GM
        indice_num: int64
            the number of indexes in the indices on UB
        
        Returns
        _______
        None
        """
        with self.tik_instance.for_range(0, self.indicesFrontDim) as indices_ub_index:
            with self.tik_instance.if_scope(indices_ub_index < indice_num // self.indices_last_dim):
                self.calc_indices(indices_ub_index)
                with self.tik_instance.if_scope(self.core_loop_index * self.indice_step <= self.var_read_index):
                    with self.tik_instance.if_scope((self.core_loop_index + 1) * self.indice_step > self.var_read_index):
                        self.traversing_adds(indices_ub_index, indices_in_index, mode)
    
    def traversing_adds(self, indices_ub_index, indices_in_index, mode):
        """
        Traversing the index in the adds

        Parameters
        __________
        indices_ub_index: int64
            Indices index on UB
        indice_in_index: int64
            Indices index on GM
        
        Returns
        _______
        None
        """
        with self.tik_instance.if_scope(self.adds_loop_num > 0):
            with self.tik_instance.for_range(0, self.adds_loop_num) as adds_loop_index:
                self.update_var((indices_in_index // self.indices_last_dim + indices_ub_index) * self.update_data_num + 
                                adds_loop_index * self.adds_ub_num, self.adds_ub_num, adds_loop_index * self.adds_ub_num +
                                self.var_read_index, mode)
        with self.tik_instance.if_scope(self.adds_last_num > 0):
            self.update_var((indices_in_index // self.indices_last_dim + indices_ub_index) * self.update_data_num +
                            self.adds_loop_num * self.adds_ub_num, self.adds_last_num, self.adds_loop_num * self.adds_ub_num +
                            self.var_read_index, mode)

    def vadd(self, update_num):
        mask = 256 // self.var_dtype_bytes_size
        loop = (update_num + mask - 1) // mask // 255
        last = (update_num + mask - 1) // mask % 255
        lastindex = loop * 255 + last - 1
        lastmaskScalar = self.tik_instance.Scalar("int64", name="lastmask")
        lastmask = update_num % mask
        lastmaskScalar.set_as(lastmask)
        with self.tik_instance.if_scope(lastmaskScalar == 0):
            lastmaskScalar.set_as(mask)
        
        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as i:
                self.tik_instance.vec_add(mask, self.adds_ub[i * 255 * mask], self.adds_ub[i * 255 * mask],
                                          self.var_ub[i * 255 * mask], 255, 8, 8, 8)
        with self.tik_instance.if_scope(last > 0):
            self.tik_instance.vec_add(mask, self.adds_ub[loop * 255 * mask], self.adds_ub[lastindex * mask],
                                      self.var_ub[loop * 255 * mask], last - 1, 8, 8, 8)
            self.tik_instance.vec_add(lastmaskScalar, self.adds_ub[lastindex * mask], self.adds_ub[lastindex * mask],
                                      self.var_ub[lastindex * mask], 1, 8, 8, 8)
    
    def vec_add_with_addr(self, update_num, adds_addr, var_addr):
        mask = 256 // self.var_dtype_bytes_size
        loop = (update_num + mask - 1) // mask // 255
        last = (update_num + mask - 1) // mask % 255
        lastindex = loop * 255 + last - 1
        lastmaskScalar = self.tik_instance.Scalar("int64", name="lastmask")
        lastmask = update_num % mask
        lastmaskScalar.set_as(lastmask)
        with self.tik_instance.if_scope(lastmaskScalar == 0):
            lastmaskScalar.set_as(mask)
        
        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as i:
                self.tik_instance.vec_add(mask, self.adds_ub[i * 255 * mask + adds_addr], self.adds_ub[i * 255 * mask + adds_addr],
                                          self.var_ub[i * 255 * mask + var_addr], 255, 8, 8, 8)
        with self.tik_instance.if_scope(last > 0):
            self.tik_instance.vec_add(mask, self.adds_ub[loop * 255 * mask + adds_addr], self.adds_ub[loop * 255 * mask + adds_addr],
                                      self.var_ub[loop * 255 * mask + var_addr], last - 1, 8, 8, 8)
            self.tik_instance.vec_add(lastmaskScalar, self.adds_ub[lastindex * mask + adds_addr], self.adds_ub[lastindex * mask + adds_addr],
                                      self.var_ub[lastindex * mask + var_addr], 1, 8, 8, 8)

    def update_var(self, adds_loop_index, update_num, var_loop_index, mode):
        """
        Update the update fragment corresponding to the index

        Parameters
        __________
        adds_loop_index: int64
            Update index on GM
        update_num: int64
            the number of indexes in the adds on UB
        var_loop_index: int64
            Var index on GM
        
        Returns
        _______
        None
        """
        if mode == 2:
            self.adds_burst_len.set_as(update_num // self.var_data_each_block)
            self.tik_instance.data_move(self.adds_ub, self.adds_gm[adds_loop_index],
                                        0, 1, self.adds_burst_len, 0, 0)
            self.tik_instance.data_move(self.var_ub, self.out_gm[var_loop_index], 0, 1,
                                        self.adds_burst_len, 0, 0)
            self.vadd(update_num)
            self.tik_instance.data_move(self.out_gm[var_loop_index], self.adds_ub, 0, 1,
                                        self.adds_burst_len, 0, 0)
        if mode == 5:
            with self.tik_instance.if_scope(update_num % self.var_data_each_block == 0):
                self.adds_burst_len.set_as(update_num // self.var_data_each_block)
            with self.tik_instance.else_scope():
                self.adds_burst_len.set_as(update_num // self.var_data_each_block + 1)
            self.tik_instance.data_move(self.adds_ub, self.adds_gm[adds_loop_index],
                                        0, 1, self.adds_burst_len, 0, 0)
            self.tik_instance.data_move(self.adds_tile_ub, self.adds_gm[adds_loop_index + update_num - self.var_data_each_block],
                                        0, 1, 1, 0, 0)
            with self.tik_instance.if_scope(update_num % self.var_data_each_block == 0):
                self.tik_instance.data_move(self.var_ub, self.out_gm[var_loop_index], 0 ,1,
                                            self.adds_burst_len, 0, 0)
                self.vadd(update_num)
                self.tik_instance.data_move(self.out_gm[var_loop_index], self.adds_ub, 0, 1,
                                            self.adds_burst_len, 0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.var_ub, self.out_gm[var_loop_index], 0, 1,
                                            self.adds_burst_len, 0, 0)
                self.tik_instance.data_move(self.var_tile_ub, self.out_gm[var_loop_index + update_num - self.var_data_each_block],
                                            0, 1, 1, 0, 0)
                self.vadd(update_num)
                self.tik_instance.vec_add(self.var_data_each_block, self.adds_tile_ub[0],
                                          self.adds_tile_ub[0], self.var_tile_ub[0], 1, 8, 8, 8)
                self.tik_instance.data_move(self.out_gm[var_loop_index + update_num - self.var_data_each_block],
                                            self.adds_tile_ub, 0, 1, 1, 0, 0)
                self.tik_instance.data_move(self.out_gm[var_loop_index], self.adds_ub, 0, 1,
                                            self.adds_burst_len - 1, 0, 0)

    def traversing_adds_32b_aligned_and_ub_enough(self, indices_in_index, indice_num):
        """
        updateDataNum is 32B aligned ub can store all addsNum

        Parameters
        __________
        indice_num: int64
            the number of indexes in the indices on UB
        Returns
        _______
        None
        """
        update_burst_len = self.adds_num // self.var_data_each_block
        self.tik_instance.data_move(self.adds_ub, self.adds_gm, 0, 1, update_burst_len, 0, 0)
        adds_burst_len = self.update_data_num // self.var_data_each_block
        with self.tik_instance.for_rang(0, self.indicesFrontDim) as indices_ub_index:
            self.calc_indices(indices_ub_index)
            with self.tik_instance.if_scope(self.core_loop_index * self.indice_step <= self.var_read_index):
                with self.tik_instance.if_scope((self.core_loop_index + 1) * self.indice_step > self.var_read_index):
                    self.tik_instance.data_move(self.var_ub, self.out_gm[self.var_read_index], 0, 1, adds_burst_len, 0, 0)
                    self.vec_add_with_addr(self.update_data_num, (indices_in_index + indices_ub_index) * self.update_data_num, 0)
                    self.tik_instance.data_move(self.out_gm[self.var_read_index],
                                                self.adds_ub[(indices_in_index + indices_ub_index) * self.update_data_num],
                                                0, 1, adds_burst_len, 0, 0)
    
    def traversing_adds_single_core_and_ub_enough(self, indices_in_index, indice_num):
        """
        updateDataNum isn't 32B aligned and less than 1 block, ub can store all addsNum

        Paramters
        _________
        indice_num: int64
            the number of indexes in the indices on UB
        Returns
        _______
        None
        """
        with self.tik_instance.if_scope(self.adds_num % self.var_data_each_block == 0):
            self.adds_burst_len.set_as(self.adds_num // self.var_data_each_block)
        with self.tik_instance.else_scope():
            self.adds_burst_len.set_as(self.adds_num // self.var_data_each_block + 1)
        self.tik_instance.data_move(self.adds_num // self.var_data_each_block + 1)
        with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.calc_indices(indices_ub_index)
            self.tik_instance.data_move(self.var_tile_ub, self.out_gm[self.var_read_index],
                                        0, 1, 1, 0, 0)
            with self.tik_instance.for_range(0, self.update_data_num) as adds_ub_index:
                self.update_value.set_as(self.adds_ub[(indices_ub_index + indices_in_index // self.indices_last_dim) *
                                                       self.update_data_num + adds_ub_index])
                self.adds_tile_ub[adds_ub_index].set_as(self.update_value)
            if self.var_dtype in ("int8", "uint8"):
                self.tik_instance.vec_conv(32, "", self.var_vconv_ub, self.var_tile_ub, 1, 8, 4)
                self.tik_instance.vec_conv(32, "", self.adds_vconv_ub, self.adds_tile_ub, 1, 8, 4)
                self.tik_instance.vec_add(self.update_data_num, self.var_vconv_ub, self.var_vconv_ub, self.adds_vconv_ub, 1, 8, 8, 8)
                self.tik_instance.vec_conv(32, "", self.var_tile_ub, self.var_vconv_ub, 1, 8, 4)
            else:
                self.tik_instance.vec_add(self.update_data_num, self.var_tile_ub, self.var_tile_ub, self.adds_tile_ub, 1, 8, 8, 8)
            self.tik_instance.data_move(self.out_gm[self.var_read_index], self.var_tile_ub, 0, 1, 1, 0, 0)

    def traversing_adds_single_core_and_ub_not_enough(self, indices_in_index, indice_num):
        """
        updateDataNum isn't 32B aligned and less than 1 block , ub can't store all addsNum

        Parameters
        __________
        indices_in_index: int64
            indices index on GM
        indice_num: int64
            the number of indexes in the indices on UB
        
        Returns
        _______
        None
        """
        with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.calc_indices(indices_ub_index)
            self.tik_instance.data_move(self.var_tile_ub, self.out_gm[self.var_read_index],
                                        0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.adds_tile_ub, self.adds_gm[(indices_in_index // self.indices_last_dim + indices_ub_index) * self.update_data_num],
                                        0, 1, 1, 0, 0)
            if self.var_dtype in ("int8", "uint8"):
                self.tik_instance.vec_conv(32, "", self.var_vconv_ub, self.var_tile_ub, 1, 8, 4)
                self.tik_instance.vec_conv(32, "", self.adds_vconv_ub, self.adds_tile_ub, 1, 8, 4)
                self.tik_instance.vec_add(self.update_data_num, self.var_vconv_ub, self.var_vconv_ub,
                                          self.adds_vconv_ub, 1, 8, 8, 8)
                self.tik_instance.vec_conv(32, "", self.var_tile_ub, self.var_vconv_ub, 1, 8, 4)
            else:
                self.tik_instance.vec_add(self.update_data_num, self.var_tile_ub, self.var_tile_ub,
                                          self.adds_tile_ub, 1, 8, 8, 8)
            self.tik_instance.data_move(self.out_gm[self.var_read_index], self.var_tile_ub, 0, 1, 1, 0, 0)
    
    def traversing_indices(self):
        """
        Traversing the index in the indices

        Parameters
        __________
        None

        Returns
        _______
        None
        """
        with self.tik_instance.if_scope(self.indices_loop_num > 0):
            with self.tik_instance.for_range(0, self.indices_loop_num) as indices_loop_index:
                self.move_indices(indices_loop_index * self.indices_ub_num, self.indices_ub_num)
        with self.tik_instance.if_scope(self.indices_last_num > 0):
            self.move_indices(self.indices_loop_num * self.indices_ub_num, self.indices_last_num)

    def scatter_nd_add_compute_tiling(self):
        """
        Main process of scatter_nd_add

        Parameters
        __________
        None

        Returns
        _______
        None
        """
        with self.tik_instance.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as core_index:
            self.tiling_ub = self.tik_instance.Tensor("int64", (TILING_ARG_NUM,), name="tiling_ub",
                                                      scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 6, 0, 0)
            self.tiling_args()
            with self.tik_instance.if_scope(self.core_num > 1):
                with self.tik_instance.if_scope(core_index < self.core_num):
                    self.init_ub_tensor()
                    self.core_loop_index.set_as(core_index)
                    self.traversing_indices()
            with self.tik_instance.else_scope():
                self.init_ub_tensor()
                self.traversing_indices()

    def scatter_nd_add_operator(self):
        """
        scatter_nd_add operation

        Parameters
        __________
        None

        Returns:
        _______
        compile info
        """
        self.scatter_nd_add_compute_tiling()
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.var_gm, self.indices_gm, self.adds_num),
                                   outputs=(self.out_gm), 
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)
        ub_size_bytes = self.ub_size_bytes // 96 * 64
        te.op.add_compile_info("var", 
                               {"ub_size": ub_size_bytes,
                                "core_num": self.ai_core_num,
                                "var_size": self.var_dtype_bytes_size,
                                "indices_size": self.indices_dtype_bytes_size})

#pylint: disable=unused-argument
@register_operator("ScatterNdAdd")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def scatter_nd_add(var, indices, adds, var_out, use_locking=False, kernel_name="scatter_nd_add"):
    """
    scatter_nd_add interface

    Parameters
    __________
    var: input var shape, dtype and range
    indices: input indices shape, dtype and range
    adds: input adds shape, dtype and range
    var_out: output shape, dtype and range
    use_locking: bool
    kernel_name: kernel_name of scatter_add op

    Returns
    _______
    compile info
    """
    obj = ScatterNdAdd(var, indices, adds, var_out, False, kernel_name)
    return obj.scatter_nd_add_operator()
