from abc import ABC, abstractmethod
from typing import Protocol
import struct
import numpy as np
import warnings

from digit_helper_function import *

class Gen_behavior():
    def __init__(self) -> None:
        pass
    # gen interface
    @abstractmethod
    def gen(self, _node):
        pass


# 
#   Leaf node representation:
#   |31    -     16|15    -     2|        1       |     0     |
#   |- pred_value -|- next_tree -|- is_last_tree -|- is_leaf -|


class Gen_leaf_node(Gen_behavior):
    # node val + accumulate of  nodes num & cur node num + node flag + class func
    def gen(self, _node):
        from node import Node
        last_tree_flag = "1" if _node.is_last_tree else "0"
        return  f_to_b(_node.node_val) + i_to_b(_node.jump_addr, 14) + last_tree_flag + "1"

class Gen_fix16_leaf_node(Gen_behavior):
    # node val + accumulate of  nodes num & cur node num + node flag + class func
    def gen(self, _node):
        from node import Node
        last_tree_flag = "1" if _node.is_last_tree else "0"
        return  f_to_fix16_b(_node.node_val) + i_to_b(_node.jump_addr, 14) + last_tree_flag + "1"

#   Non-leaf node representation:
#   |31     -     24|23    -     8|7      -      1|     0     |
#   |- num_feature -|- cmp_value -|- right_child -|- is_leaf -|
#   

class Gen_tree_node_float16(Gen_behavior):
    # feature idx + node val + @rel addr to right(val from tree) + class func
    def gen(self, _node):
        from node import Node
        # return i_to_b(_node.get_feature_idx(), 8) + i_to_b(_node.jump_addr, 7) + "1"
        return i_to_b(_node.get_feature_idx(), 8) + f_to_b(_node.node_val) + i_to_b(_node.jump_addr, 7) + "0"

class Gen_tree_node_fix_float16(Gen_behavior):
    # feature idx + node val + @rel addr to right(val from tree) + class func
    def gen(self, _node):
        from node import Node
        # return i_to_b(_node.get_feature_idx(), 8) + i_to_b(_node.jump_addr, 7) + "1"
        return i_to_b(_node.get_feature_idx(), 8) + f_to_fix16_b(_node.node_val) + i_to_b(_node.jump_addr, 7) + "0"
        
class Gen_tree_node_int16(Gen_behavior):
    pass

# special node's gen behavior
class Gen_non_func_leaf_node(Gen_behavior):
    def gen(self, _node):
        return "00000000000000000000000000000011"
class Gen_thread_node(Gen_behavior):
    def gen(self, _node):
        return "00000000000000000000000000000111"
        # return "00000007"
class Gen_class_end_node(Gen_behavior):
    def gen(self, _node):
        return "00000000000000000000000000001111"
        # return "0000000f"
class Gen_classes_end_node(Gen_behavior):
    def gen(self, _node):
        return "00000000000000000000000000011111"
        # return "0000001f"

# ============  sub function   =================



def f_to_b(val:float):
    """float_to_float16_binary"""
    if not(-65515 < val < 65515):
        warnings.warn("overflow occur in node's val")
    # bytes_representation = np.float16(val).tobytes() # binary format in little endian
    bytes_representation = np.float16(val).newbyteorder('big').tobytes() # binary format in big endian
    binary_format = ''.join(f'{byte:08b}' for byte in bytes_representation)[:16] 
    # print(binary_format)
    return binary_format

    # print(''.join(f'{c:08b}' for c in np.float16(val).tobytes()))
    # return ''.join(f'{c:08b}' for c in np.float16(val).tobytes())
    # print(struct.pack('f', value))
    # float16_value = struct.unpack('e', struct.pack('f', value))[0:1]
    # return bin(struct.unpack('H', struct.pack('e', float16_value))[0])[2:].zfill(16)