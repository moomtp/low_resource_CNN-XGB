from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass, field

from gen_behavior import Gen_behavior, Gen_leaf_node, Gen_tree_node_float16

@dataclass
class Node():
    node_val:float=0.0
    jump_addr:int=-1
    node_id:int=-1

    # leaf node poperty
    # is_last_tree:bool=False
    feature_idx:int=-1

    # left_node:Optional['Node'] = None
    # right_node:Optional['Node'] = None
    # parent_node:Optional['Node'] = None

    gen_behaivor:Gen_behavior = Gen_behavior()


    def __init__(self, node_val:float, jump_addr:int, node_id:int) -> None:
        self.node_val = node_val
        self.jump_addr = jump_addr
        self.node_id = node_id

    # generate FPGA bit string
    def gen(self) -> str:
        return self.gen_behaivor.gen(self, self)

    @abstractmethod
    def is_tree_node():
        pass
    
    # only LeafNode will implement this func
    def is_last_tree(self) -> bool:
        # if self.__class__ is Node or not hasattr(self, '_function_implemented'):
        raise NotImplementedError("function is_last_tree() must be implemented in LeafNode.")
        # else:
        #     raise RuntimeError("function is_last_tree() is incorrectly implemented in LeafNode.")

    
    # only TreeNode will implement this func
    def get_feature_idx(self):
        # if self.__class__ is Node or not hasattr(self, '_function_A_implemented'):
        raise NotImplementedError("get_feature_idx must be implemented in TreeNode.")
        # else:
        #     raise RuntimeError("get_geature_idx is incorrectly implemented in TreeNode.")

    # @left_node.setter
    # def left_node(self, node:Optional['Node']):
    #     self.left_node = node
    # @right_node.setter
    # def right_node(self, node:Optional['Node']):
    #     self.right_node = node



@dataclass
class LeafNode(Node):
    # jump addr -> to next tree
    # is_last_tree:bool
    gen_behaivor:Gen_behavior = Gen_leaf_node

    def __init__(self, node_val:float, is_last_tree:bool, next_tree_addr:int, node_id:int) -> None:
        super().__init__(node_val, next_tree_addr, node_id)
        assert 0 < next_tree_addr < 2**14, "next tree addr must be greater than 0 and less than 2^14"
        self.is_last_tree = is_last_tree
    # override
    def is_last_tree(self) -> bool:
        return self.is_last_tree
    def is_tree_node():
        return False


@dataclass
class TreeNode(Node):
    # jump addr -> to right child
    gen_behaivor:Gen_behavior = Gen_tree_node_float16
    feature_idx:int = -1
    
    def __init__(self, node_val:float, addr_to_right_child:int, feature_idx:int, node_id:int) -> None:
        super().__init__(node_val, addr_to_right_child, node_id)
        # assert 0 < @relright_child < 2^7
        assert 0 < addr_to_right_child < 2**7, "addr format error"
        # assert 0 < feature < 2^8
        assert 0 <= feature_idx < 2**8, "feature idx format error"
        self.feature_idx = feature_idx

    def is_tree_node():
        return True

    # override
    def get_feature_idx(self) -> int :
        return self.feature_idx


