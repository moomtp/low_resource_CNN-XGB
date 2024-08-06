from node import *
from digit_helper_function import *
from gen_behavior import *


from dataclasses import dataclass, field



class Tree():

    # assert list of nodes is preordering
    def __init__(self, 
                 list_of_p : list[int],
                 list_of_rc:list[int], 
                 list_of_lc:list[int], 
                 list_of_node_vals:list[float],
                 node_ref_feature: list[int],  # list from split indices
                 cur_tree_addr:int, # num of nodes from tree0's root
                 is_last_tree:bool
                 ) -> None:
        self.nodes = []
        self.nums_nodes = len(list_of_p)
        self.addr_to_next_tree = cur_tree_addr + self.nums_nodes
        self.cur_tree_addr = cur_tree_addr 
        self.list_of_rc = list_of_rc
        self.list_of_lc = list_of_lc
        self.list_of_subtree_size = [-1 for _ in range(self.nums_nodes)] # create list which storing size of tree from node's id
        self.ram_bit = 14
        # cur_tree
        # preordering : root -> lc -> rc
        # idx : node num

        # init list_of_subtree_size
        self._cal_size_from_node(0)


        node_stack = [0]
        while node_stack:
            cur_node_id = node_stack.pop()
            # case : leaf node 
            if list_of_lc[cur_node_id] == -1 and list_of_lc[cur_node_id] == -1:
                self.nodes.append(LeafNode(list_of_node_vals[cur_node_id],
                                           is_last_tree,
                                           self.addr_to_next_tree,
                                           cur_node_id))
                pass
            # case : error use
            elif list_of_rc[cur_node_id] == -1 or list_of_lc[cur_node_id] == -1:
                raise ValueError("leaf node's child should be all -1")

            #  case : tree node
            else : 
                # add new tree node to self.nodes
                self.nodes.append(TreeNode(list_of_node_vals[cur_node_id],
                                     self._get_lc_tree_size(cur_node_id)+1,
                                     node_ref_feature[cur_node_id],
                                     cur_node_id))

                # push lchild and rchild to stack
                
                node_stack.append(list_of_rc[cur_node_id])
                node_stack.append(list_of_lc[cur_node_id])
                pass
        


        # for node in self.nodes:
        #     # setting nums of node in sub tree
        #     pass




    # VHDL code format
    # Addr <= "0000000000000";
    # Trees_din <= x"3c07fd0c";
    # wait for Clk_period;
    def gen(self) -> str: 
        nodes_on_FPGA = []
        FPGA_code_lines = ""
        for idx, node in enumerate(self.nodes):
            # add vhdl code on node.gen()
            # TODO : storage nodes_on_FPGA in self?
            nodes_on_FPGA.append(node.gen())

            # cnt byte addr from 0 to 1 
            # transfer binary to hex
            FPGA_code_lines += self._addr_deco(self._node_deco(b_to_h(node.gen())), self.cur_tree_addr + idx)
            FPGA_code_lines += "\n"
        return FPGA_code_lines

    def eval(self, feature):
        cur_node = self.nodes[0]
        cur_idx = 0
        while isinstance(cur_node, TreeNode):
            print("tree node id : "+ str(cur_node.node_id))
            print("tree node val : "+ str(cur_node.node_val))
            if feature[cur_node.feature_idx] > cur_node.node_val :
                print("tree jump addr : "+ str(cur_node.jump_addr))
                cur_idx += cur_node.jump_addr
                cur_node = self.nodes[cur_idx]
            else :
                cur_idx += 1
                cur_node = self.nodes[cur_idx]
        
        if not isinstance(cur_node, LeafNode):
            raise ValueError
        print("leaf node id : "+ str(cur_node.node_id))
        print("leaf node val : "+ str(cur_node.node_val))
        return cur_node.node_val

    def _addr_deco(self, node_str:str, addr:int):
        """
        Trees_din <= x"3c07fd0c";  => 
        Addr <= "0000000000000";
        Trees_din <= x"3c07fd0c";
        """
        return ("\t\tAddr <=  \"" + i_to_b(addr, self.ram_bit) + "\";") + "\n\t\t"  + node_str

    def _node_deco(self, node_str:str):
        """
        decorator for node line
        3c07fd0c => 
        Trees_din <= x"3c07fd0c"; 
        """
        return ("Trees_din <= x\"" + node_str + "\";") + "\n\t\twait for Clk_period;"

    def _get_lc_tree_size(self, root:int) -> None: 
        
        return self.list_of_subtree_size[self.list_of_lc[root]] if self.list_of_lc[root] != -1 else 0

    def _cal_size_from_node(self, root:int): 
        """cal nums of nodes in l&r sub trees under root, by postorder"""

        # if root is leaf node
        if self.list_of_lc[root] == -1 and self.list_of_rc[root] == -1:
            self.list_of_subtree_size[root] = 1
            return
        elif self.list_of_lc[root] == -1 or self.list_of_rc[root] == -1:
            raise ValueError("should not exist single leaf child!")
        
        # if root is tree node

        lc = self.list_of_lc[root]
        rc = self.list_of_rc[root]
        self._cal_size_from_node(lc)
        self._cal_size_from_node(rc)
        self.list_of_subtree_size[root] = self.list_of_subtree_size[lc] + self.list_of_subtree_size[rc] + 1
        
        return 

    # -----  getter & setter func  -------
    def get_max_leaf_weight(self):
        "find the maxinum leaf node val in this tree (abs)"
        max_leaf_weight = 0
        for node in self.nodes:
            if isinstance(node, LeafNode):
                max_leaf_weight = max(abs(node.node_val), max_leaf_weight)

        return max_leaf_weight

    def set_leaf_node_gen_behavior(self, _gen_behavior:Gen_behavior):
        for node in self.nodes:
            if isinstance(node, LeafNode):
                node.gen_behaivor = _gen_behavior

    def set_tree_node_gen_behavior(self, _gen_behavior:Gen_behavior):
        for node in self.nodes:
            if isinstance(node, TreeNode):
                node.gen_behaivor = _gen_behavior

    def set_tree_ram_bit(self, _new_ram_bit:int):
        self.ram_bit = _new_ram_bit



class ThreadTree(Tree):
    """
    this tree only has one node for special utility -> 
    make class.vhd know the thread entry addr
    """
    def __init__(self, 
                 cur_tree_addr:int, # num of nodes from tree0's root
                 ) -> None:
        self.cur_tree_addr = cur_tree_addr 
        self.addr_to_next_tree = cur_tree_addr + 1
        self.ram_bit = 14

        leaf = LeafNode(0, True, cur_tree_addr, 0)
        leaf.gen_behaivor = Gen_thread_node
        self.nodes = [leaf]
    
    
class ClassEndTree(Tree):
    # assert list of nodes is preordering
    def __init__(self, 
                 cur_tree_addr:int, # num of nodes from tree0's root
                 ) -> None:
        self.cur_tree_addr = cur_tree_addr 
        self.addr_to_next_tree = cur_tree_addr + 1
        self.ram_bit = 14

        leaf = LeafNode(0, True, cur_tree_addr, 0)
        leaf.gen_behaivor = Gen_class_end_node
        self.nodes = [leaf]
    
class ClassesEndTree(Tree):
    # assert list of nodes is preordering
    def __init__(self, 
                 cur_tree_addr:int, # num of nodes from tree0's root
                 ) -> None:
        self.cur_tree_addr = cur_tree_addr 
        self.addr_to_next_tree = cur_tree_addr + 1
        self.ram_bit = 14

        leaf = LeafNode(0, True, cur_tree_addr, 0)
        leaf.gen_behaivor = Gen_classes_end_node
        self.nodes = [leaf]