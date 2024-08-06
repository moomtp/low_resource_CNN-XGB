from node import *
from xgb_tree import *
from gen_behavior import Gen_behavior

import xgboost as xgb
import json

class XGB_tree_manager:

    # classes_tree:list[list[Tree]] = [] # len = num_classes


    def __init__(self, filepath:str):
        # 讀取xgb.json的資料
        with open(filepath, 'r') as file:
            data = json.load(file)

        self.num_classes = int(data["learner"]["gradient_booster"]['model']["iteration_indptr"][1])
        # print(self.num_classes)
        trees_data = data["learner"]["gradient_booster"]['model']['trees']
        # print(self.classes_tree)

        self.classes_tree = self._init_tree(trees_data, self.num_classes)

    def _init_tree(self, _trees_data:list, _num_classes:int)->list[list[Tree]]:
        classes_tree = []


        # create self.classes_tree
        for _ in range(_num_classes):
            classes_tree.append([])

        assert len(classes_tree) == self.num_classes


        # cal total node of classes tree -> {[#total_nodes, #2/3nodes, #1/3nodes] , [#total_nodes, #2/3nodes, #1/3nodes], ... }
        next_thread_nodes_list = self._cal_threads_node_num(_trees_data=_trees_data, _num_classes=self.num_classes)


        is_last_tree = False
        # init self.classes_tree
        for idx, tree_data in enumerate(_trees_data):
            # set # class of tree
            class_num = idx % self.num_classes

            # set tree's jump addr
            if classes_tree[class_num] == []:
                addr_from_first_tree = 0
            else : 
                addr_from_first_tree = classes_tree[class_num][-1].addr_to_next_tree
            
            # if thread addr approch, add new thread end tree

            if addr_from_first_tree > next_thread_nodes_list[class_num][-1]:
                # add thread tree to classes_tree[class_num]
                next_thread_nodes_list[class_num].pop()
                classes_tree[class_num].append(ThreadTree(addr_from_first_tree))
                addr_from_first_tree = classes_tree[class_num][-1].addr_to_next_tree

            # print(next_thread_nodes_list)
            # set thread end flag
            # if idx / self.num_classes >= len(tree_data) / self.num_classes:
            #     is_last_tree = True


            classes_tree[class_num].append(Tree(
                                                    tree_data['parents'],
                                                    tree_data['right_children'],
                                                    tree_data['left_children'],
                                                    tree_data['split_conditions'],
                                                    tree_data['split_indices'],
                                                    addr_from_first_tree,
                                                    is_last_tree))

        # append class_end tree
        for idx, _ in enumerate(classes_tree):
            class_num = idx % self.num_classes
            # print(class_num)
            addr_from_first_tree = classes_tree[class_num][-1].addr_to_next_tree
            # if last class, append total end tree
            if idx == self.num_classes - 1:
                classes_tree[idx].append(ClassesEndTree(addr_from_first_tree))
            else:
                classes_tree[idx].append(ClassEndTree(addr_from_first_tree))


        return classes_tree
        pass
        


    def gen(self):
        print(self.classes_tree[0][0].gen())
        code = ""
        code = self._code_header_deco(code)

        for idx, class_trees in enumerate(self.classes_tree):
            code += self._class_trees_header(idx)

            for tree in class_trees:
                code += tree.gen()

        code = self._code_tailer_deco(code)

        return code

    def eval(self, feature:list[float]):
        scores = [0.0] * self.num_classes
        for idx , class_trees in enumerate(self.classes_tree):
            print(f"cur tree is :{idx}")
            for tree in class_trees:
                scores[idx] += tree.eval(feature)
        return scores

    # -----------   getter & setter func  -------

    def get_max_class_weight(self) -> float:
        "find the theoretical maximum val of one class trees could output"
        max_class_trees_weight = 0

        for class_trees in self.classes_tree:
            total_weight = 0

            # assume the eval result in all subtree, we get the maxinum weight
            for tree in class_trees:
                total_weight += tree.get_max_leaf_weight()
            max_class_trees_weight = max(total_weight, max_class_trees_weight)

        return max_class_trees_weight
    
    def set_tree_node_gen_behavior(self, _gen_behavior:Gen_behavior):
        for class_trees in self.classes_tree:
            for tree in class_trees:
                tree.set_tree_node_gen_behavior(_gen_behavior)
                
    def set_leaf_node_gen_behavior(self, _gen_behavior:Gen_behavior):
        for class_trees in self.classes_tree:
            for tree in class_trees:
                if isinstance(tree, ClassEndTree) or isinstance(tree, ClassesEndTree) or isinstance(tree, ThreadTree):
                    continue
                tree.set_leaf_node_gen_behavior(_gen_behavior)

    def set_ram_bit(self, _new_ram_bit:int):
        for class_trees in self.classes_tree:
            for tree in class_trees:
                tree.set_tree_ram_bit(_new_ram_bit)
    

    # def set_xgb_tree(self, filepath:str):
    #     pass
    def _cal_threads_node_num(self, _trees_data:list, _num_classes:int) -> list:
        class_total_node_list = []
        # init
        for _ in range(_num_classes):
            class_total_node_list.append([0])

        # cal total nodes of each class
        for idx, tree_data in enumerate(_trees_data):
            # print(tree_data)
            class_num = idx % _num_classes
            class_total_node_list[class_num][0] += int(tree_data["tree_param"]["num_nodes"])
        
        for class_num, tree_nodes in enumerate(class_total_node_list):
            class_total_node_list[class_num] = [tree_nodes[0], tree_nodes[0]*2/3, tree_nodes[0]/3]


        return class_total_node_list


    def _class_trees_header(self, class_of_i : int):
                
        return f"""
        -- Class  {class_of_i}
        -----------
"""
    


    def _code_header_deco(self, code_str:str):
        return code_str + """
        -- LOAD TREES
        -----------------------------------------------------------------------
        
        -- Load and valid trees flags
        Load_trees <= '1';
        Valid_node <= '1';
"""

        pass

    # def _code_header_deco(self, num_classes:int, num_feature:int):
    #     return """
    #     -- LOAD TREES
    #     -----------------------------------------------------------------------
        
    #     -- Load and valid trees flags
    #     Load_trees <= '1';
    #     Valid_node <= '1';
        

    #     """
        pass

    def _code_tailer_deco(self, codes_str:str):
        return codes_str + """
        -- Reset valid flag
        Valid_node <= '0';
        wait for Clk_period; 
        """




