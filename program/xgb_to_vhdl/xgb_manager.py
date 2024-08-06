from node import *
from gen_behavior import *

from xgb_tree_manager import XGB_tree_manager
from xgb_feature_manager import XGB_feature_manager



import json

@dataclass
class XGB_manager:
    tree_manager:XGB_tree_manager
    # set feature managers
    feature_managers:XGB_feature_manager
    num_classes:int
    num_feature:int
    
    
    def __init__(self, filepath:str):
        self.tree_manager = XGB_tree_manager(filepath)
        self.feature_manager = XGB_feature_manager(filepath)
        
        # set classes and feature #
        with open(filepath, 'r') as file:
            data = json.load(file)
        self.num_classes = int(data["learner"]["gradient_booster"]['model']["iteration_indptr"][1])
        self.num_feature = len(data["learner"]["feature_names"])

        self.q_format = [7,8]
        
        pass

    def create_vhdl_code(self, filepath:str, list_of_features:list[list[float]]):
        # Write the VHDL code to the file
        len_features = len(list_of_features)
        with open(filepath, 'w') as file:
            file.write(self.gen(list_of_features, [0] * len_features))

        return 

    def create_vhdl_labeling_code(self, _filepath:str, _list_of_features:list[list[float]], _labels:list[int]):
        # Write the VHDL code to the file
        with open(_filepath, 'w') as file:
            file.write(self.gen(_list_of_features, _labels))

        return 
    


    def gen(self, list_of_features:list[list[float]], _labels:list[int]):
        # create code header
        header = ""
        header = self._head_deco(header)
        return header + self.tree_manager.gen() + self.feature_manager.gen(list_of_features, _labels)

    def eval(self, feature):
        return self.tree_manager.eval(feature)

    # ----- setter & getter  -----
    def get_max_class_weight(self) -> float:
        return self.tree_manager.get_max_class_weight()

    
    def set_q_foramt(self, _q_format:list[int]):
        # set for feature manager
        self.set_q_format_of_feature_gen_func(_q_format)
        # set for tree manager
        class New_leaf_gen_bahavior(Gen_behavior):
            # override
            def gen(self, _node):
                from node import Node
                return  f_to_fix16_b(_node.node_val, _q_format[0], _q_format[1]) + i_to_b(_node.jump_addr, 14) + "0" + "1"
        class New_tree_gen_bahavior(Gen_behavior):
            # override
            def gen(self, _node):
                from node import Node
                return i_to_b(_node.get_feature_idx(), 8) + f_to_fix16_b(_node.node_val, _q_format[0], _q_format[1]) + i_to_b(_node.jump_addr, 7) + "0"

        self.set_leaf_node_gen_behavior(New_leaf_gen_bahavior)

        self.set_tree_node_gen_behavior(New_tree_gen_bahavior)


    def set_q_format_of_feature_gen_func(self, _q_format:list[int]):
        self.feature_manager.set_q_format(_q_format)


    def set_tree_node_gen_behavior(self, _gen_behavior:Gen_behavior):
        self.tree_manager.set_tree_node_gen_behavior(_gen_behavior)
    def set_leaf_node_gen_behavior(self, _gen_behavior:Gen_behavior):
        self.tree_manager.set_leaf_node_gen_behavior(_gen_behavior)

    def set_ram_bit(self, _new_ram_bit:int):
        assert _new_ram_bit <= 14
        self.tree_manager.set_ram_bit(_new_ram_bit)

    # ------  private  ------------
    def _head_deco(self, code_str:str):
        return f"""

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.types.all;

entity image_test is
    generic(TREE_RAM_BITS: positive := {self.tree_manager.classes_tree[0][0].ram_bit};
            NUM_CLASSES:   positive := {self.num_classes};
            NUM_FEATURES:  positive := {self.num_feature});
end image_test;

architecture behavior of image_test is
    
    component image
        generic(TREE_RAM_BITS: positive;
                NUM_CLASSES:   positive;
                NUM_FEATURES:  positive);
        port(-- Control signals
             Clk:   in std_logic;
             Reset: in std_logic;
             
             -- Inputs for the nodes reception (trees)
             Load_trees: in std_logic;
             Valid_node: in std_logic;
             Addr:       in std_logic_vector(TREE_RAM_BITS - 1  downto 0);
             Trees_din:  in std_logic_vector(31 downto 0);
             
             -- Inputs for the features reception (pixels)
             Load_features: in std_logic;
             Valid_feature: in std_logic;
             Features_din:  in std_logic_vector(15 downto 0);
             Last_feature:  in std_logic;
             
             -- Output signals
             --     Finish:     finish (also 'ready') signal
             --     Dout:       the selected class
             --     Greater:    the value of the selected class prediction
             --     Curr_state: the current state
             Finish:     out std_logic;
             Dout:       out std_logic_vector(log_2(NUM_CLASSES) - 1 downto 0);
             greater:    out std_logic_vector(31 downto 0);
             curr_state: out std_logic_vector(2 downto 0));
    end component;
    
    component counter is
        generic(BITS: natural);
        port(Clk:   in  std_logic;
             Reset: in  std_logic;
             Count: in  std_logic;
             Dout:  out std_logic_vector (BITS - 1 downto 0));
    end component;
    
    -- Inputs
    signal Clk:           std_logic := '0';
    signal Reset:         std_logic := '0';
    signal Load_trees:    std_logic := '0';
    signal Valid_node:    std_logic := '0';
    signal Addr:          std_logic_vector(TREE_RAM_BITS - 1 downto
                                           0) := (others => '0');
    signal Trees_din:     std_logic_vector(31 downto 0) := (others => '0');
    signal Load_features: std_logic := '0';
    signal Valid_feature: std_logic := '0';
    signal Features_din:  std_logic_vector(15 downto 0) := (others => '0');
    signal last_feature:  std_logic := '0';
    
    -- Outputs
    signal Finish:     std_logic;
    signal Dout:       std_logic_vector(log_2(NUM_CLASSES) - 1 downto 0);
    signal greater:    std_logic_vector(31 downto 0);
    signal curr_state: std_logic_vector(2 downto 0);
    
    -- Clock period definition
    constant Clk_period : time := 10 ns;
    
    -- Counter signals
    signal pc_count, hc_count: std_logic := '0';
    signal pixels, hits: std_logic_vector(15 downto 0) := (others => '0');
    
    -- Label signal
    signal class_label: std_logic_vector(log_2(NUM_CLASSES) - 1 downto 0);

begin
    
    -- Instantiate the Unit Under Test (UUT)
    uut: image
        generic map(TREE_RAM_BITS => TREE_RAM_BITS,
                    NUM_CLASSES   => NUM_CLASSES,
                    NUM_FEATURES  => NUM_FEATURES)
        port map(Clk           => Clk,
                 Reset         => Reset,
                 Load_trees    => Load_trees,
                 Valid_node    => Valid_node,
                 Addr          => Addr,
                 Trees_din     => Trees_din,
                 Load_features => Load_features,
                 Valid_feature => Valid_feature,
                 Features_din  => Features_din,
                 Last_feature  => Last_feature,
                 Finish        => Finish,
                 Dout          => Dout,
                 greater       => greater,
                 curr_state    => curr_state);
    
    -- To count the pixels
    pixel_counter: counter
        generic map(BITS => 16)
        port map(Clk   => Clk, 
                 Reset => Reset,
                 Count => pc_count,
                 Dout  => pixels);
    
    -- To count the hits
    hit_counter: counter
        generic map(BITS => 16)
        port map(Clk   => Clk, 
                 Reset => Reset,
                 Count => hc_count,
                 Dout  => hits);
    
    -- Clock process definition
    Clk_process: process
    begin
        Clk <= '0';
        wait for Clk_period/2;
        Clk <= '1';
        wait for Clk_period/2;
    end process;
    
    -- Stimulus process
    stim_proc: process
    begin
        
        Reset <= '1';
        
        -- hold reset state for 100 ns.
        wait for 100 ns;
        
        Reset <= '0';
        
        wait for Clk_period*10;
        


        """ + code_str
        pass
