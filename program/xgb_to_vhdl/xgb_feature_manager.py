from node import *
from xgb_tree import Tree
from digit_helper_function import *

import xgboost as xgb
import json
import copy

# TODO : 實現下面這行的功能
#  class_label <= std_logic_vector(to_unsigned(1, class_label'length));

class XGB_feature_manager:
    # 16 bit fix pointer format


    def __init__(self, args):
        # read feature idx from json
        if isinstance(args, str):
            with open(args, 'r') as file:
                data = json.load(file)
            feature_idx = data["learner"]["feature_names"]
        elif isinstance(args, list):
            feature_idx = args
        self.init(feature_idx)

        # set default q_format
        self.q_format = [7,8]

        # idx trans ['f123', 'f1' , ...] -> [123, 1 , ...]
    
    # # init func for unittest
    # def __init__(self, _feature_idx:list[str]):
    #     # self.feature_idx = feature_idx
    #     feature_idx = copy.deepcopy(_feature_idx)
        
    #     for i, idx in enumerate(feature_idx):
    #         feature_idx[i] = int(idx[1:])
    #     self.feature_idx = feature_idx
    #     pass

    def init(self, _feature_idx):
        feature_idx = copy.deepcopy(_feature_idx)
        
        for i, idx in enumerate(feature_idx):
            feature_idx[i] = int(idx[1:])
        self.feature_idx = feature_idx
        pass
    
    def gen(self, _list_of_features:list[list[float]] , _labels:list[int]):
        # assert features & labels have the same size
        assert len(_list_of_features) == len(_labels), f"Assertion fail: features.size() != labels.size(), {len(_list_of_features)} != {len(_labels)}"

        codes = ""

        for feature, label in zip(_list_of_features , _labels):
            codes += self._feature_header(label) 
            codes += self._gen_feature_data(feature)

        return self._features_tailer_deco(codes)

    # ------  private func  -------------
    def _gen_feature_data(self, ori_features:list[float]):
        features = self._feature_transform(ori_features)
        res = ""
        for idx, feature_ele in enumerate(features):
            # if first feature, use _first_feautre_deco
            if idx == 0:
                res += self._first_feature_deco(feature_ele)
            # if last feature, use _last_feautre_deco
            elif idx == len(features) - 1:
                res += self._last_feature_deco(feature_ele)
            else:
                res += self._feature_deco(feature_ele)

        return res

    def _feature_transform(self, ori_feature:list) ->list:
        """
        [2.34, 4.44, 5.54] -> [4.44, 5.54, 2.34] -> [01000101011, ...] (base on self.feature_idx)
        """
        #  use self.feature_idx
        res = []
        for idx in self.feature_idx:
            res.append(f_to_fix16_b(ori_feature[idx] , self.q_format[0], self.q_format[1]))
        assert len(res) < 256
        return res

    def _feature_deco(self, bit_str:str):
        return f"""
        Features_din <= "{bit_str}";
        wait for Clk_period; """

    def _last_feature_deco(self, bit_str:str):
        return f"""
        \n
        last_feature <= '1';
        pc_count     <= '1'; -- count pixel
        Features_din <= "{bit_str}";
        wait for Clk_period; 
        """ + """
        \n
        -- Reset count, last and valid flags
        pc_count      <= '0';
        Last_feature  <= '0';
        Valid_feature <= '0';
        
        -- Wait until inference is complete
        wait until Finish = '1';
        
        wait for Clk_period * 1/2;
        
        if Dout = class_label then
            hc_count <= '1';
        end if;
        
        wait for Clk_period;
        hc_count <= '0';
        
        """


    def _first_feature_deco(self, bit_str:str):
        return f"""-- Load and valid features flags
        Load_features <= '1';
        Valid_feature <= '1';
        
        Features_din <= "{bit_str}";
        wait for Clk_period;

        -- Reset load flag
        Load_features <= '0';
        """

    def _feature_header(self, label:int):
        return f"""
        class_label <= std_logic_vector(to_unsigned({label}, class_label'length));
        
        """ 
        pass



    def _features_tailer_deco(self, code_str):
        return code_str + f"""
            wait;
    end process;
end;
"""
    # ------  setter & getter func  -------------

    def set_q_format(self, _q_format:list[int]):
        assert len(_q_format) == 2
        assert _q_format[0] + _q_format[1] ==15 
        self.q_format = _q_format




