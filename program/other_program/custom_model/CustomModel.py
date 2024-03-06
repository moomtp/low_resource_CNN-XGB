import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, model):
        super(CustomModel, self).__init__()
        # 将模型的子层作为列表保存，除去最后一层
        self.layers = list(model.children())[:-1]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
