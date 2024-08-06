import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import csv
import numpy as np
import copy
import torchvision.models as models
from typing import Dict , List
import time

# import self define func
import os
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))

from other_program.custom_model import CustomModel

from .imgToFeatureVectorFunctions import computeLbpHistogram, computeHuMoments, computeColorIndex

# ==========   pytorch function  =============

# trans cv2.imread to torch.tensor
def img2tensor(img):
    img = cv2.resize(img, (224, 224))

    data = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return data



# def switchLastLayer(model:nn.modules, num_classes: int):

#     # 获取分类器中最后一个全连接层之前的部分
#     classifier = list(model.classifier.children())[:-1]

#     # 移除原始模型的最后一个全连接层
#     # 并添加一个新的全连接层，输出特征数为 7
#     classifier.append(torch.nn.Linear(4096, num_classes))

#     # 替换原始模型的分类器
#     model.classifier = torch.nn.Sequential(*classifier)
def dataloaderToFeatureData(model: nn.modules, dataloader, device:str):
    features = []
    labels = []
    for inputs, label in dataloader:
        inputs, label = inputs.to(device), label.to(device)
        with torch.no_grad():
            output = model(inputs)
        output ,label = output.to("cpu") , label.to("cpu")  
        output = flattenExceptDim0(output)
        # print(output.shape)

        features.extend(output.numpy().squeeze())
        # features.extend(output.numpy().unsqueeze())
        labels.extend(label.numpy())

        assert len(features) == len(labels)
    # flatten
    print("feature size is :{}".format(len(features)))
    print("label size is :{}".format(len(labels)))
    return np.array(features), np.array(labels)


def generateModelOutputs(model: nn.modules, input_folder: str):

    res = []

    # open every img
    for filename in os.listdir(input_folder):

        image_path  = os.path.join(input_folder, filename)
        # print(image_path)
        img = cv2.imread(image_path)
        img = img2tensor(img)

        # model setting and eval
        model.eval()
        with torch.no_grad():
            tensor = model(img)
        
        tensor = flattenTensor(tensor)
        vec = tensor.tolist()

        vec = [filename] + vec
        res.append(vec)

    return res

def flattenTensor(tensor):
    tensor = torch.squeeze(tensor)
    flatted_tensor = tensor.view(-1)
    return flatted_tensor

def flattenExceptDim0(tensor):
    # 獲取第0維的大小
    dim0_size = tensor.size(0)
    # 保留第0維的大小，將其餘維度展平
    flatted_tensor = tensor.view(dim0_size, -1)
    return flatted_tensor

# def remove_last_recursive(module, removed_layers):
#     """
#     递归地移除模块中的最后一层，并记录被移除层的名称。
#     """
#     children = list(module.named_children())
#     if len(children) == 0:
#         return None, removed_layers
#     elif len(children) > 1:
#         *rest, (last_name, last_module) = children
#         new_module, new_removed = remove_last_recursive(last_module, removed_layers)
#         if new_module is None:
#             new_module = nn.Sequential(*(m for _, m in rest))
#             new_removed = removed_layers + [last_name]
#         else:
#             new_module = nn.Sequential(*(m for _, m in rest), new_module)
#         return new_module, new_removed
#     else:
#         name, module = children[0]
#         new_module, new_removed = remove_last_recursive(module, removed_layers + [name])
#         return nn.Sequential(new_module) if new_module is not None else None, new_removed

# def create_resnet_versions(model):
#     """
#     创建包含原始模型及其逐渐减少层的版本的列表，并记录被移除层的信息。
#     """
#     models_list = [(copy.deepcopy(model), ["Original Model"])]
#     current_model = copy.deepcopy(model)

#     while True:
#         current_model, removed_layers = remove_last_recursive(current_model, [])
#         if current_model is None:
#             break
#         models_list.append((copy.deepcopy(current_model), removed_layers))

#     return models_list

def createDetailLayerVersions(model):
    remained_layers = calDetailModelLayersNum(model)
    list_of_models = []

    while remained_layers > 0:
        list_of_models.append((model, "layer:" + str(remained_layers)))
        model = removeLastLayer(model, "layer")
        remained_layers-= 1
    return list_of_models

def removeLastLayer(model, layer_label):
    children = list(model.children())
    if not children:
        # return if no sub layer
        return model
    else:
        # check if last layer has sub layers
        last_layer = children[-1]
        if list(last_layer.children()):
            # if yes, recurit it
            new_last_layer = removeLastLayer(last_layer, layer_label + ".sub")
            new_model = torch.nn.Sequential(*children[:-1] + [new_last_layer])
        else:
            # if no, deprive it
            new_model = torch.nn.Sequential(*children[:-1])
        return new_model

def calImgFeatureVector(tensor_img_data:torch.Tensor, isRGB:bool):
    """
    功能 : 算出影像的特徵向量
    輸入 : tensor影像資料
    輸出 : LBP與hu_invariant 的vector
    """
    # create loop
    # list_of_feature_vectors = []
    # for idx , (data, label) in enumerate():
    #     pass

    # tensor_img_data = tensor_img_data.permute(1,2,0)
    img = tensor_img_data.numpy()
    img = img.transpose((1,2,0))

    # print(img.shape)
    # print(type(computeHuMoments(img, isRGB)))
    # print(computeLbpHistogram(img, isRGB).shape)
    # print((computeLbpHistogram(img, isRGB=isRGB)).shape)
    # print(computeColorIndex(img, isRGB=isRGB).shape)

    # feature_vector.append(computeHuMoments(img))
    # feature_vector.append(computeLbpHistogram(img))
    vec = np.concatenate((computeHuMoments(img,isRGB) ,computeLbpHistogram(img, isRGB), computeColorIndex(img, isRGB=isRGB)))
    nan_indices = np.where(np.isnan(vec))
    if (len(nan_indices) > 1):
        print(f'Indices of NaN values: {list(zip(nan_indices[0]))}')
        print("NaN ele found error!!")
        raise ValueError
    return vec

def removeLastLayerV2(model, layer_label=""):
    children = list(model.children())
    if not children:
        # 如果模型没有子层，直接返回
        return model
    else:
        # 检查最后一层是否有子层
        last_layer = children[-1]
        if list(last_layer.children()):
            # 如果最后一层有子层，递归地对最后一层进行这个操作
            new_last_layer = removeLastLayerV2(last_layer, layer_label + ".sub")
            # 检查new_last_layer是否只有一个子层，如果是，移除嵌套
            if len(list(new_last_layer.children())) == 1:
                # 直接取得这个唯一的子层
                new_last_layer = list(new_last_layer.children())[0]
            # 根据新的最后一层重新构造模型
            new_model = torch.nn.Sequential(*children[:-1], new_last_layer)
        else:
            # 如果最后一层没有子层，去除最后一层
            new_model = torch.nn.Sequential(*children[:-1])
        return new_model

def removeLastLayerV3(model, layer_label):
    children = list(model.children())
    if not children:
        # 如果模型没有子层，直接返回
        return model
    else:
        # 检查最后一层是否有子层
        last_layer = children[-1]
        if list(last_layer.children()):
            # 如果最后一层有子层，递归地对最后一层进行这个操作
            new_last_layer = removeLastLayerV3(last_layer, layer_label + ".sub")
            # 使用CustomModel来重新组合子模块，除去最后一层
            new_model = CustomModel(torch.nn.Sequential(*children[:-1] + [new_last_layer]))
        else:
            # 如果最后一层没有子层，去除最后一层，并使用CustomModel来重新组合子模块
            new_model = CustomModel(torch.nn.Sequential(*children[:-1]))
        return new_model

def calDetailModelLayersNum(model):
    total_layers = 0
    for name, layer in model.named_modules():
        total_layers += 1
    print("總層數為: " + str(total_layers) + "層")
    return total_layers
    

def calModelEvalTime(models_list:List, device:str, isRGB:bool=True )-> Dict: 
    results = {"eval_time" : [], "model_name" : []}
    # ele in result :  (model_eval_time , "layer" + idx)

    # 10000 for error rate 8%, 100000 is better
    evaluations_per_model = 10000
    if isRGB:
        dummy_tensor = torch.rand([1, 3, 224, 224]).to(device)
    else :
        dummy_tensor = torch.rand([1, 1, 224, 224]).to(device)

    for model, model_name in models_list:
        model.eval()  # 將模型設置為評估模式
        times = []
        model = model.to(device)

        for _ in range(evaluations_per_model):
            try:
                start = time.perf_counter()
                with torch.no_grad():  # 在這裡不計算梯度
                        _ = model(dummy_tensor)
                end = time.perf_counter()
                times.append(end - start)
            except:
                print("mat error!")
                break

        average_time = sum(times) / evaluations_per_model
        results["eval_time"].append(average_time)
        results["model_name"].append(model_name)

    return results

# =============  other function  =============

def saveCsv(res: list, output_csv_name: str):
    with open(output_csv_name, 'w', newline='', encoding='utf-8') as csvfile:
            # try:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(res)
            # except Exception as e:
    #     logging.error("csv error : %s" , e)
    #     print('csv encoding error')

def addHeader(res: list, head0: str, otherHead: str):
    size = len(res[0])
    head = [head0]
    for i in range (size - 1):
        head.append(otherHead + '[' + str(i) + ']')
    res = [head] + res
    
    return res

