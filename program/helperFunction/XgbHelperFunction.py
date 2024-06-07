import time
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pandas as pd
from torch import nn
from .helperFunctions import dataloaderToFeatureData
from sklearn.metrics import accuracy_score, f1_score , log_loss
import xgboost as xgb
import numpy as np
import json
import torch
import os
import pandas as pd


def train_predict(clf, X_train, y_train, X_test, y_test , evalByMMSE = False):
    ''' 訓練並評估模型 '''
    # Indicate the classifier and the training set size
    print("訓練 {} 模型，樣本數: {}。".format(clf.__class__.__name__, len(X_train)))
    # 訓練模型
    train_classifier(clf, X_train, y_train)
    # 在訓練集上評估模型
    res1 , res2 = predict_labels(clf, X_train, y_train, evalByMMSE)
    res3, res4 = predict_labels(clf, X_test, y_test, evalByMMSE)
    if evalByMMSE :

      print("訓練集上的 MAE,RMSE : {:.6f} , {:.6f}。".format(res1, res2))

      mae, rmse = predict_labels(clf, X_test, y_test)

      print("測試集上的 MAE,RMSE : {:.6f} , {:.6f}。".format(res3, res4))

      print("different between MSE , RMSE : {:.7f} , {:.7f}".format(mae-0.375 , rmse-0.43301))

    print("訓練集的 F1 score和acc分別為: {:.4f} , {:.4f}。".format(res1 , res2))
    print("測試集的 F1 score和acc分別為: {:.4f} , {:.4f}。".format(res3 , res4))
    return res3, res4




# res : {model_name : list[str], 
#        input_size :list[int] ,
#        acc : list[float], 
#        f1 : list[float], 
#        iters : list[int]
# #        }
# def recordXGBoutput(model:nn.Sequential, test_dataloader, train_dataloader, model_name:str, res:dict , enable_muti_module=False,  test_feature_vectors):
#   device = "cuda" if torch.cuda.is_available() else "cpu"
#   test_features , test_labels = dataloaderToFeatureData(model, test_dataloader,device)
#   train_features , train_labels = dataloaderToFeatureData(model, train_dataloader, device)

#   if enable_muti_module:
#     for idx,feature in enumerate(test_features):
#         feature = np.concatenate((feature , np.array(test_feature_vectors[idx])))
#     for idx,feature in enumerate(train_features):
#         feature = np.concatenate((feature , np.array(train_feature_vectors[idx])))




# ============主要的計算function====================
def calBestIterOfXGB(train_features, train_labels, test_features, test_labels, device, enable_f1_metric=True, model_output_path:str=None):
    """算出什麼時候會overfitting"""
    dtrain = xgb.DMatrix(train_features, label=train_labels)
    dval = xgb.DMatrix(test_features, label=test_labels)

    # 设置参数，注意要更改 'objective' 为多分类的目标函数
    num_class = len(np.unique(train_labels))  # 获取类别数


    params = {'eval_metric': 'mlogloss',
              'objective': 'multi:softprob',
              'num_class': num_class,
              'device':device,
              'verbosity':2 # 1 : none, 2 : info, 3 : debugg
              } 



    evals_result = {}
    save_best_model_callback = SaveBestModel(model_output_path, "mlogloss")
    bst = xgb.train(params, dtrain, num_boost_round=100, 
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    custom_metric=custom_eval, evals_result=evals_result, 
                    early_stopping_rounds=20,
                    # callbacks=[lambda env: save_best_model(env.model, model_output_path, best_score)],
                    callbacks=[save_best_model_callback],
                    verbose_eval=False)
    # 保存訓練好的模型
    bst.save_model(model_output_path + ".json")

    print(f"Best iteration: {bst.best_iteration}")

    with open('evals_result.json', 'w') as f:
      json.dump(evals_result, f)

    evaluations_per_model = 10000

    # 計算xgb eval所需的時間
    eval_time = cal_xgb_eval_time(bst, dval, evaluations_per_model)


    # 計算XGB容量    
    XGB_size = os.path.getsize(model_output_path + ".json")

    # create回傳結果
    res = {
           "iter" : bst.best_iteration,
           "f1" : evals_result['val']['F1-score'],
           "acc" :evals_result['val']['Accuracy'] ,
           "XGB eval time" : eval_time,
           "XGB size" : XGB_size,
          }
    
    return res
    # if enable_f1_metric:
    #   return bst.best_iteration, evals_result['val']['F1-score'], evals_result['val']['Accuracy'], eval_time
    # else:
    #   return bst.best_iteration

def calFPGAFormatXGB(train_features, train_labels, test_features, test_labels, device, enable_f1_metric=True, model_output_path:str=None):
    """訓練符合FPGA foramt的XGB模型"""
    dtrain = xgb.DMatrix(train_features, label=train_labels)
    dval = xgb.DMatrix(test_features, label=test_labels)

    # 设置参数，注意要更改 'objective' 为多分类的目标函数
    num_class = len(np.unique(train_labels))  # 获取类别数


    params = {'eval_metric': 'mlogloss',
              'objective': 'multi:softprob',
              'num_class': num_class,
              'device':device,
              'verbosity':2, # 1 : none, 2 : info, 3 : debugg
              } 

    # 降精度

    evals_result = {}
    save_best_model_callback = SaveBestModel(model_output_path, "mlogloss")
    bst = xgb.train(params, dtrain, num_boost_round=105, 
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    custom_metric=custom_eval, evals_result=evals_result, 
                    early_stopping_rounds=20,
                    # callbacks=[lambda env: save_best_model(env.model, model_output_path, best_score)],
                    callbacks=[save_best_model_callback],
                    verbose_eval=False)

    bst.save_model(model_output_path +  ".json")

    iter = bst.best_iteration

    # 計算xgb eval所需的時間
    evaluations_per_model = 10000
    eval_time = cal_xgb_eval_time(bst, dval, evaluations_per_model)
    
    # 找出最重要的feature，並且儲存下來轉換的Dict
    # 獲取特徵重要性 
    importance = bst.get_score(importance_type='weight') 

    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True) 
    # 抓出N個最重要的特徵 
    #  top_n_features = ['f4', 'f1', 'f0', 'f5', 'f6', 'f3', 'f2']
    top_n_features = [feature for feature, _ in sorted_importance[:255]] 
    top_n_features_and_score = [(feature, score) for feature, score in sorted_importance[:255]] 
    print(top_n_features)
    print(top_n_features_and_score)
    # 重構dtrain 跟 dval
    # TODO assert 每筆feautre數量相同?
    num_features = len(train_features[0])
    feature_name = [f'f{i}' for i in range(num_features)]

    df_train_features  = pd.DataFrame(train_features, columns=feature_name)
    df_test_features  = pd.DataFrame(test_features, columns=feature_name)
    df_top_train_features = df_train_features[top_n_features]
    df_top_test_features = df_test_features[top_n_features]

    dtrain_top_features = xgb.DMatrix(data=df_top_train_features, label=train_labels) 
    dval_top_features = xgb.DMatrix(data=df_top_test_features, label=test_labels) 

    # TODO 保存importance of feature的順序list檔
    with open(model_output_path + "_top_feature.json" , 'w')as f:
      json.dump(top_n_features_and_score, f)

    # 根据选定的特征重新训练模型 
    # train_top_features = train_features[top_n_features] 
    # test_top_features = test_features[top_n_features] 

    # 保存top feature的idx
    # feature_indices = {feature: train_features.columns.get_loc(feature) for feature in top_n_features}
    # print("Selected features and their original indices:", feature_indices)

    # 重新訓練一次model，限制樹的數量與node的數量
    # 限制子樹最大深度(不含leaf)
    params['max_depth'] =  6

    train_features = [feature.astype(np.float16) for feature in train_features]

    FPGA_evals_result = {}
    bst = xgb.train(params, dtrain_top_features, num_boost_round=100, 
                    evals=[(dtrain_top_features, 'train'), (dval_top_features, 'val')],
                    custom_metric=custom_eval, evals_result=FPGA_evals_result, 
                    early_stopping_rounds=20,
                    # callbacks=[lambda env: save_best_model(env.model, model_output_path, best_score)],
                    callbacks=[save_best_model_callback],
                    verbose_eval=False)
    # 保存訓練好的模型
    bst.save_model(model_output_path +  "_FPGA.json")
    bst.dump_model(model_output_path +  "_dFPGA.json", dump_format='json')

    f_iter = bst.best_iteration

    # with open('evals_result.json', 'w') as f:
    #   json.dump(evals_result, f)


    # 計算XGB容量    
    XGB_size = os.path.getsize(model_output_path + ".json")
    FPGA_XGB_size = os.path.getsize(model_output_path + "_FPGA.json")

    # create回傳結果
    res = {
           "iter" : iter,
           "f_iter" : f_iter,
           "f1" : evals_result['val']['F1-score'],
           "acc" :evals_result['val']['Accuracy'] ,
           "f_f1" : FPGA_evals_result['val']['F1-score'],
           "f_acc" :FPGA_evals_result['val']['Accuracy'] ,
           "XGB eval time" : eval_time,
           "XGB size" : XGB_size,
           "f_XGB_size" : FPGA_XGB_size,
          }
    
    return res


# =========  sub function  =============



# ------ train pred sub function ------
def train_classifier(clf, X_train, y_train):
    ''' 訓練模型 '''
    # 紀錄訓練時間
    # print("訓練資料 : ".format(X_train[0:4]))
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print("訓練時間 {:.4f} 秒".format(end - start))

def predict_labels(clf, features, target, evalByMMSE):
    ''' 使用模型進行預測 '''
    # 紀錄預測時間
    start = time()
    if evalByMMSE :
      y_pred = clf.predict_proba(features)  # y_pred 會return probability
      # prob_target = []
      # for vec in target:
      #   prob_target += [fromPermutationToProbability(vec)]
      # prob_pred = []
      # for vec in y_pred:
      #   prob_pred += [fromPermutationToProbability(vec)]
      # target = oneHotVecSerial(len(target))
      # return mean_absolute_error(prob_target, prob_pred) , mean_squared_error(prob_target, prob_pred, squared=False)

    y_pred = clf.predict(features)
    end = time()
    print("預測時間 in {:.4f} 秒".format(end - start))

    return f1_score(target, y_pred, pos_label=1, average='weighted'), sum(target == y_pred) / float(len(y_pred))


# ------------  xgb related fucntion  ------------------

# class for xgb callback
class SaveBestModel(xgb.callback.TrainingCallback):
    def __init__(self, filepath, monitor_metric='mlogloss'):
        super().__init__()
        self.filepath = filepath
        self.best_score = float('inf')
        self.monitor_metric = monitor_metric

    def after_iteration(self, model, epoch, evals_log):
        # 使用正确的数据集和度量标准
        if 'val' in evals_log and self.monitor_metric in evals_log['val']:
            current_score = evals_log['val'][self.monitor_metric][-1]
            # print(evals_log['val'][self.monitor_metric])
            # print("current score is : {}".format(current_score))
            # 假设我们是在追求更高的评分是更好的场景（如AUC、准确率）
            # 如果你追踪的是损失函数，应该使用current_score < self.best_score
            if current_score < self.best_score:
                self.best_score = current_score
                model.save_model(self.filepath + '.json')  # epoch + 1 because epochs are zero-indexed
        return False  # Return False to continue training

def custom_eval(preds, dtrain:xgb.DMatrix, enable_f1_metric=True):
    """設定xgb.train的eval function"""
    labels = dtrain.get_label()

    # 将预测结果转换为类别标签
    mlogloss = log_loss(labels, preds, labels=np.unique(dtrain.get_label()))
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')  # 用 'weighted' 适应类别不平衡

    if enable_f1_metric:
      return [('Accuracy', -acc), ('F1-score', -f1), ('mlogloss', mlogloss)] # 可以更換成這個，但early stop會變成f1 score

    return [('mlogloss', mlogloss)]  # 計算出mlogloss並做為評估early stop 的標準

def cal_xgb_eval_time(bst : xgb.Booster, matrix_data : xgb.DMatrix, evaluation_num:int):

    times = []
    for _ in range(evaluation_num):
      start_time = time.perf_counter()
      res = bst.predict(matrix_data)
      end_time = time.perf_counter()
      times.append(end_time - start_time)

    average_time = sum(times) / evaluation_num
    print("res's len : {}".format(len(res)))
    print("feature's len : {}".format(matrix_data.num_row()))
    print("跑測試資料的時間:{}".format(average_time/matrix_data.num_row()))

    return average_time


def save_best_model(bst, filepath, current_best):
    if bst.best_score < current_best:
        current_best = bst.best_score
        bst.save_model(filepath.format(bst.best_iteration))
    return current_best

# ------------- etc function  -----------------

def csvkeylistToData(csv_path:str, keys:list ):
    ''' 讀取csv檔的filename資料, return為list'''
    datas = []
    df = pd.read_csv(csv_path)
    for key in keys:
      data = (df[df['filename'] == key].iloc[0])
      datas.append(data.drop(labels='filename'))

    return datas




#  -------------  recycle bin  ----------------

# def calBestIterOfXGBByF1score( train_features, train_labels, test_features, test_labels):
#     """算出什麼時候會overfitting"""
#     dtrain = xgb.DMatrix(train_features, label=train_labels)
#     dval = xgb.DMatrix(test_features, label=test_labels)

#     # 设置参数，注意要更改 'objective' 为多分类的目标函数
#     num_class = len(np.unique(train_labels))  # 获取类别数
#     params = {'eval_metric': 'mlogloss', 'objective': 'multi:softprob', 'num_class': num_class}



#     evals_result = {}
#     bst = xgb.train(params, dtrain, num_boost_round=105, 
#                     evals=[(dtrain, 'train'), (dval, 'val')],
#                     custom_metric=custom_eval, evals_result=evals_result, 
#                     early_stopping_rounds=100,
#                     verbose_eval=False)

#     print(f"Best iteration: {bst.best_iteration}")
#     return bst.best_iteration
