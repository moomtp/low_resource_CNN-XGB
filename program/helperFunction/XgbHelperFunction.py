from time import time
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pandas as pd
from torch import nn
from .helperFunctions import dataloaderToFeatureData
from sklearn.metrics import accuracy_score, f1_score , log_loss
import xgboost as xgb
import numpy as np
import torch


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


def csvkeylistToData(csv_path:str, keys:list ):
    ''' 讀取csv檔的filename資料, return為list'''
    datas = []
    df = pd.read_csv(csv_path)
    for key in keys:
      data = (df[df['filename'] == key].iloc[0])
      datas.append(data.drop(labels='filename'))

    return datas



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
    # return [('Accuracy', acc), ('F1-score', f1), ('mlogloss', mlogloss)] # 可以更換成這個，但early stop會變成f1 score

def calBestIterOfXGB( train_features, train_labels, test_features, test_labels, enable_f1_metric=True):
    """算出什麼時候會overfitting"""
    dtrain = xgb.DMatrix(train_features, label=train_labels)
    dval = xgb.DMatrix(test_features, label=test_labels)

    # 设置参数，注意要更改 'objective' 为多分类的目标函数
    num_class = len(np.unique(train_labels))  # 获取类别数
    params = {'eval_metric': 'mlogloss', 'objective': 'multi:softprob', 'num_class': num_class}



    evals_result = {}
    bst = xgb.train(params, dtrain, num_boost_round=105, 
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    custom_metric=custom_eval, evals_result=evals_result, 
                    early_stopping_rounds=20,
                    verbose_eval=False)

    print(f"Best iteration: {bst.best_iteration}")
    if enable_f1_metric:
       return bst.best_iteration, evals_result['val']['F1-score'], evals_result['val']['Accuracy']
    return bst.best_iteration

def calBestIterOfXGBByF1score( train_features, train_labels, test_features, test_labels):
    """算出什麼時候會overfitting"""
    dtrain = xgb.DMatrix(train_features, label=train_labels)
    dval = xgb.DMatrix(test_features, label=test_labels)

    # 设置参数，注意要更改 'objective' 为多分类的目标函数
    num_class = len(np.unique(train_labels))  # 获取类别数
    params = {'eval_metric': 'mlogloss', 'objective': 'multi:softprob', 'num_class': num_class}



    evals_result = {}
    bst = xgb.train(params, dtrain, num_boost_round=105, 
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    custom_metric=custom_eval, evals_result=evals_result, 
                    early_stopping_rounds=10,
                    verbose_eval=False)

    print(f"Best iteration: {bst.best_iteration}")
    return bst.best_iteration