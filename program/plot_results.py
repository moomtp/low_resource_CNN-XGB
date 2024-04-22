import glob
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import os


# ---------  sub func  -------------
def find_result_by_model_name(model_name:str , res_file_loc:str) -> List[str] :
    # 定义文件名的模式
    name_pattern = "*_{}.csv".format(model_name)
    # print(name_pattern)
    print(res_file_loc)
    print(res_file_loc + "/" + name_pattern)
    # 使用 glob 模块查找符合模式的文件
    matching_files = glob.glob(res_file_loc + "/" + name_pattern)
    return matching_files

def find_result_by_model_name_V2(model_name: str, res_file_loc: str) -> List[str]:
    # 定義文件名的模式
    name_pattern = "*_{}.csv".format(model_name)
    # print(name_pattern)
    # print(res_file_loc)
    # 使用 glob 模塊查找符合模式的文件
    matching_files = glob.glob(os.path.join(res_file_loc, name_pattern))
    return matching_files

def plot_max_val(df):
    max_y = df['acc'].max()
    max_index = df['acc'].idxmax()  # 获取最大值的索引
    # 在最大值的位置添加标记
    plt.scatter(max_index, max_y, color='red')  # 使用 plt.scatter() 函数添加标记
    plt.text(max_index, max_y, f'{max_y:.4f}', horizontalalignment='left', verticalalignment='bottom')  # 添加标记的数值

def get_baseline_acc_score(CNN_model_name, dataset_name):
    # 定義一個字典，鍵為(dataset_name, CNN_model_name)元組，值為準確率
    accuracy_scores = {
        ("chest CT", "resnet18"): 0.7846,
        ("chest CT", "vgg16"): 0.7689,
        ("skin cancer", "resnet18"): 0.8440, # 0.8440
        ("skin cancer", "vgg16"): 0.8299,
        ("ocularDisease", "resnet18"): 0.6226,
        ("ocularDisease", "vgg16"): 0.6492,
    }

    # 使用get方法從字典中獲取準確率，如果找不到匹配的鍵，則返回一個預設值
    accuracy = accuracy_scores.get((dataset_name, CNN_model_name), "Not found")

    if accuracy == "Not found":
        print("Dataset or model name not found.")
        raise ValueError("Dataset or model name not found.")
    else:
        return accuracy


# def plot_data(*res_files):
#     colors = ['orange', 'green', 'blue', 'red', 'purple', 'brown']  # 預定義顏色列表
#     for i, file in enumerate(res_files):
#         df = pd.read_csv(file)
#         color = colors[i % len(colors)]  # 循環使用顏色
#         label = file.split('_model')[0][2:]  # 從文件名中提取標籤
#         plt.plot(df['model_name'], df['acc'], color=color, label=label)  # 繪製折線圖
#         plot_max_val(df, color)  # 標註最大值
#     plt.legend()
#     plt.xlabel('Model Name')
#     plt.ylabel('Accuracy')
#     plt.title('Model Accuracy Comparison')
#     plt.show()

def plot_label_and_save(file_name:str):
    plt.legend(loc='lower left')
    # plt.xlabel()  # 设置X轴标签
    plt.ylabel('acc')  # 设置Y轴标签
    plt.title(CNN_model_name + '(' + dataset_name + ')')  # 设置图表标题
    plt.xticks(rotation=45)  # 旋转X轴标签以便更好地显示
    plt.grid(True)  # 显示网格
    plt.savefig(file_name)
    plt.show()  # 显示图表


# ---------  main  -------------

if __name__ == "__main__":
    # var setting
    # dataset_name = "chest CT"
    # dataset_name = "skin cancer"
    dataset_name = "ocularDisease"

    # CNN_model_name = 'resnet18'
    CNN_model_name= 'vgg16'

    res_file_loc = "../results/" + dataset_name
    mm_res_file_loc = "../results/" + dataset_name + " MM"



    res_files = find_result_by_model_name(CNN_model_name, res_file_loc)
    mm_res_files = find_result_by_model_name(CNN_model_name, mm_res_file_loc)

    # print(res_files)

    baseline_acc = get_baseline_acc_score(CNN_model_name, dataset_name)



    # load csv data
    df1 = pd.read_csv(res_files[0])
    print("df1 is :" + res_files[0])
    mm_df1 = pd.read_csv(mm_res_files[0])

    df2 = pd.read_csv(res_files[1])
    print("df2 is :" + res_files[1])
    mm_df2 = pd.read_csv(mm_res_files[1])

    df3 = pd.read_csv(res_files[2])
    print("df3 is :" + res_files[2])
    mm_df3 = pd.read_csv(mm_res_files[2])


    # # plot csv data 
    # plt.figure(figsize=(8, 6), dpi=150)
    # min_col_len = min(len(df1['acc']), len(df2['acc']), len(df3['acc']))
    # plt.plot(df1['model_name'][:min_col_len], df1['acc'][:min_col_len], color="orange",label=res_files[0].split('_model')[0][2:])  # 绘制折线图
    # plot_max_val(df1)
    # plt.plot(df2['model_name'][:min_col_len], df2['acc'][:min_col_len], color="green",label=res_files[1].split('_model')[0][2:])  # 绘制折线图
    # plot_max_val(df2)
    # plt.plot(df3['model_name'][:min_col_len], df3['acc'][:min_col_len], color="blue",label=res_files[2].split('_model')[0][2:])  # 绘制折线图
    # plot_max_val(df3)

    # setting dCNN only acc
    # plt.axhline(y=baseline_acc, color='red', linestyle='--')
    # plt.text(9, baseline_acc, f'{baseline_acc:.4f}', color='red', horizontalalignment='right', verticalalignment='bottom')
    # plot_label_and_save(CNN_model_name + "_" + dataset_name + "_three_model.png")


    # # -------  plot three res (w/ MM & w/o MM)  ------- 
    # plt.figure(1,figsize=(10, 6), dpi=150)
    # min_col_len = min(len(df1['acc']),len(mm_df1['acc']))
    # plt.plot(df1['model_name'][:min_col_len], df1['acc'][:min_col_len], color="orange",label=res_files[0].split('_model')[0][2:])  # 绘制折线图
    # plot_max_val(df1)
    # plt.plot(mm_df1['model_name'][:min_col_len], mm_df1['acc'][:min_col_len], color="yellow",label=res_files[0].split('_model')[0][2:]+"_MM")  # 绘制折线图
    # plot_max_val(mm_df1)
    # plot_label_and_save("Linear.png")
    

    # plt.figure(2,figsize=(10, 6), dpi=150)
    # min_col_len = min(len(df2['acc']),len(mm_df2['acc']))
    # plt.plot(df2['model_name'][:min_col_len], df2['acc'][:min_col_len], color="green",label=res_files[1].split('_model')[0][2:])  # 绘制折线图
    # plot_max_val(df2)
    # plt.plot(mm_df2['model_name'][:min_col_len], mm_df2['acc'][:min_col_len], color="#90EE90",label=res_files[1].split('_model')[0][2:]+"_MM")  # 绘制折线图
    # plot_max_val(mm_df2)
    # plot_label_and_save("RF.png")

    plt.figure(3,figsize=(10, 6), dpi=150)
    min_col_len = min(len(df3['acc']),len(mm_df3['acc']))
    plt.plot(df3['model_name'][:min_col_len], df3['acc'][:min_col_len], color="blue",label=res_files[2].split('_model')[0][2:])  # 绘制折线图
    plot_max_val(df3)
    plt.plot(mm_df3['model_name'][:min_col_len], mm_df3['acc'][:min_col_len], color="cyan",label=res_files[2].split('_model')[0][2:]+"_MM")  # 绘制折线图
    plot_max_val(mm_df3)

    # plot_label_and_save("XGB.png")

    plt.axhline(y=baseline_acc, color='red', linestyle='--')
    plt.text(9, baseline_acc, f'{baseline_acc:.4f} (vgg)', color='red', horizontalalignment='right', verticalalignment='bottom')
    resnet_acc = get_baseline_acc_score('resnet18',dataset_name)
    plt.axhline(y=resnet_acc, color='magenta', linestyle='--')
    plt.text(9, resnet_acc , f'{resnet_acc:.4f} (resnet)', color='magenta', horizontalalignment='right', verticalalignment='bottom')
    plot_label_and_save("XGB.png")