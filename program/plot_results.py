import glob
from typing import List
import pandas as pd
import matplotlib.pyplot as plt


# ---------  sub func  -------------
def find_result_by_model_name(model_name:str , res_file_loc:str) -> List[str] :
    # 定义文件名的模式
    name_pattern = "*_{}.csv".format(model_name)

    # 使用 glob 模块查找符合模式的文件
    matching_files = glob.glob(res_file_loc + "/" + name_pattern)
    return matching_files

def plot_max_val(df):
    max_y = df['acc'].max()
    max_index = df['acc'].idxmax()  # 获取最大值的索引
    # 在最大值的位置添加标记
    plt.scatter(max_index, max_y, color='red')  # 使用 plt.scatter() 函数添加标记
    plt.text(max_index, max_y, f'{max_y:.4f}', horizontalalignment='left', verticalalignment='bottom')  # 添加标记的数值

def get_baseline_acc_score(dataset_name, CNN_model_name):
    # 定義一個字典，鍵為(dataset_name, CNN_model_name)元組，值為準確率
    accuracy_scores = {
        ("chest CT", "resnet18"): 0.7846,
        ("chest CT", "vgg16"): 0.7689,
        ("skin cancer", "resnet18"): 0.8440,
        ("skin cancer", "vgg16"): 0.8299,
    }

    # 使用get方法從字典中獲取準確率，如果找不到匹配的鍵，則返回一個預設值
    accuracy = accuracy_scores.get((dataset_name, CNN_model_name), "Not found")

    if accuracy == "Not found":
        print("Dataset or model name not found.")
        raise ValueError("Dataset or model name not found.")
    else:
        return accuracy


def plot_data(*res_files):
    colors = ['orange', 'green', 'blue', 'red', 'purple', 'brown']  # 預定義顏色列表
    for i, file in enumerate(res_files):
        df = pd.read_csv(file)
        color = colors[i % len(colors)]  # 循環使用顏色
        label = file.split('_model')[0][2:]  # 從文件名中提取標籤
        plt.plot(df['model_name'], df['acc'], color=color, label=label)  # 繪製折線圖
        plot_max_val(df, color)  # 標註最大值
    plt.legend()
    plt.xlabel('Model Name')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.show()

def plot_label():
    plt.legend(loc='lower left')
    # plt.xlabel()  # 设置X轴标签
    plt.ylabel('acc')  # 设置Y轴标签
    plt.title(CNN_model_name + '(Chest CT)')  # 设置图表标题
    plt.xticks(rotation=45)  # 旋转X轴标签以便更好地显示
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图表


# ---------  main  -------------

if __name__ == "__main__":
    # var setting
    dataset_name = "chest CT"
    # dataset_name = "skin cancer"

    # CNN_model_name = 'resnet18'
    CNN_model_name= 'vgg16'

    res_file_loc = "../resutls/" + dataset_name
    mm_res_file_loc = "../resutls/" + dataset_name + " MM"



    res_files = find_result_by_model_name(CNN_model_name, res_file_loc)
    mm_res_files = find_result_by_model_name(CNN_model_name, mm_res_file_loc)



    baseline_acc = get_baseline_acc_score(CNN_model_name, dataset_name)



    # load csv data
    df1 = pd.read_csv(res_files[0])
    mm_df1 = pd.read_csv(mm_res_files[0])
    df2 = pd.read_csv(res_files[1])
    mm_df2 = pd.read_csv(mm_res_files[1])
    df3 = pd.read_csv(res_files[2])
    mm_df3 = pd.read_csv(mm_res_files[2])


    # plot csv data 
    plt.plot(df1['model_name'], df1['acc'], color="orange",label=res_files[0].split('_model')[0][2:])  # 绘制折线图
    plot_max_val(df1)
    plt.plot(df2['model_name'], df2['acc'], color="green",label=res_files[1].split('_model')[0][2:])  # 绘制折线图
    plot_max_val(df2)
    plt.plot(df3['model_name'], df3['acc'], color="blue",label=res_files[2].split('_model')[0][2:])  # 绘制折线图
    plot_max_val(df3)

    # plot three res (w/ MM & w/o MM)
    plt.figure(1)
    plt.plot(df1['model_name'], df1['acc'], color="orange",label=res_files[0].split('_model')[0][2:])  # 绘制折线图
    plt.plot(mm_df1['model_name'], mm_df1['acc'], color="orange",label=res_files[0].split('_model')[0][2:])  # 绘制折线图
    plot_label()
    

    plt.figure(2)
    plt.plot(df2['model_name'], df2['acc'], color="orange",label=res_files[0].split('_model')[0][2:])  # 绘制折线图
    plt.plot(mm_df2['model_name'], mm_df2['acc'], color="orange",label=res_files[0].split('_model')[0][2:])  # 绘制折线图
    plot_label()

    plt.figure(3)
    plt.plot(df3['model_name'], df3['acc'], color="orange",label=res_files[0].split('_model')[0][2:])  # 绘制折线图
    plt.plot(mm_df3['model_name'], mm_df3['acc'], color="orange",label=res_files[0].split('_model')[0][2:])  # 绘制折线图
    plot_label()

    # setting dCNN only acc
    plt.axhline(y=baseline_acc, color='red', linestyle='--')
    plt.text(15, baseline_acc, f'{baseline_acc:.4f}', color='red', horizontalalignment='right', verticalalignment='bottom')
    # plt.ylim(bottom=50)




    # plt.legend(loc='lower left')
    # # plt.xlabel()  # 设置X轴标签
    # plt.ylabel('acc')  # 设置Y轴标签
    # plt.title(CNN_model_name + '(Chest CT)')  # 设置图表标题
    # plt.xticks(rotation=45)  # 旋转X轴标签以便更好地显示
    # plt.grid(True)  # 显示网格
    # plt.show()  # 显示图表