import glob
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter

# ---------  sub func  -------------
def find_result_by_model_name(model_name:str , res_file_loc:str) -> List[str] :
    # 定义文件名的模式
    name_pattern = "XGB*_{}.csv".format(model_name)
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

def get_baseline_acc_score(CNN_model_name, dataset_name):
    # 定義一個字典，鍵為(dataset_name, CNN_model_name)元組，值為準確率
    accuracy_scores = {
        ("chest CT", "resnet18"): 0.7846,
        ("chest CT", "vgg16"): 0.7689,
        ("HAM10000", "resnet18"): 0.8440, # 0.8440
        ("HAM10000", "vgg16"): 0.8299,
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

def percent_formatter(x, _):
    return f'{x:.0f}%'  # Format y-axis values with % symbol

def layer_to_funcblock_info(res_df):
    
    return 0

# -----------------  plot helper functions  -------------------

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


def plot_multiple_lines_with_inset(x, ys, labels, colors, title='Multiple Line Chart with Inset',
                                   xlabel='X-axis', ylabel='Y-axis', inset_position=[0.5, 0.5, 0.4, 0.4],
                                   grid=True, file_name:str="fig.png"):
    # Ensure x-axis labels are strings
    x = x.tolist() if hasattr(x, 'tolist') else x

    fig, ax = plt.subplots(figsize=(8, 6))

    for y, label, color in zip(ys, labels, colors):
        y = y.tolist() if hasattr(y, 'tolist') else y
        ax.plot(x, y, marker='o', label=label, color=color)
    
    ax.set_title(title)
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    
    if grid:
        ax.grid(True)

    ax.legend()

    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x,rotation=45)

    # Format y-axis values with % symbol
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

    # Add inset window to show only colors and labels
    ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper right', bbox_to_anchor=inset_position, bbox_transform=ax.figure.transFigure)
    
    for label, color in zip(labels, colors):
        ax_inset.plot([], [], label=label, color=color, marker='o', linestyle='')  # Only add legend items

    # ax_inset.legend()
    ax_inset.axis('off')  # Hide axes
    plt.savefig(file_name)
    plt.show()

def plot_label_and_save (file_name:str , y_label:str):
    plt.legend(loc='lower left')
    # plt.xlabel()  # 设置X轴标签
    plt.ylabel('acc')  # 设置Y轴标签
    plt.title(CNN_model_name + '(' + dataset_name + ')')  # 设置图表标题
    plt.xticks(rotation=45)  # 旋转X轴标签以便更好地显示
    plt.grid(True)  # 显示网格
    plt.savefig(file_name)
    plt.show()  # 显示图表


def plot_max_val(df):
    max_y = df['acc'].max()
    max_index = df['acc'].idxmax()  # 获取最大值的索引
    # 在最大值的位置添加标记
    plt.scatter(max_index, max_y, color='red')  # 使用 plt.scatter() 函数添加标记
    plt.text(max_index, max_y, f'{max_y:.4f}', horizontalalignment='left', verticalalignment='bottom')  # 添加标记的数值

# =============  main  ===============

if __name__ == "__main__":
    # var setting
    # dataset_name = "chest CT"
    # dataset_name = "HAM10000"
    dataset_name = "ocularDisease"

    # CNN_model_name = 'resnet18'
    CNN_model_name= 'vgg16'

    res_file_loc = "../results/" + dataset_name
    mm_res_file_loc = "../results/" + dataset_name + " MM"



    res_files = find_result_by_model_name(CNN_model_name, res_file_loc)
    mm_res_files = find_result_by_model_name(CNN_model_name, mm_res_file_loc)


    baseline_acc = get_baseline_acc_score(CNN_model_name, dataset_name)



    # load csv data
    df_cnn_xgb = pd.read_csv(res_files[0])
    print("df1 is :" + res_files[0])
    df_mm_cnn_xgb = pd.read_csv(mm_res_files[0])

    if dataset_name == "HAM10000":
        dataset_name = "skin cancer"
    if dataset_name == 'resnet18':
        df_cnn_xgb = layer_to_funcblock_info()


    # Acc only

    x = df_cnn_xgb["model_name"]
    y = [df_cnn_xgb['Acc'], df_mm_cnn_xgb['acc']*100]
    label = ['before MM', 'after MM']
    color = ['red', '#FF69B4']

    plot_multiple_lines_with_inset(x, y, label, color, title=f"Accuracy", file_name="Accuracy")

    x = df_cnn_xgb["model_name"]
    y = [df_cnn_xgb['F1'], df_mm_cnn_xgb['f1']*100]
    label = ['before MM', 'after MM']
    color = ['orange', 'yellow']

    plot_multiple_lines_with_inset(x, y, label, color, title=f"F1-score", file_name="F1-score")