import glob
from typing import List
import pandas as pd
import matplotlib.pyplot as plt


# ---------  sub func  -------------
def find_result_by_model_name(model_name:str) -> List[str] :
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

# ---------  main  -------------
res_file_loc = "../resutls/chest CT"
# res_file_loc = "../resutls/skin cancer"

# model = 'resnet18'
model = 'vgg16'


res_files = find_result_by_model_name(model)

# baseline_acc = 0.7771  # resnet acc
baseline_acc = 0.7672  # vgg16 acc


# load csv data
df1 = pd.read_csv(res_files[0])
df2 = pd.read_csv(res_files[1])
df3 = pd.read_csv(res_files[2])


# plot csv data 
plt.plot(df1['model_name'], df1['acc'], color="orange",label=res_files[0].split('_model')[0][2:])  # 绘制折线图
plot_max_val(df1)
plt.plot(df2['model_name'], df2['acc'], color="green",label=res_files[1].split('_model')[0][2:])  # 绘制折线图
plot_max_val(df2)
plt.plot(df3['model_name'], df3['acc'], color="blue",label=res_files[2].split('_model')[0][2:])  # 绘制折线图
plot_max_val(df3)

# setting dCNN only acc
plt.axhline(y=baseline_acc, color='red', linestyle='--')
plt.text(15, baseline_acc, f'{baseline_acc:.4f}', color='red', horizontalalignment='right', verticalalignment='bottom')
# plt.ylim(bottom=50)




plt.legend(loc='lower left')
# plt.xlabel()  # 设置X轴标签
plt.ylabel('acc')  # 设置Y轴标签
plt.title(model + '(Chest CT)')  # 设置图表标题
plt.xticks(rotation=45)  # 旋转X轴标签以便更好地显示
plt.grid(True)  # 显示网格
plt.show()  # 显示图表