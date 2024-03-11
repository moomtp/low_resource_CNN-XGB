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


# ---------  main  -------------
res_file_loc = "./"
model = 'resnet18'
# model = 'vgg16'

target_csv = "Linear_model_results_resnet18.csv"

res_files = find_result_by_model_name(model)

baseline_acc = 0.79


# load csv data
df1 = pd.read_csv(res_files[0])
df2 = pd.read_csv(res_files[1])
df3 = pd.read_csv(res_files[2])


# plot csv data 
plt.plot(df1['model_name'], df1['acc'], color="yellow",label=res_files[0].split('_model')[0][:3])  # 绘制折线图
plt.plot(df2['model_name'], df2['acc'], color="green",label=res_files[1].split('_model')[0][:3])  # 绘制折线图
plt.plot(df3['model_name'], df3['acc'], color="blue",label=res_files[2].split('_model')[0][:3])  # 绘制折线图

# setting dCNN only acc
plt.axhline(y=baseline_acc, color='red', linestyle='--')
plt.ylim=(0,12)


plt.legend(loc='lower left')
plt.xlabel('root')  # 设置X轴标签
plt.ylabel('acc')  # 设置Y轴标签
plt.title('title')  # 设置图表标题
plt.xticks(rotation=45)  # 旋转X轴标签以便更好地显示
plt.grid(True)  # 显示网格
plt.show()  # 显示图表