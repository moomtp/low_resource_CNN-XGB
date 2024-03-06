import pandas as pd

# 載入CSV檔案
feature_file_path = '../data/img_feature.csv'
model_file_path =  '../data/resnet_output.csv'

df1 = pd.read_csv(feature_file_path)
df2 = pd.read_csv(model_file_path)

# 顯示檔案的前幾行以了解其結構
df1_head = df1.head()
df2_head = df2.head()

df1_head, df2_head


# 合併檔案，使用 file_name 作為鍵
merged_df = pd.merge(df1, df2, on='file_name', how='inner')

# 儲存合併後的檔案
output_file_path = '../data/merged_img_resnet_features.csv'
merged_df.to_csv(output_file_path, index=False)

# 顯示合併後檔案的前幾行
merged_head = merged_df.head()
merged_head, output_file_path