import pandas as pd

def CreateFileLableDict(groundtruth_file : str):
    groundtruth_data = pd.read_csv(groundtruth_file)

    def find_indices_of_ones(row):
        # 尋找前六個元素中 1 的位置
        return [(i-1) for i, x in enumerate(row[:8]) if x == 1]

    # 將函數應用於每行並創建新列 'label'
    groundtruth_data['label'] = groundtruth_data.apply(find_indices_of_ones, axis=1)

    # groundtruth_data.head()

    # find ele == 1
    filename_to_label_dict = groundtruth_data.set_index('image')['label'].to_dict()

    # {'1234' : [2] ,'1235' : [3] } -> {'1234' : 2 ,'1235' : 3 }
    filename_to_label_dict =  {key: value[0] if value else None for key, value in filename_to_label_dict.items()}

    type(filename_to_label_dict['ISIC_0024306'])

    # {'1234' : 2 ,'1235' : 3 } -> {'1234.jpg' : 2 ,'1235.jpg' : 3 }
    filename_to_label_dict =  {key + ".jpg": value for key, value in filename_to_label_dict.items()}




    # check if value is out of range
    null_keys = [key for key, value in filename_to_label_dict.items() if value is None]


    all_values_in_range = all(0 <= value <= 6 for value in filename_to_label_dict.values())

    # 輸出結果
    print("所有的值都在範圍內：" if all_values_in_range else "有些值不在範圍內。")

    if(all_values_in_range):
        return filename_to_label_dict   
    else:
        return None