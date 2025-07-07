from joblib import load
import numpy as np

file_path = "/home/zy/桌面/B15_PenOri-2_38000/B15_PenOri-2_38000_20250618_1803.pkl"

try:
    # 使用 joblib 加载文件
    data_dict = load(file_path)

    print("成功读取文件！字典包含以下键:")
    # for key in data_dict.keys():
    #     print(f"- {key} ({type(data_dict[key])})")


    for key in data_dict.keys():
        print(f"- {key} {type(data_dict[key])}\n")
        print(data_dict[key])
except Exception as e:
    print(f"读取文件时出错: {e}")