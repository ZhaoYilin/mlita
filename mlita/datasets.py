import pandas as pd
import numpy as np
from sklearn.datasets.base import Bunch

def load_rbb():
    """
    获取双色球数据集
    :return:
    """
    data_csv = pd.read_csv("crb.csv", header=None)
    rbb = Bunch()
    rbb.data = _get_rbbdata(data_csv)
    rbb.target = _get_rbbtarget(data_csv)
    rbb.DESCR = _get_rbbdescr(data_csv)
    rbb.feature_names = _get_feature_names()
    rbb.target_names = _get_target_names()

    return rbb
