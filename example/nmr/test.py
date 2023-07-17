from mlita.ml import *
import numpy as np

model =  MachineLearning.from_csv(['nmr_H.csv','nmr_O.csv'])

#构建模型，超参调整
params={'max_depth':8,
    'n_estimators':29,
    'min_child_weight':8,
    'subsample':0.9,
    'colsample_bytree':0.6,
    'reg_alpha':0.1,
    'reg_lambda':0.7
    }
results = model.xgboost(params=params,save_data=False,save_picture=False)
print(results)


scores = model.make_score(results)
print(scores)

model.to_csv(results)

