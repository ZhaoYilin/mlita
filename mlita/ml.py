import math
import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler,StandardScaler

#from plot import *
from mlita.plot import *


class MachineLearning(object):
    """ machine learning 
    """

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    @classmethod
    def from_csv(cls, file):
        """
        """
        if isinstance(file, list):
            data_list = []
            for f in file:
                data_i = pd.read_csv(f)
                data_list.append(data_i)
            data = pd.concat(data_list)
            print(data) 
   
        else:
            data = pd.read_csv(file)

        x_data = data.iloc[:,0:-1].to_numpy()
        y_data = data.iloc[:,-1].to_numpy()

        obj = cls(x_data, y_data)
        return obj
        
    def xgboost(self, params=None, save_data=True, save_picture=True):
        """ XGBoost (eXtreme Gradient Boosting) implementation.

        Parameters
        ----------
        x_data : ndarray
         
        y_data : ndarray

        params : dict

        save : bool
            If save the results.

        Return
        ------
        """
        x_data = self.x_data
        y_data = self.y_data

        if params==None:
            #构建模型，超参调整
            params={'max_depth':8,
                'n_estimators':29,
                'min_child_weight':8,
                'subsample':0.9,
                'colsample_bytree':0.6,
                'reg_alpha':0.1,
                'reg_lambda':0.7
                }

        # Splitting the dataset into the Training set and Test set
        x_train, x_test, y_train, y_test=train_test_split(x_data, y_data, random_state= 3, test_size=0.15, shuffle=True)
        sc=MinMaxScaler()
        x_train_std=sc.fit_transform(x_train)
        x_test_std=sc.transform(x_test)

        # Fitting XGBoost to the Training set
        clf=XGBRegressor(**params)
        clf.fit(np.array(x_train_std),y_train)
        y_train_pre=clf.predict(x_train_std)
        y_test_pre=clf.predict(x_test_std)
        feature_importance=clf.feature_importances_

        results = {
            'x_data': x_data,
            'y_data': y_data,
            'y_train': y_train,
            'y_train_pre': y_train_pre,
            'y_test': y_test,
            'y_test_pre': y_test_pre,
            'feature_importance': feature_importance
            }               
        return results

    def make_score(self, results):
        """make score
        """
        a, b = results['y_test'], results['y_test_pre']
        mse = mean_squared_error(a,b)
        mae = mean_absolute_error(a,b)
        r2 = r2_score(a,b)
        r = np.sum((a-np.average(a))*(b-np.average(b)))/math.sqrt(np.sum((a-np.average(a))**2)*np.sum((b-np.average(b))**2))

        scores = {'mse': mse,
            'mae': mae,
            'r2': r2,
            'r': r
            }               
        return scores 

    def to_csv(self, results):
        #预测值重新放到一个表格中
        for key, value in results.items():
            print(key+'.csv')
            if key=='feature_importance':
                # 将特征重要性信息放到Excel表格中
                fea=pd.DataFrame()
                fea['value']= value 
                fea_0=fea.loc[fea['value'] > 0.05]
                fea['value'].to_csv(key+'.csv')
            else:
                pd_value = pd.DataFrame(value)
                pd_value.to_csv(key+'.csv')

    def save_picture(self):
        if save_picture==True:
            fig1 = scatter(y_data, y_train, y_train_pre, y_test, y_test_pre)
            fig1.savefig('scatter_plot')
            fig2 = bar(x_data, feature_importance)
            fig2.savefig('bar_plot')

        return True

