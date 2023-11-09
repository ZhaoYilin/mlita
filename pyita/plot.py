import numpy as np
import matplotlib.pyplot as plt

def scatter(y, y_train, y_train_pre, y_test, y_test_pre):
    #plot
    plt.scatter(y_train, y_train_pre,label='Train set', color='b',marker='o',alpha=0.7)
    plt.scatter(y_test, y_test_pre,label='Test set', color='r',marker='o', alpha=0.7)
    plt.legend() #显示图例


    y = np.array(y)
    plt.plot([y.min(),y.max()],
         [y.min(),y.max()],
         '--k',linewidth=1)
    plt.tick_params(labelsize=15)
    plt.xlabel('Calculated',size=15)
    plt.ylabel('Predicted',size=15)
    plt.title('XGBoost',size=15)


    return plt

def bar(x_data, feature_importance):
    #绘制特征重要性图(打印所有描述符)
    indices=np.argsort(feature_importance)
    feature_list=x_data.columns.values
    plt.figure(figsize=(10,15))
    plt.barh(range(len(indices)),feature_importance[indices],align='center')
    plt.yticks(range(len(indices)),feature_list[indices],fontsize=15)
    plt.xticks(fontsize=15)
    plt.xlabel('Importance',size=15)
    plt.title('XGBoost',size=15)
    return plt
