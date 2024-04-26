
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
# 加载数据
# iris = datasets.load_iris()


eval_data = np.load('/Users/huzuwang/我的雲端硬碟/code/python/vici_holdings_test/Test/eval_data.npy')
eval_labels = np.load('/Users/huzuwang/我的雲端硬碟/code/python/vici_holdings_test/Test/eval_labels.npy')
train_data = np.load('/Users/huzuwang/我的雲端硬碟/code/python/vici_holdings_test/Test/train_data.npy')
train_labels = np.load('/Users/huzuwang/我的雲端硬碟/code/python/vici_holdings_test/Test/train_labels.npy')

print(eval_data)
# df_eval_data = pd.DataFrame(eval_data)
# df_eval_labels = pd.DataFrame(eval_labels)
# df_train_data = pd.DataFrame(train_data)
# df_train_labels = pd.DataFrame(train_labels)

# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
#
# # 转换为Dataset数据格式
# train_data = lgb.Dataset(X_train, label=y_train)
# validation_data = lgb.Dataset(X_test, label=y_test)
#
# # 参数
# params = {
#     'learning_rate': 0.1,
#     'lambda_l1': 0.1,
#     'lambda_l2': 0.2,
#     'max_depth': 4,
#     'objective': 'multiclass',  # 目标函数
#     'num_class': 3,
# }
#
# # 模型训练
# gbm = lgb.train(params, train_data, valid_sets=[validation_data])
#
# # 模型预测
# y_pred = gbm.predict(X_test)
# y_pred = [list(x).index(max(x)) for x in y_pred]
# print(y_pred)
#
# # 模型评估
# print(accuracy_score(y_test, y_pred))