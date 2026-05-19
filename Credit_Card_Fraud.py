import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


#-------------数据分析---------------
df = pd.read_csv('D:/VScode_context/python/creditcard.csv')

# print("-------1.数据预览-------")
# print("形状(行,列):",df.shape)
# print("\n数据前五行:\n")
# print(df.head())

# print("\n-------2.列信息-------\n")
# print(df.dtypes.value_counts())

# print("\n-------3.缺失值检查-------\n")
# print(df.isnull().sum().sum(),"个缺失值")

# print("\n-------4.目标变量分布-------\n")
# print(df['Class'].value_counts(normalize=True))

# print("\n-------5.数值特征基本统计-------\n")
# print(df.describe())


#---------------------建模--------------------

x, y = df.drop('Class', axis = 1), df['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# --------------------随机森林------------------
forest = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # n_estimators:森林中树个数
forest.fit(x_train, y_train)
# yf_pre = forest.predict(x_test)
# print(classification_report(y_test,yf_pre))

# -------------------逻辑回归------------------
# lr= LogisticRegression(max_iter= 1000, random_state=42, class_weight='balanced')  # max_iter:迭代次数
# lr.fit(x_train, y_train)
# yr_pre = lr.predict(x_test)
# print(classification_report(y_test,yr_pre))


# ConfusionMatrixDisplay.from_estimator(forest, x_test, y_test)
# plt.title("Confusion Matrix - Fraud Detection")
# plt.show()
y_prob = forest.predict_proba(x_test)[:, 1]
print(roc_auc_score(y_test, y_prob))


# --------------------conclusion-------------------
# 当前的数据集非常不平衡，正常交易占99.8% 异常占比0.17%, 因此我训练模型时提高了异常交易的权重并且看report也以F1为准，
# 但是实际上看输出结果recall仍然不是很高
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00     85307
#            1       0.96      0.79      0.86       136

#     accuracy                           1.00     85443
#    macro avg       0.98      0.89      0.93     85443
# weighted avg       1.00      1.00      1.00     85443

# 下面是逻辑回归(相当于线性回归基础上，通过Sigmoid 函数把直线压成S型并把线性回归的结果映射到0-1的概率区间，这样根据概率将结果分为了0或1两种)
# 根据结果可以看到相比于随机森林，逻辑回归不太适合这个问题，可能是因为欺诈问题中features太复杂，分布不平衡导致无法通过线性划分结果
#               precision    recall  f1-score   support

#            0       1.00      0.97      0.98     85307
#            1       0.05      0.93      0.09       136

#     accuracy                           0.97     85443
#    macro avg       0.52      0.95      0.54     85443
# weighted avg       1.00      0.97      0.98     85443
