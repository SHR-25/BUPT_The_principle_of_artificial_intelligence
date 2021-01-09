#SVC
import numpy as np
import os
import time
import codecs
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
import datetime
import pickle
import os
import sys
import pandas as pd
import numpy as np
from sklearn.utils import Bunch
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import brier_score_loss

bunch_train = Bunch()
bunch_test = Bunch()

# 从本地加载数据到内存
with open('G:\AI\Bunch_1w\Bunch_train', 'rb') as file_obj:  # word_bag_filepath为本地文件路径
    bunch_train = pickle.load(file_obj)
with open('G:\AI\Bunch_1w\Bunch_test', 'rb') as file_obj_2:  # word_bag_filepath为本地文件路径
    bunch_test = pickle.load(file_obj_2)

def metrics_result(actual,predict):
    print("精度：{0:.3f}".format(metrics.precision_score(actual,predict,average='micro')))
    print("召回：{0:0.3f}".format(metrics.recall_score(actual,predict,average='micro')))
    print("f1-score:{0:.3f}".format(metrics.f1_score(actual,predict,average='micro')))

X_train = bunch_train.label
y_train = bunch_train.tfidf_weight_matrics
clf = LinearSVC(C=1, tol=1e-5)
begintime_train = datetime.datetime.now()
clf.fit(y_train, X_train)
endtime_train = datetime.datetime.now()
print("训练完毕，训练时长为：" + str((endtime_train - begintime_train).seconds)+ "秒")

begintime_test = datetime.datetime.now()
predicted = clf.predict(bunch_test.tfidf_weight_matrics)
metrics_result(bunch_test.label, predicted)
endtime_test = datetime.datetime.now()
print("预测完毕，预测时长为：" + str((endtime_test - begintime_test).seconds)+ "秒")

print(classification_report(bunch_test.label, predicted))	# 打印结果

types = ['IT', '住房', '体育','军事','娱乐','教育','汽车','犯罪','能源','股票']

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
confusion_matrix = pd.DataFrame(confusion_matrix(bunch_test.label, predicted), columns=types, index=types)
print(confusion_matrix)