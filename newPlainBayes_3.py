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
import threading

bunch_train = Bunch()
bunch_test = Bunch()

# 从本地加载数据到内存
with open('./data/Bunch_train', 'rb') as file_obj:  # word_bag_filepath为本地文件路径
    bunch_train = pickle.load(file_obj)
with open('./data/Bunch_test', 'rb') as file_obj_2:  # word_bag_filepath为本地文件路径
    bunch_test = pickle.load(file_obj_2)


#Bunch(label = bunch.label,	# 所有文章的标签
#                        filenames,	# 所有文章的文件名
#                        tfidf_weight_matrics = [],# TF-IDF权重矩阵
#                        vocabulary = {})
#

#矩阵的格式为 (0,5458,0.14573526)指的是，第0+1=1篇文章，在词典第5458个词上有权值，权值为0.145..

#各类在训练集中的概率都是0.1，所以朴素贝叶斯中不予计算

#求这个max{ (ln(n1 * P(x1|yi) + 1) + ln(n2 * P(x2|yi) + 1) + …… + ln(nn * P(xn|yi) + 1)) * P(yi) }
#对于第j个词：第j个词出现次数*P（xj|第i类）然后把每类文本所有这些 乘起来
#为了避免之前说的 可能为0，每个ln+1，测试集的矩阵val，就等价于出现次数

m=bunch_train.tfidf_weight_matrics.toarray();
n=bunch_test.tfidf_weight_matrics.toarray();
P = np.zeros((1000,10)) #P[j][i] = P（xj|第i类）


outp=np.zeros((50000,2))
outs=np.zeros((((((((((10,10))))))))))

threadLock = threading.Lock()
threads = []

class myThread (threading.Thread):
    def __init__(self, beginPos):
        threading.Thread.__init__(self)
        self.beginPos = beginPos
    def run(self):
        # 对于每一篇文章遍历，然后对每一个词遍历，去找这个词在训练集中的概率
        # 也就是对训练集中所有该列，分10类遍历，求最大
        for w in range(self.beginPos, self.beginPos + 5000):
            print("正在预测第" + str(w) + "篇")
            s = np.zeros(10)
            for i in range(0, 1000):
                for j in range(0, 10):
                    s[j] += np.log(n[w][i] * P[i][j] + 1)
            pos = GetPos(s, (np.max(s)))
            threadLock.acquire()
            outp[w][0] = int(w / 5000)
            outp[w][1] = int(pos)
            outs[int(w / 5000)][int(pos)] += 1
            threadLock.release()
            print("预测类别:" + str(int(pos)) + ', 实际类别:' + str(int(w / 5000)))


def GetPos(shuzu,zuida):
    i=0
    objx=tuple(shuzu)
    for i in range(len(objx)):
        if(objx[i]==zuida):
            return i

def init():
    for j in range(1000):   #遍历词汇
        tot_sum = 0.0
        type_sum = np.zeros(10)
        for i in range(50000):  #遍历文章
            now_type = int(i/5000)
            now_val = m[i][j]
            type_sum[now_type] += now_val
            tot_sum += now_val
        for t in range(10):
            print("第" + str(j) + "类词在第" + str(t) + "类文本中出现的概率为" ,type_sum[t] / tot_sum)
            P[j][t] = type_sum[t] / tot_sum


begintime = datetime.datetime.now()
init()
for i in range(10):
    threads.append(myThread(5000 * i))
    threads[i].start()
for i in range(10):
    threads[i].join()

endtime = datetime.datetime.now()
print("执行完毕，用时为：" + str((endtime - begintime).seconds)+ "秒")

types = ['IT', '住房', '体育','军事','娱乐','教育','汽车','犯罪','能源','股票']

for op1 in range(0,10):
    print(types[op1]+"   ",end="")
print(" ")

for op1 in range(0,10):
    print(types[op1]+" ",end="")
    for op2 in range(0,10):
        print(str(int(outs[op1][op2]))+" ",end="")
    print(" ")

success=0
for op1 in range(0,10):
    success+=outs[op1][op1]

print(success)

#total=(success*1.0)/50000
#recall=(1.0*success)/50000

#f1=2*(((success*1.0)/50000)*((success*1.0)/50000))/(((success*1.0)/50000)+((success*1.0)/50000))
total=0
totalRecall=0
f1=0

for op1 in range(0,10):
    recall=outs[op1][op1]/5000
    totalRecall+=recall
    sum=0
    for i in range(0,10):
        sum+=outs[op1][i]
    sp=outs[op1][op1]/sum
    total+=sp
    f=(2.0*recall*sp)/(sp+recall)
    f1+=f
    print(types[op1] + "类准确率为：" + str(sp)+"，召回率为："+str(recall)+"，f1-score为："+str(f))

totalRecall=(totalRecall*1.0)/10
total=(total*1.0)/10
f1=(f1*1.0)/10

print("总体正确率为："+str(total))
print("召回率为："+str(totalRecall))
print("F1 Score = "+ str(f1))