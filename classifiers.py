import pandas as pd
import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import  accuracy_score , precision_score , recall_score ,f1_score
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

train_sets= pd.read_csv("C:/Users/User/Desktop/train_set.csv" ,sep = '\t')
test_sets= pd.read_csv("C:/Users/User/Desktop/test_set.csv" ,sep = '\t')
train_based_on = 'Content'


additional_stop_words = ['said','will','come','says','it','year','he','one','did','think','just']
stop_words = ENGLISH_STOP_WORDS.union(additional_stop_words)
count_vect = CountVectorizer(stop_words=stop_words)

X_train_counts =count_vect.fit_transform(train_sets["Content"])
tf_train_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_train_transformer.transform(X_train_counts)

svd = TruncatedSVD(n_components = 100)
X_lsi = svd.fit_transform(X_train_counts)
clfSVD=SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42).fit(X_lsi,train_sets["Content"])

X_test_counts = count_vect.fit_transform(test_sets["Content"])
tf_test_transformer = TfidfTransformer(use_idf=False).fit(X_test_counts)
X_test_tf = tf_test_transformer.transform(X_test_counts)


n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True)
n_try = 0
bnbPrecision = 0
bnbAccuracy = 0
bnbRecall = 0
bnbF_measure = 0

for train_index, test_index in kf.split(train_sets):
    bnb = BernoulliNB()
    cur_X_train_counts = count_vect.transform(np.array(train_sets["Content"])[train_index])
    cur_X_test_counts = count_vect.transform(np.array(train_sets["Content"])[test_index])
    bnb.fit(cur_X_train_counts, train_sets["Category"][train_index])

    my_pred = bnb.predict(cur_X_test_counts)
    bnbPrecision += precision_score(train_sets["Category"][test_index], my_pred, average="macro")
    bnbAccuracy += accuracy_score(train_sets["Category"][test_index], my_pred)
    bnbRecall += recall_score(train_sets["Category"][test_index], my_pred, average="macro")
    bnbF_measure += f1_score(train_sets["Category"][test_index], my_pred, average="macro")

bnbPrecision = bnbPrecision / n_splits
bnbAccuracy = bnbAccuracy / n_splits
bnbRecall = bnbRecall / n_splits
bnbF_measure = bnbF_measure / n_splits

print("Precision is:    ", bnbPrecision)
print("Accuracy  is:    ", bnbAccuracy)
print("Recall    is:    ", bnbRecall)
print("F measure is:    ", bnbF_measure)




n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True)
n_try = 0

rfPrecision = 0
rfAccuracy = 0
rfRecall = 0
rfF_measure = 0
for train_index, test_index in kf.split(train_sets):
    rf_clf = RandomForestClassifier()
    cur_X_train_counts = count_vect.transform(np.array(train_sets["Content"])[train_index])
    cur_X_test_counts = count_vect.transform(np.array(train_sets["Content"])[test_index])
    rf_clf.fit(cur_X_train_counts, train_sets["Category"][train_index])


    my_pred = rf_clf.predict(cur_X_test_counts)
    rfPrecision += precision_score(train_sets["Category"][test_index], my_pred, average="macro")
    rfAccuracy += accuracy_score(train_sets["Category"][test_index], my_pred)
    rfRecall += recall_score(train_sets["Category"][test_index], my_pred, average="macro")
    rfF_measure += f1_score(train_sets["Category"][test_index], my_pred, average="macro")

rfPrecision = rfPrecision / n_splits
rfAccuracy = rfAccuracy / n_splits
rfRecall = rfRecall / n_splits
rfF_measure = rfF_measure / n_splits

print("Precision is:    ", rfPrecision)
print("Accuracy  is:    ", rfAccuracy)
print("Recall    is:    ", rfRecall)
print("F measure is:    ", rfF_measure)




n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True)
n_try = 0

svmPrecision = 0
svmAccuracy = 0
svmRecall = 0
svmF_measure = 0
for train_index, test_index in kf.split(train_sets):
    svm_clf = svm.SVC()
    cur_X_train_counts = count_vect.transform(np.array(train_sets["Content"])[train_index])
    cur_X_test_counts = count_vect.transform(np.array(train_sets["Content"])[test_index])
    svm_clf.fit(cur_X_train_counts, train_sets["Category"][train_index])


    my_pred = svm_clf.predict(cur_X_test_counts)
    svmPrecision += precision_score(train_sets["Category"][test_index], my_pred, average="macro")
    svmAccuracy += accuracy_score(train_sets["Category"][test_index], my_pred)
    svmRecall += recall_score(train_sets["Category"][test_index], my_pred, average="macro")
    svmF_measure += f1_score(train_sets["Category"][test_index], my_pred, average="macro")

svmPrecision = svmPrecision / n_splits
svmAccuracy = svmAccuracy / n_splits
svmRecall = svmRecall / n_splits
svmF_measure = svmF_measure / n_splits

print("Precision is:    ", svmPrecision)
print("Accuracy  is:    ", svmAccuracy)
print("Recall    is:    ", svmRecall)
print("F measure is:    ", svmF_measure)

with open("C:/Users/User/Desktop/EvaluationMetric_10fold.cvs", 'w', ) as classifier:
    classifierWriter = csv.DictWriter(classifier,fieldnames=("Statistic Measure", "Naive Bayes", "Random Forest", "SVM", "KNN"),dialect='excel-tab')
    classifierWriter.writeheader()
    classifierWriter.writerow({"Statistic Measure": "Precision", "Naive Bayes": bnbPrecision, "Random Forest": rfPrecision,"SVM": svmPrecision, "KNN": "0"})
    classifierWriter.writerow({"Statistic Measure": "Accuracy", "Naive Bayes": bnbAccuracy, "Random Forest": rfAccuracy, "SVM": svmAccuracy,"KNN": "0"})
    classifierWriter.writerow({"Statistic Measure": "Recall", "Naive Bayes": bnbRecall, "Random Forest": rfRecall, "SVM": svmRecall,"KNN": "0"})
    classifierWriter.writerow({"Statistic Measure": "F_measure", "Naive Bayes": bnbF_measure, "Random Forest": rfF_measure,"SVM": svmF_measure, "KNN": "0"})

    classifierWriter = csv.excel_tab.delimiter