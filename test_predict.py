
import pandas as pd
import csv
import numpy as np

#import wordcloud from WordCloud
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report , accuracy_score , precision_score , recall_score ,f1_score
from sklearn.model_selection import KFold , cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

train_sets= pd.read_csv("C:/Users/User/Desktop/train_set.csv" ,sep = '\t')

test_sets= pd.read_csv("C:/Users/User/Desktop/test_set.csv" ,sep = '\t')

train_based_on = 'Content'

additional_stop_words = ['said','will','come','says','it','year','he','one','did','think','just']
stop_words = ENGLISH_STOP_WORDS.union(additional_stop_words)
tfid_vect = TfidfVectorizer(stop_words=stop_words)
X_train_counts = tfid_vect.fit_transform(train_sets["Content"])
svd=TruncatedSVD(n_components=100)
X_train_counts=svd.fit_transform(X_train_counts)

X_test_counts = tfid_vect.fit_transform(test_sets["Content"])
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
    cur_X_train_counts = tfid_vect.transform(np.array(train_sets["Content"])[train_index])
    cur_X_test_counts = tfid_vect.transform(np.array(train_sets["Content"])[test_index])
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


testing_data=tfid_vect.transform(np.array(test_sets["Content"]))
bernouli_predictions=bnb.predict(testing_data)
print(bnb.predict(testing_data))

with open ("C:/Users/User/Desktop/testSets_categories.cvs",'w',) as test_predict:
    test_predictWriter = csv.DictWriter(test_predict,  fieldnames=("Id", "Predictions"),dialect='excel-tab')
    test_predictWriter.writeheader()
    for Id,category in zip(test_sets.Id,bernouli_predictions):
        test_predictWriter.writerow({"Id": Id,"Predictions": category})
    test_predictrWriter = csv.excel_tab.delimiter