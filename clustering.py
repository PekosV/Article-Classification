import pandas as pd
import csv
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.cluster import KMeansClusterer , cosine_distance
from sklearn.feature_extraction.text import TfidfVectorizer

class Clusters_Category:
    def __init__(self):
        self.football=0
        self.technology=0
        self.business=0
        self.politics=0
        self.films=0
        self.all_articles=0

    def category_count(self,category):
        self.all_articles += 1
        if category=="Technology":
            self.technology += 1
        elif category=="Politics":
            self.politics += 1
        elif category== "Film":
            self.films += 1
        elif category== "Business":
            self.business += 1
        elif category== "Football":
            self.football += 1
        return


train_sets= pd.read_csv("C:/Users/User/Desktop/train_set.csv" ,sep = '\t')



additional_stop_words = ['said','will','come','says','it','year','he','one','did','think','just']
stop_words = ENGLISH_STOP_WORDS.union(additional_stop_words)
tfid_vect = TfidfVectorizer(stop_words=stop_words)
X_train_counts = tfid_vect.fit_transform(train_sets["Content"])
svd=TruncatedSVD(n_components=100)
X_train_counts=svd.fit_transform(X_train_counts)


clusterer = KMeansClusterer(5, cosine_distance, repeats=10)
clusters = clusterer.cluster(X_train_counts, True)

cluster_list= [Clusters_Category() for i in range (0,5)]
category_counter=0
for i in clusters:

    cluster_list[i].category_count(train_sets.Category[category_counter])
    category_counter += 1



with open ("C:/Users/User/Desktop/clustering_KMeans.cvs",'w',) as clustering:

    clusteringWriter = csv.DictWriter (clustering, fieldnames=("Clusters", "Technology", "Politics", "Film", "Business", "Football"),dialect='excel-tab' )
    clusteringWriter.writeheader()

    avg_football=[cluster_list[i].football/cluster_list[i].all_articles for i in range (0,5)]
    avg_business = [cluster_list[i].business / cluster_list[i].all_articles for i in range (0,5)]
    avg_films = [cluster_list[i].films / cluster_list[i].all_articles for i in range (0,5)]
    avg_politics = [cluster_list[i].politics / cluster_list[i].all_articles for i in range (0,5)]
    avg_technology = [cluster_list[i].technology / cluster_list[i].all_articles for i in range (0,5)]

    for i in range (0,5):
        round_avg_technology=round( avg_technology[i],2)
        round_avg_football=round(avg_football[i],2)
        round_avg_films=round(avg_films[i],2)
        round_avg_business=round (avg_business[i],2)
        round_avg_politics=round (avg_politics[i],2)
        clusteringWriter.writerow({"Clusters": "Cluster" , "Technology": round_avg_technology, "Politics":round_avg_politics, "Film":round_avg_films, "Business": round_avg_business, "Football" : round_avg_football})

    clusteringWriter = csv.excel_tab.delimiter