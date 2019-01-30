import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS




train_sets= pd.read_csv("C:/Users/User/Desktop/train_set.csv" ,sep = '\t')


additional_stop_words = ['said','will','come','says','it','year','he','one','did','think','just']
stop_words = ENGLISH_STOP_WORDS.union(additional_stop_words)



names = set(train_sets["Category"])
allcategories = names
for category in names:
    print(category)

    df = train_sets[train_sets["Category"] == category]
    Ar = np.array(df[["Content"]])
    text = ""

    for i in range(Ar.shape[0]):
        for j in range(Ar.shape[1]):
            text += str(Ar[i, j]) + "j"
    wordcloud = WordCloud(background_color="black", max_words=2000, stopwords=stop_words).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()