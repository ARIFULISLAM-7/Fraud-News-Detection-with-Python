# import prerequisite libraries
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# reading the data
df = pd.read_csv('Fraud News Detection\\news.csv')
# getting shape
df.shape
df.head()
# geting level from dataframe - DataFlair
labels = df.label
labels.head()
# split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, 
                                                    test_size=0.2, random_state=7)
# fit and transform the vectorizer on the train set, 
# and transform the vectorizer on the test set.
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)
# calculate the accuracy with accuracy_score() from sklearn.metrics.
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score * 100, 2)}%')

# confusion matrix to gain insight into the number of false and true
# negatives and positives.
cm = confusion_matrix (y_test, y_pred, labels = ['FAKE', 'REAL'])
print(cm)

# visualize a heatmap.
import seaborn as sns
import matplotlib.pyplot as plt
# create a heatmap.
sns.heatmap(cm, annot=True, fmt = 'd', cmap = 'Blues',
            xticklabels = ['FAKE', 'REAL'],
            yticklabels = ['FAKE', 'REAL'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.show()
