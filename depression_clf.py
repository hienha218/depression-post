from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn.feature_extraction import text

data = pd.read_csv("depression_dataset_reddit_cleaned.csv")
data.rename({'clean_text':'text', 'is_depression':'depression'}, axis='columns', inplace=True)
data

# remove stop words (and some extras floating letters because the cleaned dataset got rid of punctuations)
# tokenize each text block and get a matrix of token count
stop_words = text.ENGLISH_STOP_WORDS.union("wa", "was", "didn","t", "m", "s", "don", "ve")
f = feature_extraction.text.CountVectorizer(stop_words=stop_words)
X = f.fit_transform(data['text'])

# randomly split into training dataset (80%) and test dataset (20%)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['depression'], test_size=0.2, random_state=42)
# randomly split the training dataset into training (60%) and validation dataset (20%)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25, random_state=42)


# SVM
# By default, scikit's SVC uses radial basis function (RBF) kernel, C=1.0 and it gave really good results
# so I just test different C values to see if it can get any better
print("SVM results:")
list_C = np.arange(1, 100, 10)
score_train = np.zeros(len(list_C))
score_val = np.zeros(len(list_C))
count = 0
# train various models with different Cs then test on the validation dataset
for C in list_C:
    svc = svm.SVC(C=C)
    svc.fit(X_train, y_train)
    score_train[count] = svc.score(X_train, y_train)
    score_val[count]= svc.score(X_val, y_val)
    count = count + 1

# show models' accuracy scores
matrix = np.matrix(np.c_[list_C, score_train, score_val])
models = pd.DataFrame(data = matrix, columns = ['C', 'Train Accuracy', 'Validation Accuracy'])
models.head(n=10)

# choose the model with the highest test accuracy score (the best C=11)
best_index = models['Validation Accuracy'].idxmax()
C = models.iloc[best_index, :][0]

# build the model with the best C value
svc = svm.SVC(C=C)
svc.fit(X_train, y_train)
# show details of the prediction from the chosen model
m_confusion_test = metrics.confusion_matrix(y_test, svc.predict(X_test))
display(pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1']))

# these rates are hard-coded, but they should not change unless you change the random_state values above
print("Correctly identified",(771/(771+12))*100, "% of non-depression posts.")
print("Correctly identified",(705/(59+705))*100, "% of depression posts.")



# Naive Bayes
# Multinomial NB of sklearn have alpha value to handle words that does not exist in the training dataset
# Similar to SVM, I just plug in various alpha values to train then determine which model is the most robust
print("Naive Bayes results:")
list_alpha = np.arange(0.01, 20, 0.10)
score_train = np.zeros(len(list_alpha))
score_val = np.zeros(len(list_alpha))
count = 0
for alpha in list_alpha:
    bayes = naive_bayes.MultinomialNB(alpha=alpha)
    bayes.fit(X_train, y_train)
    score_train[count] = bayes.score(X_train, y_train)
    score_val[count]= bayes.score(X_val, y_val)
    count = count + 1 

matrix = np.matrix(np.c_[list_alpha, score_train, score_val])
models = pd.DataFrame(data = matrix, columns = 
             ['alpha', 'Train Accuracy', 'Validation Accuracy'])
models.head(n=10)

# choose the model with the highest validation accuracy point (chosen alpha=0.11)
best_index = models['Validation Accuracy'].idxmax()
alpha = models.iloc[best_index, :][0]

# build the model with the chosen alpha value
bayes = naive_bayes.MultinomialNB(alpha=alpha)
bayes.fit(X_train, y_train)
# detailed results when use on test dataset
m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test))
display(pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1']))
print("Correctly identified",(679/(679+104))*100, "% of non-depression posts.")
print("Correctly identified",(736/(28+736))*100, "% of depression posts.")