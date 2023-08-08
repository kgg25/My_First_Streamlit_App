import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix


df = pd.read_csv('methods/datas/iris.csv')

def train_nb_model():
    
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=1)
    NB_model = GaussianNB()
    NB_model.fit(X_train, Y_train)
    Y_pred = NB_model.predict(X_test)
    
    score = metrics.accuracy_score(Y_test, Y_pred)
    report = metrics.classification_report(Y_test, Y_pred)
    conf_matrix = metrics.confusion_matrix(Y_test, Y_pred)
    
    score = metrics.accuracy_score(Y_test, Y_pred)
    return (NB_model, score, report, conf_matrix)
