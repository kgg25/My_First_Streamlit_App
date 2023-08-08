import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics


df = pd.read_csv('methods/datas/iris.csv')

def train_knn_model():
    
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=1)
    KNN_model = KNeighborsClassifier(n_neighbors=10)
    KNN_model.fit(X_train, Y_train)
    Y_pred = KNN_model.predict(X_test)
    
    score = metrics.accuracy_score(Y_test, Y_pred)
    report = metrics.classification_report(Y_test, Y_pred)
    conf_matrix = metrics.confusion_matrix(Y_test, Y_pred)
    
    return (KNN_model, score, report, conf_matrix)
