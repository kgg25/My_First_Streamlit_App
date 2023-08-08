import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('methods/datas/iris.csv')

label_encoder = LabelEncoder()
to_predict = label_encoder.fit_transform(data['Species'])

X = data.iloc[:, :-1]
Y = pd.Series(to_predict)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=10)

mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 100, 50),
                               max_iter=300,
                               activation='relu',
                               solver='adam'
                               )

mlp_classifier.fit(X_train, Y_train)

Y_pred = mlp_classifier.predict(X_test)

score = accuracy_score(Y_test, Y_pred)

conf_matrix = confusion_matrix(Y_test, Y_pred)

cls_report = classification_report(Y_test, Y_pred)

loss_curve = mlp_classifier.loss_curve_