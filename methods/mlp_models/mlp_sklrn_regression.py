import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


boston = pd.read_csv('methods/datas/boston.csv')

def train_mlp_sklrn_regr():
    
    X = boston.iloc[:, :-1]
    Y = boston.iloc[:, -1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=123)
    
    mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, 100, 50),
                               max_iter=300,
                               activation='relu',
                               solver='adam',
                               random_state=123
                               )
    
    mlp_regressor.fit(X_train, Y_train)

    Y_pred = mlp_regressor.predict(X_test)

    score = r2_score(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    loss_curve = mlp_regressor.loss_curve_
    
    return(mlp_regressor, score, mse, loss_curve)


