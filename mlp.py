import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pickle

def fit_predict(X_train, y_train, X_test, y_test, X_eva, args={}):
    clf = MLPClassifier(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print( confusion_matrix(y_test, y_pred))
    print( classification_report(y_test, y_pred, digits=4))
    #Dump the model
    pickle_file = 'mlp__model.pickle'
    pickle.dump(clf, open(pickle_file,"wb"))
    y_eva_mlp = clf.predict(X_eva)
    return y_pred,y_eva_mlp

