#3.3
import numpy as np
from cross_validation_error import cross_validation_error, compute_error
from sklearn.svm import SVC

def SVM_results(X_train, y_train, X_test, y_test):
    linear_models = {"SVM_linear":SVC(kernel='linear')}

    degree_values = [2,4,6,8,10]
    poly_models = {"SVM_poly_"+str(d):SVC(kernel='poly', degree=d) for d in degree_values}

    gamma_values = [0.001,0.01,0.1,1.0,10]
    rbf_models = {"SVM_rbf_"+str(gamma):SVC(kernel='rbf',gamma=gamma) for gamma in gamma_values}

    models = {}
    models.update(linear_models)
    models.update(poly_models)
    models.update(rbf_models)

    result_dictionary = {}
    for index, (name, model) in enumerate(models.items()):
        print(index, ". Training ", name)
        average_train_error, average_validation_error = cross_validation_error(X_train, y_train, model, 5)
        print("Cross validation done")
        model.fit(X_train,y_train)
        print("Normal training done")
        y_pred = model.predict(X_test)
        test_error = compute_error(y_pred,y_test)
        result_dictionary[name] = average_train_error, average_validation_error, test_error

    return result_dictionary

if __name__ == "__main__":
    SVM_results("","","","")