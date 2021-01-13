import numpy as np

def compute_error(model_prediction, truth):
    correct_pairs = [(pred, test) for (pred, test) in zip(model_prediction, truth) if pred == test]
    accuracy = len(correct_pairs) / len(truth)
    error = 1.0-accuracy
    return error

def cross_validation_error(X, y, model, folds):
    average_train_error = 0
    average_val_error = 0
    (m, d) = X.shape
    (my, ) = y.shape
    if not (m == my and  m > folds):
        raise AttributeError("parameters don't suffice")
    batch_size = int(m/folds)

    for i in range(folds):
        val_start = batch_size*i
        val_end = batch_size*(i+1) if batch_size*(i+1)<len(y) else len(y)
        val_x = X[val_start: val_end]
        train_x = np.delete(X, range(val_start, val_end), axis=0)
        val_y = y[val_start: val_end]
        train_y = np.delete(y,range(val_start, val_end), axis=0)
        model.fit(train_x, train_y)

        model_train_prediction =  model.predict(X=train_x)
        train_error = compute_error(model_train_prediction, train_y)
        average_train_error += train_error

        model_val_prediction =  model.predict(X=val_x)
        val_error = compute_error(model_val_prediction, val_y)
        average_val_error += val_error

    average_train_error /= folds
    average_val_error /= folds

    return(average_train_error, average_val_error)


