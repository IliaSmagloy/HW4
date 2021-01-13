#3.1
import numpy as np

np.random.seed(42)

from sklearn.datasets import fetch_openml


def fetch_mnist():
    # Download MNIST dataset
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    # Randomly sample 8000 images
    np.random.seed(2)
    indices = np.random.choice(len(X), 8000, replace=False)
    X, y = X[indices], y[indices]
    return X, y


if __name__ == "__main__":
    print("Starting")
    X, y = fetch_mnist()
    print(X.shape, y.shape)

    # From here on my own code (Ilia)
    #3.2

    import matplotlib.pyplot as plt
    #
    # fig = plt.figure(figsize=(3,4))
    # rows, columns = 3,4
    #
    # for index, image in enumerate(X[0:10]):
    #     correct_shaped_image = np.reshape(image, (28,28))
    #     image_figure = fig.add_subplot(rows,columns,index+1)
    #     plt.imshow(correct_shaped_image, cmap = "binary")
    #     image_figure.title.set_text(str(index+1))
    #     plt.axis('off')
    #
    #
    # plt.show()

    #3.4

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from SVM_results import  SVM_results
    result_dictionary = SVM_results(X_train,y_train,X_test,y_test)

    print(''.join(['{0}: {1}\n'.format(k, v) for k, v in result_dictionary.items()]))

    labels_positions= np.arange(len(result_dictionary.keys()))

    errors = [list(l) for l in zip(*result_dictionary.keys())]

    labels = [ "Average Train Erorrs", "Average Validation Errors", "Test Errors"]
    colors = ['r', 'g','b']
    bar_width = 0.25

    for i in range(3):
        plt.bar(labels_positions+(i-1)*bar_width, errors[i], bar_width, color=colors[i], label=labels[i])

    plt.xticks(labels_positions, [label[4:] for label in result_dictionary.keys()])
    plt.xlabel("Different Models")
    plt.ylabel("Error rate")
    plt.title("Error rates for each Model")
    plt.show()