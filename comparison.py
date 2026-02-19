"""
For my comparison I will use a k-nearest neighbors classifier. It will use the same training and test datasets,
but the principle is slightly difference. A KNN classifier will look at the k (parameter) closest trianing samples
to the test sample and will predict which class the test sample belongs to based on the classes of those nearest
neighbors. Basically rather than deciding a binary label based on a linear decision (positive or negative side of
a sigmoid function), it will decide based on the classification of the k nearest comperable samples.

I chose KNN because I used a KNN measrument in my COMPS as part of my evaluation metrics to give me a classification
accuracy for real vs synthetic data. At the time I just used a stock model without fully understanding why I was using
it, since it wasn't a part of my actual experiment/investigation, and was just an evaluation metric. I wanted to take
this chance to actually get to understand the model better.
"""

from sklearn.neighbors import KNeighborsClassifier
from binary_classification import load_data


def run_knn(n_neighbors=5):
    # Load data (returns torch tensors)
    X_train, X_test, y_train, y_test, feature_names = load_data()

    # Convert to numpy arrays for scikit-learn
    X_train_np = X_train.numpy()
    X_test_np = X_test.numpy()
    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()

    # Initialize and train KNN
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_np, y_train_np)

    # Predictions
    train_pred = model.predict(X_train_np)
    test_pred = model.predict(X_test_np)

    # Compute accuracies
    train_acc = (train_pred == y_train_np).mean()
    test_acc = (test_pred == y_test_np).mean()

    return train_acc, test_acc, len(y_train_np), len(y_test_np), X_train_np.shape[1]


if __name__ == "__main__":
    print("Loading data...")
    train_acc, test_acc, n_train, n_test, n_features = run_knn()
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")
    print(f"Features: {n_features}")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

'''
KNN: 
- Training accuracy: 0.9802
- Test accuracy: 0.9474

Binary Classifier:
- Training accuracy 0.9868
- Test accuracy: 0.9912

For my tests the binary classifier actually performed better than the KNN classifier. I was definitely
surprised that my implemented-from-scratch model performed better than the scikit-learn model. However,
with such a small batch of training data, my guess is that most 'stock' models (whether implemented from
scratch or not) will perform very similarly. Additionally with the type of data that is being observed, 
KNN is probably not the best model to use. It relies heavily on the difference between samples, which with
relatively few features and a small dataset, there won't be that many distinct samples to compare to. 

'''