import pandas
import sklearn.metrics
from pyod.models.abod import ABOD
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from sklearn.model_selection import train_test_split

from LoadInjector import current_ms

if __name__ == "__main__":
    """
    Entry point for the Supervised ML Examples
    """

    # Reading a CSV file into a DataFrame pandas object
    my_df = pandas.read_csv("input_folder/dataset_arancino_monitor.csv", sep=',')
    

    # splitting the dataframe in features (x) and label (y)
    y = my_df["label_bin"]
    x = my_df.drop(columns=["_timestamp", "label", "label_bin"])

    # partitioning the dataset into a train and a test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, shuffle=False)

    # COmputing metrics to understand how good an algorithm is
    all_0 = [0 for i in range(0, len(y_test))]
    accuracy = sklearn.metrics.accuracy_score(y_test, all_0)
    print("Accuracy with always 0 is %.4f" % (accuracy))

    # Set of classifiers that I want to run and compare
    classifiers = [HBOS(contamination=0.5, n_bins=20), ABOD(contamination=0.5), COPOD(contamination=0.5)]

    for clf in classifiers:
        # Training an algorithm
        before_train = current_ms()
        clf = clf.fit(x_train)
        after_train = current_ms()

        # Testing the trained model
        predicted_labels = clf.predict(x_test)
        end = current_ms()

        # COmputing metrics to understand how good an algorithm is
        accuracy = sklearn.metrics.accuracy_score(y_test, predicted_labels)
        print("Accuracy is %.4f, train time: %d, test time: %d" % (accuracy, after_train-before_train, end-after_train))

