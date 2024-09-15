import pandas
import sklearn.metrics
from pyod.models.abod import ABOD
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from sklearn.model_selection import train_test_split
from pyod.utils.data import evaluate_print

from LoadInjector import current_ms

if __name__ == "__main__":
    """
    Entry point for the Supervised ML Examples
    """

    # Reading a CSV file into a DataFrame pandas object
    my_df = pandas.read_csv("input_folder/monitored_data_labeled.csv", sep=',')
    

    # splitting the dataframe in features (x) and label (y)
    y = my_df["label"]
    x = my_df.drop(columns=["_timestamp","label"])
    norm_map = {"normal": 1, "anomaly": 0}
    y_ = y.map(norm_map)

    # partitioning the dataset into a train and a test set
    x_train, x_test, y_train, y_test = train_test_split(x, y_, test_size = 0.15, shuffle=False)

    # COmputing metrics to understand how good an algorithm is
    # all_0 = [0 for i in range(0, len(y_test))]
    # accuracy = sklearn.metrics.accuracy_score(y_test, all_0)
    # print("Accuracy with always 0 is %.4f" % (accuracy))

    # # Set of classifiers that I want to run and compare
    classifiers = [HBOS(contamination=0.15, n_bins=20), ABOD(contamination=0.15), COPOD(contamination=0.15)]
    # HBOS ABOD COPOD
    for clf in classifiers:
        # Training an algorithm
        before_train = current_ms()
        clf = clf.fit(x_train)
        after_train = current_ms()

        ##
        y_train_scores  = clf.decision_scores_

        y_test_pred  = clf.predict(x_test)
        y_test_scores = clf.decision_function(x_test)

        # it is possible to get the prediction confidence as well
        y_test_pred, y_test_pred_confidence = clf.predict(x_test, return_confidence=True)  # outlier labels (0 or 1) and confidence in the range of [0,1]

        # evaluate and print the results
        print("\nOn Training Data:")
        evaluate_print("Model", y_train, y_train_scores)
        # print("\nOn Test Data:")
        # evaluate_print("Model", y_test, y_test_scores)

