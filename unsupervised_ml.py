import pandas
import sklearn.metrics
from pyod.models.abod import ABOD
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.knn import  KNN
from sklearn.model_selection import train_test_split
from pyod.utils.data import evaluate_print
from pyod.utils.data import get_outliers_inliers
from pyod.utils.utility import precision_n_scores
import pickle
from LoadInjector import current_ms
from pyod.utils.utility import standardizer
import numpy as np

if __name__ == "__main__":

    model: None = None
    best_model = ''
    temp = 0
    accuracy = 0
    out_lier_temp = 0
    outlier = 0

    # Reading a CSV file into a DataFrame pandas object
    my_df = pandas.read_csv("input_folder/monitored_data_labeled.csv", sep=',')

    # splitting the dataframe in features (x) and label (y)
    y = my_df["label"]
    x = my_df.drop(columns=["_timestamp","label"])
    norm_map = {"normal": 1, "anomaly": 0}
    y_ = y.map(norm_map)

    # partitioning the dataset into a train and a test set
    x_train, x_test, y_train, y_test = train_test_split(x, y_, test_size = 0.15, shuffle=True)

    # # Set of classifiers that I want to run and compare
    classifiers = [HBOS(contamination=0.15, n_bins=20), ABOD(contamination=0.15), KNN(contamination=0.15), KNN(method='mean',contamination=0.15)]
    # HBOS ABOD COPOD
    for clf in classifiers:
        # Training an algorithm
        before_train = current_ms()
        clf = clf.fit(x_train)
        after_train = current_ms()
        ##

        train_scores = clf.decision_function(x_train)
        test_scores = clf.decision_function(x_test)


        y_train_pred = clf.predict(x_train)

        n_inliers = len(y_train_pred) - np.count_nonzero(y_train_pred)
        n_outliers = np.count_nonzero(y_train_pred == 1)

        #print(n_outliers, n_inliers)

        precision = precision_n_scores(y_train,clf.labels_)

        # it is possible to get the prediction confidence as well
        y_test_pred, y_test_pred_confidence = clf.predict(x_test, return_confidence=True )  # outlier labels (0 or 1) and confidence in the range of [0,1]

        n_inliers_test = len(y_test_pred) - np.count_nonzero(y_test_pred)
        n_outliers_test = np.count_nonzero(y_test_pred == 1)

        precision_on_test = precision_n_scores(y_test,test_scores)

        print("precision_on_test:", precision_on_test, "n_inliers_test:", n_inliers_test, "n_outliers_test:",
              n_outliers_test)

        if precision_on_test > temp and n_outliers_test > out_lier_temp :
            temp = precision_on_test
            out_lier_temp = outlier
            outlier = n_outliers_test
            best_accuracy = precision_on_test
            model = clf

    print ("best model is: ", clf, "tmp: ", temp, "outlier: ", outlier)

    with open('trained_unsuper_anomaly_model.pkl', 'wb') as f:
        pickle.dump(model, f)