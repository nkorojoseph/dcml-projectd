import pandas
import sklearn.metrics
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pickle

from LoadInjector import current_ms
model: None = None
best_model= ''
temp = 0
accuracy = 0

if __name__ == "__main__":
    """
    Entry point for the Supervised ML Examples
    """

    # Reading a CSV file into a DataFrame pandas object
    my_df = pandas.read_csv("input_folder/monitored_data_labeled.csv", sep=',')
    

    # splitting the dataframe in features (x) and label (y)
    y = my_df["label"]
    x = my_df.drop(columns=["_timestamp", "label"])

    # partitioning the dataset into a train and a test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, shuffle=False)
    # print(x_train)
    #
    # print(y_train)

    # Set of classifiers that I want to run and compare
    classifiers = [VotingClassifier(estimators=[('lda', LinearDiscriminantAnalysis()),
                                                ('nb', GaussianNB()),
                                                ('dt', tree.DecisionTreeClassifier())]),
                   StackingClassifier(estimators=[('lda', LinearDiscriminantAnalysis()),
                                                ('nb', GaussianNB()),
                                                ('dt', tree.DecisionTreeClassifier())],
                                      final_estimator=RandomForestClassifier(n_estimators=10)),
                   tree.DecisionTreeClassifier(), GaussianNB(),
                   LinearDiscriminantAnalysis()
                   ]

    for clf_name,clf in enumerate(classifiers):
        # Training an algorithm
        before_train = current_ms()
        clf.fit(x_train, y_train)
        after_train = current_ms()

        # Testing the trained model
        predicted_labels = clf.predict(x_test)
        end = current_ms()

        # COmputing metrics to understand how good an algorithm is
        accuracy = sklearn.metrics.accuracy_score(y_test, predicted_labels)
        print("Accuracy is %.4f, train time: %d, test time: %d" % (accuracy, after_train-before_train, end-after_train))

        if accuracy > temp:
            temp = accuracy
            best_accuracy = temp
            model = clf
            best_model = clf_name


    print('best accuracy ',best_accuracy , ' best model ', best_model)

    #Replace 'model' with the captured trained model variable
    with open('trained_anomaly_model.pkl', 'wb') as f:
        pickle.dump(model, f)