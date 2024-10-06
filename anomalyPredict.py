import pickle
import numpy as np
import  pandas as pd

if __name__ == "__main__":

    my_df = pd.read_csv("output_folder/realdata_without_fault.csv")
    newData = my_df.drop(columns=["_timestamp"])

    testdata = np.array(newData)
    with open('trained_unsuper_anomaly_model.pkl', 'rb') as modelfile:
        modelx = pickle.load(modelfile)
        for data in testdata:
            tdata = np.array([data])
            # Make predictions
            predictions = modelx.predict(tdata)
            if predictions[0] == 1:
                print("anomaly")
            else:
                print("normal")
