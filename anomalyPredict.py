import pickle
import numpy as np

if __name__== "__main__":


    #data to predict
    new_data = np.array([
        [
            19502.90625, 12956.85938, 2672142.844, 949.078125, 814.84375, 1072014919, 570929084, 0, 4255844374, 1.09,
            0.31, 0.11, 28991029248, 10489864192, 18501165056, 36.2, 0, 0, 16832106496, 3589865472, 78.7, 13242241024,
            3589865472, 1020000000000.00, 664000000000.00, 359000000000.00, 64.9, 11417373, 3184623, 180000000000.00,
            135000000000.00, 11238, 1612, 25833493, 106151737, 135917, 130192, 0, 0, 0, 0
        ]
    ])
    # Load the trained model
    # supervised learning
    with open('trained_anomaly_model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Make predictions
    predictions = model.predict(new_data)
    print("supervised: ", predictions)

    #unsupervised learning
    with open('trained_unsuper_anomaly_model.pkl','rb') as modelfile:
        modelx = pickle.load(modelfile)
        # Make predictions
        predictions = modelx.predict(new_data)
    print("unsupervised: ",predictions)