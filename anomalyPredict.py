import pickle
import numpy as np

if __name__== "__main__":
    # Load the trained model
    with open('trained_anomaly_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Prepare new data
    new_data = np.array([
        [1723899333,15952.4375,10530.5625,2434991.734,801.40625,743.765625,872135711,
        497431793,0,3560416178,0.48,0.27,0.12,28991029248,11298611200,17692418048,39,
        0,0,16832106496,2326458368,86.2,14505648128,2326458368,1020000000000.00,664000000000.00,
        359000000000.00,64.9,10772594,2874953,170000000000.00,125000000000.00,10949,1500,
        31941672,550264437,232004,435151,0,0,0,0,]])  # Replace with your actual data

    # Make predictions
    predictions = model.predict(new_data)

    print(predictions)