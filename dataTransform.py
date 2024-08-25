import pandas as pd

if __name__ == "__main__":
    # Reading a CSV file into a DataFrame pandas object
    my_df = pd.read_csv("input_folder/monitored_data_labeled.csv", sep=',')
    df = pd.DataFrame(my_df)
    print(my_df.shape)


    # Remove the apostrophes and convert to numeric
    # df['disk_io.read_bytes'] = df['disk_io.read_bytes'].str.replace("'", "").astype(float)
    # df['disk_io.write_bytes'] = df['disk_io.write_bytes'].str.replace("'", "").astype(float)
    # print(df)