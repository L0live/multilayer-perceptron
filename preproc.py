from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def main():

    df = read_csv("data.csv", header=None, names=['ID', 'Diagnosis'] + [f'Feature{i+1}' for i in range(30)])
    df.drop('ID', axis=1, inplace=True)

    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Diagnosis'])

    scaler = StandardScaler()
    feature_cols = df.columns[1:]
    train.loc[:, feature_cols] = scaler.fit_transform(train[feature_cols])
    test.loc[:, feature_cols] = scaler.transform(test[feature_cols])

    train.to_csv("data_train.csv", index=False)
    test.to_csv("data_valid.csv", index=False)

    print("Shape: ", df[df.columns[1:]].shape)

if __name__ == "__main__":
    main()