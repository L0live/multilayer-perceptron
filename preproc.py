from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def main():
    
    # On charge les données et on affiche les statistiques descriptives avant et après le nettoyage
    def get_names(range_=30):
        names = ['Diagnosis']
        for i in range(range_):
            names.append(f'Feature{i+1}')
        return names

    df = read_csv("data.csv", header=None, names=['ID'] + get_names())
    df.drop(df.columns[0], axis=1, inplace=True)

    scaler = StandardScaler()
    for feature in df.columns[1:]:
        df[feature] = scaler.fit_transform(df[[feature]])

    diagnosis = df[df.columns[0]]
    print("Diagnosis: \n", diagnosis.head(10))
    features = df[df.columns[1:]]
    print("Features: \n", features.head(10))
    
    features_train, features_test, diagnosis_train, diagnosis_test, train_test_split(features, diagnosis, test_size=0.2, random_state=42, stratify=diagnosis)

if __name__ == "__main__":
    main()