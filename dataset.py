import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def main():
    
    # On charge les données et on affiche les statistiques descriptives avant et après le nettoyage
    def get_names(range_=30):
        names = ['Diagnosis']
        for i in range(range_):
            names.append(f'Feature{i+1}')
        return names

    df = pd.read_csv("data.csv", header=None, names=['ID'] + get_names())
    df.drop(df.columns[0], axis=1, inplace=True)

    scaler = StandardScaler()
    for feature in df.columns[1:]:
        df[feature] = scaler.fit_transform(df[[feature]])

    def plot_swarm(value_vars):
        df_melted = df.melt(id_vars=[df.columns[0]], value_vars=value_vars, var_name='Features', value_name='Value')
        print(df_melted.describe())
        print(df_melted.head())
        sns.stripplot(data=df_melted, x='Features', y='Value', hue=df.columns[0], dodge=False, jitter=True) # Rapide
        # sns.swarmplot(data=df_melted, x='Features', y='Value', hue=df.columns[0], dodge=False) # Qualité supérieure mais plus lent
        plt.xticks(rotation=40)

    # On divise par moitié les features
    mid_point = (len(df.columns) - 1) // 2

    plt.figure(figsize=(20, 12))
    plt.subplot(211)
    plot_swarm(df.columns[1:mid_point+1])
    plt.subplot(212)
    plot_swarm(df.columns[mid_point+1:])
    plt.show()

if __name__ == "__main__":
    main()
    # Doc: https://pmc.ncbi.nlm.nih.gov/articles/PMC9913345/#sec3-cancers-15-00681