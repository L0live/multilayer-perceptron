import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def main():
    
    # On charge les données et on affiche les statistiques descriptives avant et après le nettoyage
    df = pd.read_csv("data.csv", header=None)
    df.drop(df.columns[0], axis=1, inplace=True)
    print(df.describe())

    # On transforme le dataframe pour avoir une colonne "Features" et une colonne "Value"
    df_melted = df.melt(id_vars=[df.columns[0]], value_vars=df.columns[1:], var_name='Features', value_name='Value')
    print(df_melted.describe())

    # Normalisation des données
    scaler = StandardScaler()
    df_melted['Value'] = scaler.fit_transform(df_melted[['Value']])
    print(df_melted.describe())

    # On divise par moitié les features
    mid_point = (len(df.columns) - 1) // 2
    df1 = df_melted[df_melted['Features'].isin(df.columns[1:mid_point+1])]
    df2 = df_melted[df_melted['Features'].isin(df.columns[mid_point+1:])]

    def plot_swarm(df_):
        plt.figure(figsize=(20, 10))
        sns.stripplot(data=df_, x='Features', y='Value', hue=df.columns[0], dodge=False, jitter=True) # Rapide
        # sns.swarmplot(data=df_, x='Features', y='Value', hue=df.columns[0], dodge=False) # Qualité supérieure mais plus lent
        plt.show()

    plot_swarm(df1)
    plot_swarm(df2)

if __name__ == "__main__":
    main()
    # Doc: https://pmc.ncbi.nlm.nih.gov/articles/PMC9913345/#sec3-cancers-15-00681