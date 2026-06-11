from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def main():

    df = read_csv("data.csv", header=None, names=['ID', 'Diagnosis'] + [f'Feature{i+1}' for i in range(30)])
    df.drop('ID', axis=1, inplace=True)

    scaler = StandardScaler()
    df[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])

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

    # relevantFeaturesNb = ['1', '3', '4', '7', '8', '11', '13', '14', '21', '23', '24', '28']
    

if __name__ == "__main__":
    main()
    # Doc: https://pmc.ncbi.nlm.nih.gov/articles/PMC9913345/#sec3-cancers-15-00681