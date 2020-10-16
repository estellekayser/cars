import pandas as pd
import matplotlib as plt

carsData = pd.read_csv("data/carData.csv")

print("(lignes, colonnes) :", carsData.shape)

print(carsData.describe(include='all'))

vquanti =("Year","Selling_Price","Present_Price","Kms_Driven")

for v in vquanti:
    plt.hist(v, range = (0, 5), bins = 5, color = 'yellow',
                edgecolor = 'red')
    plt.xlabel('valeurs')
    plt.ylabel('nombres')
    plt.title('Exemple d\' histogramme simple')



