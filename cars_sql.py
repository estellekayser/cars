import mysql.connector
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

##### Connexion a MySQL
cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='cars')

mycursor = cnx.cursor()

##### Import table "cars" de MySQL
mycursor.execute(
    "SELECT Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner \
    FROM cars")
request = mycursor.fetchall()

##### Creation dataframe
df = pd.DataFrame([[ij for ij in i] for i in request])
df.rename(columns={0:'Car_Name', 1:'Year', 2:'Selling_Price', 3:'Present_Price', 4:'Kms_Driven', 5:'Fuel_Type', 6:'Seller_Type', 7:'Transmission', 8:'Owner' }, inplace=True)
print(df.info(),"\n")
df['Year'] = df['Year'].astype(str)
df['Owner'] = df['Owner'].astype(str)
print(df.info(),"\n")

##### Analyse univariee
print(df.describe(include='all'))

vquali = ('Car_Name','Year','Fuel_Type', 'Seller_Type', 'Transmission', 'Owner')
for var in vquali:
    freq = pd.crosstab(df[var], "frequence")
    rep = pd.crosstab(df[var], "repartition", normalize=True)
    print(pd.concat([freq, rep], axis=1), "\n")

vquanti = ("Selling_Price", "Present_Price", "Kms_Driven")
for var in vquanti:
    sns.catplot(x="Year", y=var, kind="box", data=df)
    plt.title(var)
    plt.show()

##### Analyse bivariee
pd.plotting.scatter_matrix(df)
plt.show()
# On suppose une correlation entre Present_Price et Selling_Price

coeff_pearson,_ = pearsonr(df['Present_Price'],df['Selling_Price'])
print("coefficient de Pearson = {}".format(coeff_pearson))
#  coeff_pearson : 0.88 --> Correlation positive

coeff_pearson,_ = pearsonr(df['Kms_Driven'],df['Selling_Price'])
print("coefficient de Pearson = {}".format(coeff_pearson))
# coeff_pearson : 0.03 --> Pas de correlation 

coeff_pearson,_ = pearsonr(df['Kms_Driven'],df['Present_Price'])
print("coefficient de Pearson = {}".format(coeff_pearson))
# coeff_pearson : 0.20 --> Faible correlation 
