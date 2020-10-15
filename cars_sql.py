import mysql.connector
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import numpy as np
import scipy as sp
from sklearn.linear_model import LinearRegression

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

##### Creation variable age
df['Age'] = df.apply(lambda row: 2020-row.Year, axis=1)

##### Analyse univariee
df['Year'] = df['Year'].astype(str)
df['Owner'] = df['Owner'].astype(str)
print(df.info(),"\n")

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
print("Present_Price, Selling_Price - Coefficient de Pearson = {}".format(coeff_pearson))
#  coeff_pearson : 0.88 --> Correlation positive

coeff_pearson,_ = pearsonr(df['Kms_Driven'],df['Selling_Price'])
print("Kms_Driven, Selling_Price - Coefficient de Pearson = {}".format(coeff_pearson))
# coeff_pearson : 0.03 --> Pas de correlation 

coeff_pearson,_ = pearsonr(df['Kms_Driven'],df['Present_Price'])
print("Kms_Driven, Present_Price - Coefficient de Pearson = {} \n".format(coeff_pearson))
# coeff_pearson : 0.20 --> Faible correlation 

##### Regressions linaires

x = df['Age']
y = df['Selling_Price']

## Numpy
a,b = np.polyfit(x,y,1)
print("Numpy - a: %f    b: %f \n" % (a, b))

## Scipy
slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x, y)
print("Scipy - slope: %f  intercept: %f" % (slope, intercept))
print("R-squared: %f \n" % r_value**2)
plt.plot(x, y, 'o')
plt.plot(x, intercept + slope*x, 'r', label=("y = %fx+%f" % (slope, intercept)))
plt.title('Univariate regression with Scipy')
plt.xlabel('Age')
plt.ylabel('Selling_Price')
plt.legend()
plt.show()

## Sklearn

# Converting the X and Y variables to numpy arrays (matrices/vectors) and also add a column of 1’s to X (as the value of X0 is 1 in w0. 
# x0 i.e. coefficient of intercept is assumed to be 1)
xs = np.column_stack([np.ones(len(df)),df['Age'].values])
ys = df['Selling_Price'].values
reg = LinearRegression().fit(xs, ys)
print('Sklearn - Coefficients:', reg.coef_[1])
print('Sklearn - Intercept:', reg.intercept_)
plt.plot(xs, ys, 'o')
plt.plot(xs, reg.intercept_ + reg.coef_[1]*xs, 'r',  label=("y = %fx+%f" % (reg.coef_[1], reg.intercept_)))
plt.title('Univariate regression with Sklearn')
plt.xlabel('Age')
plt.ylabel('Selling_Price')
plt.legend()
plt.show()


