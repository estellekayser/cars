import mysql.connector
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn import svm

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
#  coeff_pearson : 0.88 --> Correlation

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

## Sklearn

# Converting the X and Y variables to numpy arrays (matrices/vectors) and also add a column of 1’s to X (as the value of X0 is 1 in w0. 
# x0 i.e. coefficient of intercept is assumed to be 1)
xs = np.column_stack([np.ones(len(df)),df['Age'].values])
ys = df['Selling_Price'].values
reg = LinearRegression().fit(xs, ys)
print(xs)
print(ys)
print('Sklearn - Coefficients:', reg.coef_)
print('Sklearn - Intercept: \n', reg.intercept_)

# ##### Regression lineaire multiple
data = pd.DataFrame(df, columns=['Age','Kms_Driven', 'Selling_Price'])

col_transform = pd.DataFrame(df, columns=['Transmission'])
dummies = pd.get_dummies(col_transform)

xm = np.array(data.join(dummies))
ym = df['Selling_Price'].values

reg = LinearRegression().fit(xm, ym)
print('Sklearn multiple - Coefficients:', reg.coef_)
print('Sklearn multiple - Intercept: \n', reg.intercept_)

### Creer son propre algo de regression lineaire multiple
class myLinearRegression():

    def cal_coef(X,y):
        """Function to calculate coefficients or w or theta or b0 & b1"""
        coeffs = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        return coeffs
  
    def predict_normeq(self, new_x,coeffs):
        """Function to predict linear regression using normal equation"""
        new_y = np.dot(new_x, coeffs)
        return new_y
  
    x_test = 6.1101

    coeffs = cal_coef(X,y) # b0-intercept, b1-slope
    print(f'Intercept: {coeffs[0]} Slope:{coeffs[1]}')

    #add ones to x matrix
    if np.isscalar(x_test):
        new_x = np.array([1, x_test])
    else:
        new_x = np.column_stack([np.ones(len(x_test), dtype=np.float32),x_test])
            
    normeq_preds = predict_normeq(X,coeffs)
    print(f'Profit Prediction for {x_test*1000} is {predict_normeq(new_x,coeffs)*10000}')

    ## Model Evaluation
    SSE = sum((y-normeq_preds)**2) # Sum of squared error
    SST = sum((y-np.mean(y))**2) # Sum of squared total
    n=len(X) # Number of obeservations
    q=len(coeffs) # Number of coefficients
    k=len(coeffs) # Number of parameters
    MSE = SSE/(n-q) # Mean Squared Error
    MST = SST/(n-1) # Mean Squared Total

    R_squared = 1-(SSE/SST) # R Square
    Adj_Rsquared = 1-(MSE/MST) # Adjusted R square
    std_err = np.sqrt(MSE) # Standard Error or Root mean squared error
    MSR = (SST-SSE)/(q-1) # Mean Squared Regression
    f_static = MSR/MSE # F Statics
    MAPE = sum(np.abs(y-normeq_preds))/n # Mean Absolute Percentage Error

    print(f'R Squared: {R_squared}\n\nAdj. R-Squared: {Adj_Rsquared}\n\nStd.Error: {std_err}\n\nF Static: {f_static}\n')
    print(f'MeanAbsPercErr. : {MAPE}')







# ##### Support Vector Machines (SVM)

from sklearn.metrics import mean_squared_error
z = df['Age']
y = df['Selling_Price']

x = np.array(z).reshape(-1,1)

svr = svm.SVR(kernel="linear").fit(x,y)
print('SVR - Coefficients:', svr.coef_)
print('SVR - Intercept:', svr.intercept_)



yfit = svr.predict(x)
plt.scatter(x, y, s=5, color="blue", label="original")
plt.plot(x, yfit, lw=2, color="red", label="fitted")
plt.legend()
plt.show()

score = svr.score(x,y)
print("R-squared:", score)
print("MSE:", mean_squared_error(y, yfit))