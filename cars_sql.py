import mysql.connector
import seaborn as sns
import pandas as pd

# Connexion a MySQL
cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='cars')

mycursor = cnx.cursor()

# Import table "cars" de MySQL
mycursor.execute(
    "SELECT Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner \
    FROM cars")
request = mycursor.fetchall()

# Creation de la db
df = pd.DataFrame([[ij for ij in i] for i in request])
df.rename(columns={0:'Car_Name', 1:'Year', 2:'Selling_Price', 3:'Present_Price', 4:'Kms_Driven', 5:'Fuel_Type', 6:'Seller_Type', 7:'Transmission', 8:'Owner' }, inplace=True)

print(df.head)