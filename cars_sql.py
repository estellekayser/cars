import mysql.connector

cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='cars')

mycursor = cnx.cursor()

mycursor.execute("SELECT * FROM `table 1`")

myresult = mycursor.fetchall()

for x in myresult:
  print(x)