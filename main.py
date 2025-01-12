import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt 
df = pd.read_csv(r"D:\data science\excel files ds\Car.csv").iloc[:,1:]
df

# missing values 
df.isna().sum()
df = df.dropna()
print("length of dataset", len(df))
df = df.reset_index(drop = True)

df = df.drop(columns = ["torque"])
df

# max power
def fxn(x) :
    return x.split()[0]

df["max_power"] = df["max_power"].apply(fxn)
df 
df.dtypes

# convert int into float
#df["max_power"] = df["max_power"].astype("float64")
df

non_convertable_indexes = []
for i in range(len(df)) :
    try :
        float(df.iloc[i,-2])
    except :
        non_convertable_indexes.append(i)
print(non_convertable_indexes)

# now drop the indexes

# now recdf["max_power"] = df["max_power"].astype("float64")
df
# convert the column into float
df = df.drop(index = non_convertable_indexes)
df = df.reset_index(drop = True)
df

def fxn(x) :
    return x.split()[0]

df["engine"] = df["engine"].apply(fxn)
df
non_convertable_indexes = []
for i in range(len(df)) :
    try :
        float(df.iloc[i,-3])
    except :
        non_convertable_indexes.append(i)
print(non_convertable_indexes)
df = df.drop(index = non_convertable_indexes)
df = df.reset_index(drop = True)
df["engine"] = df["engine"].astype("float64")
df

# drawing plot of above column
plt.hist(df["engine"])
plt.show()

# we need to visualise the column so tht we can check distribution, how many are outliers, we can catogrise the column

# mileage 

def fxn(x) :
    return x.split()[0]

df["mileage"] = df["mileage"].apply(fxn)
df


df["mileage"] = df["mileage"].astype("float64")
df

df["owner"].value_counts()

# fifth category should be merge with four and other category
df["owner"] = df["owner"].replace({"Fifth":"Fourth & Above Owner"})
df["owner"].value_counts()

import seaborn 
seaborn.violinplot(data=df,x="selling_price",y="owner")
# below violin plot is showing the distribution of selling price in each owner category
# as we can see the distribution of sellin price becme compact as we increase the number of owner 
# drop the test drive car as it is destoying the relation of column with independent 
f = df["owner"]=="Test Drive Car"
df = df[~f] # return only non test drive car 

df["owner"].value_counts()

# transmissions 
df["transmission"].value_counts()

df["seller_type"].value_counts()
import seaborn 
seaborn.violinplot(data=df,x="selling_price",y="seller_type")

df["fuel"].value_counts()
import seaborn 
seaborn.violinplot(data=df,x="selling_price",y="fuel")
df["fuel"] = df["fuel"].replace({"LPG":"ECO","CNG":"ECO"})  # ECO new variable create 
df
# lpg and cng distribution normally same i.e weather car is lpg or cng it is not impacting its selling price 

# km driven 
plt.scatter(df["km_driven"],df["selling_price"])
plt.xlabel("km_driven")
plt.ylabel("selling_price")
plt.show()

pd.set_option('display.float_format', lambda x: '%.3f' % x)  # to stop scientific noatation, flaot show upto 3 decimal

df["name"].value_counts()
# acc. to this data there are approx 1980 car 
"""
when there are two much dummy variable in hot encoding that increase the sparcity in data  
therefore cannot do onehot encoding then we see that column with respect to selling price and do the remaning column constant
in reality brand is impacting the selling price of car
"""
df["name"] = df["name"].apply(lambda x:x.split()[0])
df   

# how to simplify this column 
# we can see that there are group of brand like expansive car, family car, outdated car 
# we have to group these category according to average brand value selling price
# find the average selling price of each brand 
groups = df.groupby("name")
means = groups["selling_price"].mean()
means
# sort the selling price
means = means.sort_values(ascending = False)
means

# avoid loops in numpy and pandas 
def fxn(brand):
    if brand in means.index[:10]:
        return 2 
    elif brand in means.index[10:25] :
        return 1 
    else :
        return 0 
df["name"] = df["name"].apply(fxn)
df

non_conertable_indexes = []
for i in range(len(df)):
    try :
        float(df.iloc[i,-3])
    except:
        non_conertable_indexes.append(i)

print(non_conertable_indexes)

df = df.drop(index = non_conertable_indexes)
df
df = df.reset_index(drop = True)
df["engine"] = df["engine"].astype("float64")
df

df["engine"] = df["engine"].astype("float64")
df

# stastistical analysis
# Make two different variable of categorical and numeric columns 
numeric = df[["year","selling_price","km_driven","engine","max_power"]]
category = df[["name","fuel","seller_type","transmission","owner","seats"]]

#convert categorical column into number temporariy 
category["owner"] = category["owner"].replace({"First Owner": 0,"Second Owner":1,"Second Owner":2,"Third Owner":3,"Fourth & Above Owner":4})
category["fuel"] = category["fuel"].replace({"Diesel": 0,"Petrol":1,"ECO":2})
category["seller_type"] = category["seller_type"].replace({"Individual": 0,"Dealer":1,"Trustmark Dealer":2})
category["transmission"] = category["transmission"].replace({"Manual": 0,"Automatic":1})

df

plt.hist(df["mileage"],density = True)        # histogram tell frequency
# density = true now it is telling probability
plt.show()

plt.violinplot(df["mileage"]) 
df

numeric["km_driven"].skew()
numeric["km_driven"].kurt()
df["mileage"].skew()
df["mileage"].kurt()

corr = numeric.corr()     # correlation matrix
corr

import seaborn
seaborn.heatmap(corr)
# lighter colour i.e positive correlation among column
# darker the colour more negative correlation

from sklearn.feature_selection import f_classif, SelectKBest
# f test for anova 
# SelectKBest is a class made to find the topmost important column in data withrespect to dependent 
# we have to pass the value of k which is number of column we want , it wll return those number of top important column according to f value
s = SelectKBest(f_classif,k = 6)
columns = s.fit_transform(category,numeric["selling_price"])

# in data first should be dataframe of category column and second should be dependent column 
# in column there would be those column which is returned by SelectKbest according to f value

print(s.scores_)


data = numeric[["engine","selling_price"]]   #, engine distribution not normal
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps = 0.06,min_samples = 10 )
dbscan.fit(data)
dbscan.labels_  # it will return the label for each lablels, labels indicate min neighbours of each points, if any point is noisy it will be represented by -1


from sklearn.preprocessing import MinMaxScaler 
mn = MinMaxScaler()
data = mn.fit_transform(data)
data

# nearest neighbour distance of each point 
from sklearn.neighbors import NearestNeighbors 
NN = NearestNeighbors(n_neighbors = 10)  # NearestNeighbors algo wil calculate the distance and index of that point which is nearest to each point 
NN.fit(data)
# we have to pass how much neighbour we want 
# find the distance of top 10 points and their indexes with respect to each point, it will give two array on distance and indexes

distance,index = NN.kneighbors(data)
print(distance)

# extract our distance of nearest neighbour 
distance = distance[:,1]   # 0 not taken because it is itself distance 
distance = np.sort(distance,axis = 0)   # ascending order 

plt.plot(distance)
plt.title("k-distance graph")
plt.show()

# choose vaalue which have more curvature and we select epsilon value 
# min neighbours depend upon field knowledge 

# task give different colour to core,border,noise points
plt.scatter(data[:,0],data[:,1])
plt.show()
len(data)

numeric.columns
df

y=numeric["selling_price"]
numeric = numeric.drop(columns=["selling_price"])
X = pd.concat((numeric,category),axis=1)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("encode",OneHotEncoder(),[6,7])],remainder="passthrough")

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))