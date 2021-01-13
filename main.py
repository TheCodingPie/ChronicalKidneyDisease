
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from  sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.metrics import accuracy_score
from  sklearn.metrics import plot_roc_curve
from sklearn.metrics import  *
from  sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from  sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from  sklearn.decomposition import PCA
from sklearn.metrics import  ConfusionMatrixDisplay

from sklearn import preprocessing

data=pd.read_csv("ckd.csv")
data.columns=['id','age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad','appet','pe','ane','class']
data=data.drop(columns=['id'])
print(data.head(5))

data=data.replace(to_replace='?',value=np.nan)
print(data.head(10))

print(data.shape)

data['age']=pd.to_numeric(data['age'],errors='coerce')
data['bp']=pd.to_numeric(data['bp'],errors='coerce')
data['bgr']=pd.to_numeric(data['bgr'],errors='coerce')
data['bu']=pd.to_numeric(data['bu'],errors='coerce')
data['sc']=pd.to_numeric(data['sc'],errors='coerce')
data['sod']=pd.to_numeric(data['sod'],errors='coerce')
data['pot']=pd.to_numeric(data['pot'],errors='coerce')
data['hemo']=pd.to_numeric(data['hemo'],errors='coerce')
data['pcv']=pd.to_numeric(data['pcv'],errors='coerce')
data['wbcc']=pd.to_numeric(data['wbcc'],errors='coerce')
data['rbcc']=pd.to_numeric(data['rbcc'],errors='coerce')


print(data.describe())
print(data.mode())

nominalColumns=['sg','al','su','rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','class']


def showData(data):
    for col in data:
        if(col in nominalColumns):
            sea.countplot(col, data=data, hue='class')
        else:
            sea.distplot(data[[col]],hist=False)
        plt.show()

print(data.count())
#showData(data)

dataNotNan=data.dropna(axis=0)
print(dataNotNan.shape)

dataNotPCVNotWBCC=data.drop(axis=1,columns=['wbcc','pcv'])
print(dataNotPCVNotWBCC.shape)

#da li ima duplikata
print(data.drop_duplicates().shape)

def indicies_of_outliers(x):
    q1=x.quantile(0.25)
    q3=x.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    print( ( x > upper_bound) | (x < lower_bound))


print(indicies_of_outliers(data))

def removeOutliers(data):
    for ind,row in data.iterrows():
        for col in data:
            if(col not  in nominalColumns):
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                if((row[col]< (Q1 - 1.5 * IQR))|(row[col] > (Q3 + 1.5 * IQR))):
                    data=data.drop([ind])
                    break


    return data

dataWoOutliers=removeOutliers(data)
print(dataWoOutliers)

def nominalToNumeric(data):
    le = LabelEncoder()
    for col in nominalColumns:
        le.fit(data[col])
        data[col]=le.transform(data[col])
    return data


def impute(data):
    imputedDataMean = pd.DataFrame()
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mostFrequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    for col in data:
        if (col in nominalColumns):
            imp_mostFrequent = imp_mostFrequent.fit(data[[col]])
            imputedDataMean[col] = imp_mostFrequent.transform(data[[col]]).ravel()

        else:
            imp_mean = imp_mean.fit(data[[col]])
            imputedDataMean[col] = imp_mean.transform(data[[col]]).ravel()
    return imputedDataMean

def MinMaxScale(data):
    for col in data:
        scale=MinMaxScaler(feature_range=(-1, 1))
        data = pd.DataFrame(scale.fit_transform(data.values), columns=data.columns, index=data.index)
    return data

def StandardScale(data):
    scale=StandardScaler()
    datapom=pd.DataFrame()
    clas = data['class']
    datapom = data.loc[:, data.columns != 'class']
    datapom = pd.DataFrame(scale.fit_transform(datapom.values), columns=datapom.columns, index=data.index)
    datapom['class'] = clas
    return datapom

#KLASIFIKACIJA
#Data=data.loc[:,data.columns!='class']
#target=pd.DataFrame()
#target['class']=data['class']

def Gaussan(data,name,visualize,t):
    Data = data.loc[:, data.columns != 'class']
    target = pd.DataFrame()
    target['class'] = data['class']
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=t, random_state=42,shuffle=True)
    gaus=GaussianNB()
    pred=gaus.fit(data_train,target_train).predict(data_test)
    print("Naive Bayes "+name+"  accuracy: ", accuracy_score(target_test,pred,normalize=True))
    print(classification_report(target_test,pred))
    if (visualize):

        plot_roc_curve(gaus, data_test, target_test)
        print(confusion_matrix(target_test,pred))
        plt.show()
    return {'Gaus':{'datatrain':data_train,'targettrain':target_train,'datatest':data_test,'targettest':target_test,'name':name}}
def KNN(data,name,visualize,t):
    Data = data.loc[:, data.columns != 'class']
    target = pd.DataFrame()
    target['class'] = data['class']
    data_train, data_test, target_train, target_test = train_test_split(Data, target, test_size=t, random_state=42,shuffle=True)
    neigh=KNeighborsClassifier(n_neighbors=5)
    pred = neigh.fit(data_train, target_train).predict(data_test)
    print("KNN "+name+"  accuracy: ", accuracy_score(target_test, pred, normalize=True))
    print(classification_report(target_test, pred))
    if(visualize):
        plot_roc_curve(neigh,data_test,target_test)
        print(confusion_matrix(target_test, pred))
        plt.show()
    return {
        'KNN': {'datatrain': data_train, 'targettrain': target_train, 'datatest': data_test, 'targettest': target_test,
                 'name': name}}
def LRegression(data,name,visualize,t):
    target = pd.DataFrame()
    target['class'] = data['class']
    Data = data.loc[:, data.columns != 'class']
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=t, random_state=42,
                                                                        shuffle=True)
    lr =LogisticRegression()
    pred = lr.fit(data_train, target_train).predict(data_test)
    print("Logistic Regression "+name+" accuracy: ", accuracy_score(target_test, pred, normalize=True))
    print(classification_report(target_test, pred))
    if (visualize):
        plot_roc_curve(lr, data_test, target_test)
        print(confusion_matrix(target_test, pred))
        plt.show()
    return {
        'LR': {'datatrain': data_train, 'targettrain': target_train, 'datatest': data_test, 'targettest': target_test,
                'name': name}}
def DTree(data,name,visualize,t):
    Data = data.loc[:, data.columns != 'class']
    target = pd.DataFrame()
    target['class'] = data['class']
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=t, random_state=42,
                                                                        shuffle=True)
    dt = DecisionTreeClassifier(criterion='gini',max_depth=10)
    pred = dt.fit(data_train, target_train).predict(data_test)
    print("Decision tree"+name+" accuracy: ", accuracy_score(target_test, pred, normalize=True))
    print(classification_report(target_test, pred))
    if (visualize):
        plot_roc_curve(dt, data_test, target_test)
        print(confusion_matrix(target_test, pred))
        plt.show()
    return {
        'DT': {'datatrain': data_train, 'targettrain': target_train, 'datatest': data_test, 'targettest': target_test,
               'name': name}}







#Gaussan(impute(data))


dataWoOutliers=impute(dataWoOutliers)
dataNotPCVNotWBCC=impute(dataNotPCVNotWBCC)
dataNotPCVNotWBCC=nominalToNumeric(dataNotPCVNotWBCC)

def callAllClassifiers(data,name,visualize,test):
    KNN(data,name,visualize,test)
    Gaussan(data,name,visualize,test)
    LRegression(data,name,visualize,test)
    DTree(data,name,visualize,test)


d1=pd.DataFrame()
d1=impute(data)
d1=nominalToNumeric(d1)
print(d1)
callAllClassifiers(d1,'Imputed data',True,0.3)
callAllClassifiers(d1,'Imputed data',True,0.2)
callAllClassifiers(MinMaxScale(d1),'Min Max scale imuted Data',False,0.3)
#print(StandardScale(impute(data)))
callAllClassifiers(StandardScale(d1),'Standard scale scale imuted Data',False,0.3)
callAllClassifiers(nominalToNumeric(dataWoOutliers),'data without outliers',False,0.3)
callAllClassifiers(dataNotPCVNotWBCC,'data without some columns',False,0.3)

#FeatureSelection
Data = d1.loc[:, d1.columns != 'class']
target = pd.DataFrame()
target['class'] = d1['class']
model=ExtraTreesClassifier(n_estimators=10)

model.fit(Data,target)

feat_importances = pd.Series(model.feature_importances_, index=Data.columns)
print(feat_importances)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
callAllClassifiers(d1[['hemo','htn','sg','al','dm','rbcc','pcv','appet','rbc','pcc','class']],'Obican samo sa Feature selectionom',True,0.3)

Data = MinMaxScale(d1).loc[:, d1.columns != 'class']
target = pd.DataFrame()
target['class'] = d1['class']
model=ExtraTreesClassifier(n_estimators=10)

model.fit(Data,target)

feat_importances = pd.Series(model.feature_importances_, index=Data.columns)
print(feat_importances)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
callAllClassifiers(d1[['hemo','htn','sg','al','dm','rbcc','pcv','appet','rbc','pcc','class']],'Obican samo sa Feature selectionom',True,0.3)


plt.figure(figsize=(16,5))
print('Regular data')
sea.heatmap(d1.corr(), annot = True,fmt='.1g')
plt.show()
print('Min max scaler data')
sea.heatmap(MinMaxScale(d1).corr(), annot = True)
plt.show()
print('Standard scaler data')
sea.heatmap(StandardScale(d1).corr(), annot = True)
plt.show()
print('No outlier data')
sea.heatmap(nominalToNumeric(dataWoOutliers).corr(), annot = True)

plt.show()

#Reducing Features Feature extraction
