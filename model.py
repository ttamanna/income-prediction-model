import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

train_data = pd.read_csv('./adult.data')
test_data1 = pd.read_csv('./adult.test')
test_data = pd.read_csv('./adult.test', header =1)  # to remove the header row

# Adding column names to train and test data
columns = ['age', 'workclass', 'fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
train_data.columns=columns
test_data.columns=columns

# train_data.head()
# test_data_head()

# train_data.shape
# test_data.shape

## Processing Train data
'''
#Checking null values
print(train_data.isnull().sum())

# Checking info
train_data.info()
print(train_data.select_dtypes(exclude=np.number).columns)

# Checking unusual values
train_data['workclass'].value_counts()
train_data['marital-status'].value_counts()
train_data['occupation'].value_counts()
train_data['relationship'].value_counts()
train_data['race'].value_counts()
train_data['sex'].value_counts()
train_data['native-country'].value_counts()
train_data['income'].value_counts()
'''
# Removing all 'nan' or '?' values
train_data1 = train_data.applymap(lambda x: float('nan') if isinstance(x, str) and x.strip() == '?' else x).dropna()

data = train_data1  # copying the train data for processing

# Categorical encoding for the 'object' values
data['workclass_codes']=data['workclass'].astype('category').cat.codes+1
data['marital-status_codes']=data['marital-status'].astype('category').cat.codes+1 
data['occupation_codes']=data['occupation'].astype('category').cat.codes+1
data['relationship_codes']=data['relationship'].astype('category').cat.codes+1
data['race_codes']=data['race'].astype('category').cat.codes+1
data['sex_codes']=data['sex'].astype('category').cat.codes+1
data['native-country_codes']=data['native-country'].astype('category').cat.codes+1
data['income_codes']=data['income'].astype('category').cat.codes+1

# Categories representing the codes
workclass_codes = data['workclass'].astype('category').cat.categories
marital_status_codes = data['marital-status'].astype('category').cat.categories
occupation_codes = data['occupation'].astype('category').cat.categories
relationship_codes= data['relationship'].astype('category').cat.categories
race_codes = data['race'].astype('category').cat.categories
sex_codes = data['sex'].astype('category').cat.categories
native_country_codes= data['native-country'].astype('category').cat.categories
income_codes_train= data['income'].astype('category').cat.categories

'''
# Printing the categories
print(workclass_codes)
print(marital_status_codes)
print(occupation_codes)
print(relationship_codes)
print(native_country_codes)
print(race_codes)
print(sex_codes)
print(income_codes_train)
'''

# Dropping the other columns to keep only numerical ones
train_data_final = data.drop(['workclass', 'education', 'marital-status', 'occupation',
       'relationship', 'race', 'sex', 'native-country', 'income'], axis=1)

print(train_data_final.head())

# Creating a csv file
# train_data_final.to_csv('./train_data_final.csv', index=False)

## Processing Test data
'''
#Checking null values
print(test_data.isnull().sum())

# Checking info
test_data.info()
print(test_data.select_dtypes(exclude=np.number).columns)

# Checking unusual values
test_data['workclass'].value_counts()
test_data['marital-status'].value_counts()
test_data['occupation'].value_counts()
test_data['relationship'].value_counts()
test_data['race'].value_counts()
test_data['sex'].value_counts()
test_data['native-country'].value_counts()
test_data['income'].value_counts()
'''
# Removing all nan or '?' values
test_data1 = test_data.applymap(lambda x: float('nan') if isinstance(x, str) and x.strip() == '?' else x).dropna()

data1 = test_data1 # copying the test data for processing

# Categorical encoding for the 'object' values
data1['workclass_codes']=data1['workclass'].astype('category').cat.codes+1
data1['marital-status_codes']=data1['marital-status'].astype('category').cat.codes+1
data1['occupation_codes']=data1['occupation'].astype('category').cat.codes
data1['relationship_codes']=data1['relationship'].astype('category').cat.codes+1
data1['race_codes']=data1['race'].astype('category').cat.codes+1
data1['sex_codes']=data1['sex'].astype('category').cat.codes+1
data1['native-country_codes']=data1['native-country'].astype('category').cat.codes+1
data1['income_codes']=data1['income'].astype('category').cat.codes+1


# Dropping the other columns to keep only numerical ones

test_data_final = data1.drop(['workclass', 'education', 'marital-status', 'occupation',
       'relationship', 'race', 'sex', 'native-country', 'income'], axis=1)

print(test_data_final.head())

# test_data_final.to_csv('./test_data_final.csv', index=False)

## Training annd Testing data for the model

target_train = train_data_final['income_codes']
target_test = test_data_final['income_codes']

features_train = train_data_final.drop(['income_codes'], axis=1)
features_test = test_data_final.drop(['income_codes'], axis=1)

# Training the KNN model
knn = KNeighborsClassifier()
knn.fit(features_train, target_train)

# Testing the model
predicted = knn.predict(features_test)
expected = target_test

# Displaying wrong predictions
wrong = [(p,e) for (p,e) in zip(predicted, expected) if p !=e]
# print(wrong)

# Accuracy of the model
print("Accuracy without dropping less correlated variables:", format(knn.score(features_test, target_test), ".2%"))


## Modification using correlation 

# Checking correlation with income
correlation_with_income = train_data_final.corrwith(train_data_final['income_codes'])

print(correlation_with_income)

# Dropping the columns with insignificant correlation : 'fnlwgt', 'marital-status_codes', 'relationship_codes'
# new train and test data

to_drop = ['fnlwgt', 'native-country_codes','race_codes']

modified_test_data = test_data_final.drop(to_drop, axis=1)

modified_train_data = train_data_final.drop(to_drop, axis=1)

## Training annd Testing modified data for the model

target_modified_train = modified_train_data['income_codes']
target_modified_test = modified_test_data['income_codes']

features_modified_train = modified_train_data.drop(['income_codes'], axis=1)
features_modified_test = modified_test_data.drop(['income_codes'], axis=1)

# Training the KNN model 
knn1 = KNeighborsClassifier()
knn1.fit(features_modified_train, target_modified_train)

# Testing the model
predicted1 = knn1.predict(features_modified_test)
expected1 = target_modified_test

# Displaying wrong predictions
wrong1 = [(p,e) for (p,e) in zip(predicted1, expected1) if p !=e]
# print(wrong1)
print(len(wrong1))

# Confusion matrix
# Print the confusion matrix
confusion = confusion_matrix(y_true=expected1, y_pred=predicted1)
print(confusion)

# Accuracy of the model
print("Accuracy after dropping less correlated variables:",format(knn1.score(features_modified_test, target_modified_test), ".2%"))

# Heatmap
confusion_df = pd.DataFrame(confusion, index=range(2), columns=range(2))
figure = plt.figure(figsize = (7,6))

axes = sns.heatmap(confusion_df, annot = True, cmap = plt.cm.nipy_spectral_r, fmt='g')

plt.show()