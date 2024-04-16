# sivakumar-guruswamy
Coding
from sklearn.model_selection import cross_val_score
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph. I like it most for plot
##from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer ### best way to impute both numerical & categorical variables
impute_mode = SimpleImputer(strategy = 'most_frequent')
data=pd.read_csv("/content/horse.csv")
df=pd.DataFrame(data)
df.head(3)
impute_mode.fit(df[['pulse','respiratory_rate','temp_of_extremities','peripheral_pulse','mucous_membrane','capillary_refill_time',
                   'pain','peristalsis','abdominal_distention','nasogastric_tube','nasogastric_reflux','nasogastric_reflux_ph',
                    'rectal_exam_feces','abdomen','packed_cell_volume','abdomo_appearance','rectal_temp','total_protein','abdomo_protein']])
df[['pulse','respiratory_rate','temp_of_extremities','peripheral_pulse','mucous_membrane','capillary_refill_time',
                   'pain','peristalsis','abdominal_distention','nasogastric_tube','nasogastric_reflux','nasogastric_reflux_ph',
                    'rectal_exam_feces','abdomen','packed_cell_volume','abdomo_appearance','rectal_temp','total_protein','abdomo_protein']] = impute_mode.transform(df[['pulse','respiratory_rate','temp_of_extremities','peripheral_pulse','mucous_membrane','capillary_refill_time',
                   'pain','peristalsis','abdominal_distention','nasogastric_tube','nasogastric_reflux','nasogastric_reflux_ph',
                    'rectal_exam_feces','abdomen','packed_cell_volume','abdomo_appearance','rectal_temp','total_protein','abdomo_protein']])
df.isnull().sum()
# distinction is based on the number of different values in the column
columns = list(df.columns)

categoric_columns = []
numeric_columns = []

for i in columns:
    if len(df[i].unique()) > 6:
        numeric_columns.append(i)
    else:
        categoric_columns.append(i)

categoric_columns = categoric_columns[:-1]

print('Numerical fetures: ',numeric_columns)
print('Categorical fetures: ',categoric_columns)

df['pulse'].value_counts()

##df['age']=df['age'].map({'adult':1,'young':0})
##df['age']
### 1st way of one hot encoding was not working fine
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
# Encoding multiple columns. Unfortunately you cannot pass a list here so you need to copy-paste all printed categorical columns.
transformer = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'),
     ['surgery', 'age', 'temp_of_extremities', 'peripheral_pulse', 'capillary_refill_time', 'pain', 'peristalsis',
      'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux', 'rectal_exam_feces', 'abdomen', 'abdomo_appearance',
      'outcome', 'surgical_lesion', 'lesion_2', 'lesion_3']))
### 2nd way of one hot encoding
#Extract categorical columns from the dataframe
#Here we extract the columns with object datatype as they are the categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

#Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Apply one-hot encoding to the categorical columns
one_hot_encoded = encoder.fit_transform(df[categorical_columns])

#Create a DataFrame with the one-hot encoded columns
#We use get_feature_names_out() to get the column names for the encoded data
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the one-hot encoded dataframe with the original dataframe
df_encoded = pd.concat([df, one_hot_df], axis=1)

# Drop the original categorical columns
df_encoded = df_encoded.drop(categorical_columns, axis=1)

# Display the resulting dataframe
print(f"Encoded Employee data : \n{df_encoded}")
## 3rd way for the one hot encoding using dummies, its not working
df1=pd.get_dummies(df,drop_first=True)
df1.head(3)
