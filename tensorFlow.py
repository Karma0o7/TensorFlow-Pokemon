# Importing Modules
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Function to create dummy
# In dummy_categories give an list of names of column to change to dummy
def dummy_creation(df, dummy_categories):
    for i in dummy_categories:
        # get_dummies converts them to dummy variables
        df_dummy = pd.get_dummies(df[i])
        df = pd.concat([df,df_dummy],axis = 1)
        df = df.drop(i, axis=1)
    return df

# Function to split csv into training(70% of file) and testing(30% of file only gen 1) DataFrames
def train_test_splitter(DataFrame, column):
    # .loc is used to get the all the data in a column and works with boolean values like here.
    df_train = DataFrame.loc[df[column] != 1]
    df_test = DataFrame.loc[df[column] == 1]
    df_train = df_train.drop(column, axis=1)
    df_test = df_test.drop(column, axis=1)

    return(df_train,df_test)

# Function to create data and label.
def label_delineator(df_train, df_test, label):
    train_data = df_train.drop(label, axis=1).values
    train_labels = df_train[label].values
    test_data = df_test.drop(label, axis=1).values
    test_labels = df_test[label].values
    
    return(train_data, train_labels, test_data, test_labels)


# Function to Normalize the given Data
def data_normalizer(train_data, test_data):
    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    
    return(train_data, test_data)


# Function to take the test data and the predict specific value using it
def predictor(test_data, test_labels, index):
    prediction = model.predict(test_data)
    if np.argmax(prediction[index]) == test_labels[index]:
        print(f'This was correctly predicted to be a \"{test_labels[index]}\"!')
    else:
        print(f'This was incorrectly predicted to be a \"{np.argmax(prediction[index])}\". It was actually a \"{test_labels[index]}\".')
        return(prediction)


# Import the csv using Pandas
df = pd.read_csv(".\pokemon.csv")
# Select the columns to work with.
df = df[['isLegendary','Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Color','Egg_Group_1','Height_m','Weight_kg','Body_Style','hasMegaEvolution']]
# Change the bool type value to int type 1 or 0.
df['isLegendary'] = df['isLegendary'].astype(int)

#Get users pokemon index
poke_index = int(input("Enter the Index of your pokemon: "))
#Enter the trait to check
print(['isLegendary','Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Color','Egg_Group_1','Height_m','Weight_kg','Body_Style','hasMegaEvolution'])
poke_trait = str(input("Choose form Above Options: "))
if poke_trait not in df:
    exit(1)

# create Dummy veriables
df = dummy_creation(df, ['Egg_Group_1', 'Body_Style', 'Color','Type_1', 'Type_2'])
# Slpit the given data or csv file and get the DataFrame. On the basis of generation.
df_train, df_test = train_test_splitter(df, 'Generation')
# Split the DataFrame into data and labels
train_data, train_labels, test_data, test_labels = label_delineator(df_train, df_test, poke_trait)
# Normalize the obtained data
train_data, test_data = data_normalizer(train_data, test_data)

length = train_data.shape[1]
model = keras.Sequential()
model.add(keras.layers.Dense(500, activation='relu', input_shape=[length,]))
model.add(keras.layers.Dense(2, activation='softmax'))
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=400)
loss_value, accuracy_value = model.evaluate(test_data, test_labels)
print(f'Our test accuracy was {accuracy_value}')

# Use the predictor. Change Index Value to pokemon gen value to see different pokemons.
predictor(test_data, test_labels, poke_index-1)

