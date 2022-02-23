import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

df = pd.read_csv("TensorFlow-Pokemon-Course\pokemon.csv")
print(df.columns)