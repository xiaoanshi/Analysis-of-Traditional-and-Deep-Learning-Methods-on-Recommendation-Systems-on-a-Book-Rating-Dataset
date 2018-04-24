# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 22:54:12 2018

@author: rsaluja
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 03:17:04 2018

@author: rsaluja
"""

## All Libraries used.

import numpy as np   
import pandas as pd    # Used for Dataframes. 
from sklearn.metrics import mean_squared_error # Used for Mean Squared Error
import time            # Used for Measuring the time
import matplotlib.pyplot as plt  # Used for Plotting
import random


## Reading the CSV File.

# Reading this CSV is done a liitle differently as there is ";" instead ","(Comma).
# Postprocessing to make the proper Dataframe is in the next Cell. 

df = pd.read_csv('BX-Book-Ratings.csv',sep='delimiter') # Reading the CSV File

## Preprocessing to form the proper Dataset.

series = df['"User-ID";"ISBN";"Book-Rating"'].astype(str)  # Converting into Series to perform the operations

df = pd.DataFrame(series.str.split(';',expand=True)) # Splitting the Series(strings) with a delimeter (';')

df.columns = ['User-ID', 'ISBN', 'Ratings'] # Changing the Column Name

## Removing the  '""' from each column and converting them into intergers. 

df['User-ID'] = df['User-ID'].map(lambda x: x.lstrip('"').rstrip('"'))
df['ISBN'] = df['ISBN'].map(lambda x: x.lstrip('"').rstrip('"'))
df['Ratings'] = df['Ratings'].map(lambda x: x.lstrip('"').rstrip('"'))

df['User-ID'] = df['User-ID'].astype(int)
df['Ratings'] = df['Ratings'].astype(int)
df['ISBN'] = df['ISBN'].astype(str)

## Outputting the Ratings Data Frame

ratings_df = df

## Filtering out impotant books to make the dataset smaller

# users with less than 200 ratings and books with less than 100 ratings are excluded

c = ratings_df['User-ID'].value_counts()

ratings_df = ratings_df[ratings_df['User-ID'].isin(c[c >= 200].index)]

c = ratings_df['Ratings'].value_counts()

ratings_df = ratings_df[ratings_df['Ratings'].isin(c[c >= 100].index)]

ratings_df.head()

"""# Number of Users & Books"""

num_users = ratings_df['User-ID'].unique().shape[0]
num_books = ratings_df['ISBN'].unique().shape[0]
print(str(num_users) + ' users')
print(str(num_books) + ' books')

def data_splitter(ratings_df, percentage):
    m,n = ratings_df.shape
    V = round(m*(100-percentage)/100)
    A = ratings_df.copy()
    
    if percentage == 100:
        return A
    
    
    a = random.sample(range(ratings_df.shape[0]),int(V))
    
    A['Ratings'].iloc[a] = np.nan
    
    return A



def train_test_split(ratings_df,percentage):
  m,n = ratings_df.shape
  V = round(m*(100-percentage)/100)
  train = ratings_df.copy()


  a = random.sample(range(ratings_df.shape[0]),int(V))

  train['Ratings'].iloc[a] = np.nan
  
  
  test = train.copy()

  test = test.fillna(1000)
  
  test.ix[test['Ratings'] < 15, 'Ratings'] = np.nan

  test.ix[test['Ratings'] == 1000, 'Ratings'] = 0
  
  test['Ratings'] = test['Ratings'] + ratings_df['Ratings']
  
  
  return train,test


## Training Data (75%) of the Data

train_df, test_df = train_test_split(ratings_df, 75)


print(train_df.shape)

print(test_df.shape)



train_mat = train_df.pivot(index ='User-ID', columns ='ISBN', values ='Ratings').fillna(0).values

test_mat = test_df.pivot(index ='User-ID', columns ='ISBN', values ='Ratings').fillna(0).values

## Function for Obtaining the Error

def rmse(Mat, Q, P):
    I = (Mat != 0)  
    er = I * (Mat - np.dot(P, Q.T))  
    err = er**2  
    RMSE = np.sqrt(np.sum(err)/np.sum(I)) 
    return  RMSE

## Defining Hyper-Parameters

L = 20  # Number of latent factors
lmbda = 0.3 # Regularisation Factor
eta = 0.0001  # Learning rate
epoch = 50  # Number of loops through training data


train_errors = []
test_errors = []
users,items = train_mat.nonzero()

def svd_sgd(train_mat,test_mat,L,lmbda,eta,epoch):
    
    print 'SGD Begins!!'
    
    
    train_errors = []
    test_errors = []
    users,items = train_mat.nonzero()
    
    P =  4*np.random.rand(train_mat.shape[0], L) 
    Q =  4*np.random.rand(train_mat.shape[1], L)
    
    for z in range(epoch):
        print 'SGD-Epoch ' + str(z)
        
        for u, i in zip(users,items):
            e = train_mat[u,i] - np.dot(P[u,:], Q[i,:].T)  
            P[u,:] += eta*(e*Q[i,:] - lmbda*P[u,:]) 
            Q[i,:] += eta*(e*P[u,:] - lmbda*Q[i,:])
            
        train_errors.append(rmse(train_mat,Q,P)) 
        print 'Training Error ='+ str(rmse(train_mat,Q,P))
        test_errors.append(rmse(test_mat,Q,P)) 
        print 'Testing Error ='+ str(rmse(test_mat,Q,P))    
        
    print 'SGD Ends!!'
    
    iteration = range(1,epoch+1)
    
    plt.plot(iteration, train_errors,'r',label='Training Error')
    plt.plot(iteration, test_errors,'b',label='Testing Error')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('SVD using SGD - RMSE v/s Epoch')
    plt.grid()
    plt.legend()
    plt.show()
    
    return P,Q,train_errors,test_errors



P,Q,train_errors,test_errors = svd_sgd(train_mat,test_mat,L,lmbda,eta,epoch)


