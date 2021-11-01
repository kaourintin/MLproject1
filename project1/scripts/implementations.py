# Useful starting lines
import numpy as np
from proj1_helpers import *

#Pensez à retirer 'fermeli' du path
DATA_TRAIN_PATH = '/home/fermeli/ML_course/projects/project1/data/train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

JET_NUM_INDEX = 22

def clean_data(x):
    #replace -999 by the mean of the valid features of the column
    x[x == -999] = np.NaN

    #remove all columns exclusively composed of nan
    x = x[:,~np.all(np.isnan(x), axis=0)]
    #removed all columns with identical values (where std is 0)
    
    #x = x[:,~np.all(np.std(x) == 0.0, axis=0)]
    
    std = np.squeeze(np.std(x,axis=0))

        
    to_delete = []
    indices_deleted = []
    for i in range(len(std)):
        if(std[i] == 0 or np.isnan(std[i])):
            to_delete.append(False)
            indices_deleted
        else:
            to_delete.append(True)
    
    x = x[:,to_delete]#np.delete(x,18,0)
    
    
    
    mean = np.nanmean(x,axis=0)
    inds = np.where(np.isnan(x))    
    x[inds]= np.take(mean, inds[1])
    
    #standardize features 
    std= np.std(x,axis=0)
        
    newMean = np.nanmean(x,axis=0)
    
    return (x-newMean)/std

def split_matrix(x, y,ids):
    x0 = []
    x1 = []
    x2 = []
    x3 = []
    y0 = []
    y1 = []
    y2 = []
    y3 = []
    i0 = []
    i1 = []
    i2 = []
    i3 = []
    
    jet_num_id = JET_NUM_INDEX
    for i in range(x.shape[0]):
        if x[i][jet_num_id] == 0:
            x0.append(x[i])
            y0.append(y[i])
            i0.append(ids[i])
        elif x[i][jet_num_id] == 1:
            x1.append(x[i])
            y1.append(y[i])
            i1.append(ids[i])
        elif x[i][jet_num_id] == 2:
            x2.append(x[i])
            y2.append(y[i])
            i2.append(ids[i])
        elif x[i][jet_num_id] == 3:
            x3.append(x[i])
            y3.append(y[i])
            i3.append(ids[i])
    return np.asarray(x0), np.asarray(x1), np.asarray(x2), np.asarray(x3), np.asarray(y0), np.asarray(y1), np.asarray(y2),np.asarray( y3),i0,i1,i2,i3

def calculate_mse(e):
   return 1/2*np.mean(e**2)

def ridge_regression_solve(y, tx, lambda_):
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

def ridge_regression(y, tx, lambda_ ):
    y= y.reshape(y.shape[0],1)
    w = ridge_regression_solve(y, tx, lambda_)
    err = y - tx.dot(w)
    rmse = np.sqrt(2 * calculate_mse(err))
    return rmse, w

def expand_X_with_pairwise_products(X, M):
    
    expand_without_pairwise_products = np.ones((X.shape[0],1))
    for idx in range(1,M+1): expand_without_pairwise_products=np.hstack((expand_without_pairwise_products, X**idx))

    # create the interactions between two variable
    # X is (N, d), we first make it as (N, d, 1) and (N, 1, d), then compute the interaction
    X_inter = np.expand_dims(X, axis=1)
    X_inter_ = np.expand_dims(X, axis=2)
    full_interactions = np.matmul(X_inter_, X_inter)
    # np.triu_indices: Return the indices for the upper-triangle of a matrix
    indices = np.triu_indices(full_interactions.shape[1], k=1)
    interactions = np.zeros((X.shape[0], len(indices[0])))
    for n in range(X.shape[0]):
        interactions[n] = full_interactions[n][indices]
    #print(interactions.shape)

    return np.concatenate((expand_without_pairwise_products, interactions), axis=1)
    

x0,x1,x2,x3,y0,y1,y2,y3,_,_,_,_ = split_matrix(tX, y,np.zeros(len(tX)))

x0 = clean_data(x0)
x1 = clean_data(x1)
x2 = clean_data(x2)
x3 = clean_data(x3)

lambda0 = 0.30888435964774846
lambda1 = 0.002811768697974231
lambda2 = 0.0010985411419875584
lambda3 = 0.0010985411419875584

poly0 = expand_X_with_pairwise_products(x0,2)
poly1 = expand_X_with_pairwise_products(x1,2)
poly2 = expand_X_with_pairwise_products(x2,2)
poly3 = expand_X_with_pairwise_products(x3,2)

_,w0 = ridge_regression(y0, poly0, lambda0)
_,w1 = ridge_regression(y1, poly1, lambda1)
_,w2 = ridge_regression(y2, poly2, lambda2)
_,w3 = ridge_regression(y3, poly3, lambda3)

DATA_TEST_PATH = '/home/fermeli/ML_course/projects/project1/data/test.csv' # TODO: download train data and supply path here ,
_,tXtest, ids_test = load_csv_data(DATA_TEST_PATH)



x0,x1,x2,x3,y0,y1,y2,y3,i0,i1,i2,i3 = split_matrix(tXtest,np.zeros(len(tXtest)), ids_test)
x0 = clean_data(x0)
x1 = clean_data(x1)
x2 = clean_data(x2)
x3 = clean_data(x3)




xs = [x0,x1,x2,x3]
ws = [w0,w1,w2,w3] 

y_pred = []
for i in range(len(xs)):
    xi = expand_X_with_pairwise_products(xs[i], 2)
    xs[i] = None
    y_pred.append(predict_labels(ws[i], xi))

y_pred0 = y_pred[0]
y_pred1 = y_pred[1]
y_pred2 = y_pred[2]
y_pred3 = y_pred[3]


result0 = []
for i in range(len(y_pred0)):
    result0.append((i0[i],y_pred0[i]))

result1 = []
for i in range(len(y_pred1)):
    result1.append((i1[i],y_pred1[i]))

result2 = []
for i in range(len(y_pred2)):
    result2.append((i2[i],y_pred2[i]))

result3 = []
for i in range(len(y_pred3)):
    result3.append((i3[i],y_pred3[i]))
 

result = np.concatenate([result0,result1,result2,result3])

result = result[result[:, 0].argsort()]
predictions = np.delete(result,0,axis=1)

OUTPUT_PATH = '/home/fermeli/ML_course/projects/project1/data/sample-submission-test_script.csv' # TODO: fill in desired name of output file for submission
#y_pred = predict_labels(w, poly)
create_csv_submission(ids_test, predictions, OUTPUT_PATH)

