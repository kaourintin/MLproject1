# MLproject1
This project is designed to find the best possible solution to the boson higs dataset. The output is -1 or 1 depending on if the specific input row
is the result of an experiment with a signal decay from the boson or if it is not.


INITIAL SETUP : 
We assume you have a working python workflow, with numpy packages.

=> When using the provided script and algorithm, don't forget to change the address to the path to point towards the data.
=> implementation.py has to be in the same folder as the helpers.


DATA CLEANING AND SPLITTING FUNCTION

the split_matrix function allows the user to split the dataset into four different sets depending on a specific value of PR_jet_Num. It also does the same operation on the y prediction
set and the indices so that the user is able to reconstruct the sets after prediction.
the clean_data function cleans the data accordingly to the choices we made, and outputs the cleaned dataset.


DATA ANALYSIS FUNCTIONS
We have provided six staple functions of machine learning that are quite popular : 

least squares GD(y, tx, initial w,max iters, gamma)  Linear regression using gradient descent

least squares SGD(y, tx, initial w,max iters, gamma) Linear regression using stochastic gradient descent

least squares(y, tx) Least squares regression using normal equations

ridge regression(y, tx, lambda ) Ridge regression using normal equations

logistic regression(y, tx, initial w,max iters, gamma) Logistic regression using gradient descent or SGD (y ∈{0, 1})

reg logistic regression(y, tx, lambda ,initial w, max iters, gamma) Regularized logistic regression using gradient descentor SGD (y ∈ {0, 1}, with regularized term λkwk2)



