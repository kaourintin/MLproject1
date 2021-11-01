# MLproject1
This project is designed to find the best possible solution to the boson higs dataset. The output is -1 or 1 depending on if the specific input row
is the result of an experiment with a signal decay from the boson or if it is not.


################################################DATA CLEANING AND SPLITTING FUNCTIONS##############################################################################################
the split_matrix function allows the user to split the dataset into four different sets depending on a specific value of PR_jet_Num. It also does the same operation on the y prediction
set and the indices so that the user is able to reconstruct the sets after prediction.
the clean_data function cleans the data accordingly to the choices we made, and outputs the cleaned dataset.




=> When using the provided script and algorithm, don't forget to change the address to the path to point towards the data.
=> implementation.py has to be in the same folder as the helpers.
