import pandas as pd  # Using pandas python library to process the data
from sklearn.model_selection import train_test_split


def categorize_dataset(dataset_dictionary):
    ''' This function categorizes the dataset into different columns with labels.
    It returns data organized in class, text and source of dataset column
    Sample 0th row returned for dataset:
    class                                                     0
    text      How do you wonder whether John said Mary solve...
    origin                                Dataset '''

    dataset_list = []  # initializing a list to store all the dataset with label and text.
    for origin, address in dataset_dictionary.items():  # origin:dataset and address:path of file
        data = pd.read_csv(address, names=['class', 'text'], sep='\t')  # reading the dataset and naming the first
        # column as class and second column as text which are seperated by tabs.
        data['origin'] = origin
        dataset_list.append(data)
    data_final = pd.concat(dataset_list)  # final dataset contains the first column with class of the text and
    # the column is labelled as class too. Similarly, second column contains text and column labelled as text.
    # Todo: uncomment the line below if needed to see the 0th row of the returned dataset
    # print(data_final.iloc[0]) #to simply have a look at the first line of dataset
    return data_final


def splitting_data(categorized_dataset):
    '''This function takes the output dataset from the categorize_dataset() function and divides it into test and
     train datasets. It returns splitted train and test sentences and their respective labels'''
    data_final=categorized_dataset
    data_final_mls = data_final[data_final['origin'] == 'Dataset']  #This step basically makes sure that
    # the data_final_mls contains data from Dataset. This is useful when we have data from many sources.
    sentences = data_final_mls['text'].values  # returns a NumPy array of text
    y = data_final_mls['class'].values  # returns a NumPy array of class
    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
    #  we set aside 0.25 portion of dataset to be used as test data
    return sentences_train, sentences_test, y_train, y_test
