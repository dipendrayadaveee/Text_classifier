import os
directory_list = os.listdir('../sample_raw_dataset')   # get the list of all the files in the given directory
for f in directory_list:  # for each file in the list
    if f != 'data_preprocessing.py':  # This step is to skip this code file which also is in the same folder
        with open(f, "r") as file:  # Opens the file in read mode
            lines = file.readlines()  # Reads all the lines present in the file
            for line in lines:  # Considers one line at a time
                print(line)  # Prints the contents of the line
    else:
        pass