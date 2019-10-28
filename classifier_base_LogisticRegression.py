from input_fn.input_fn_text_classifier import categorize_dataset
from input_fn.input_fn_text_classifier import splitting_data
from util.util_text_classifier import vectorize_sentences

dataset_dictionary ={'Dataset':  'data/dataset.txt'}   # creating a dictionary
# with key as source of data. And the value part of dictionary is
# the address of the dataset.
categorized_dataset = categorize_dataset(dataset_dictionary)   # returns data organized in class, text and
# source of dataset
sentences_train, sentences_test, y_train, y_test = splitting_data(categorized_dataset)  # it splits the organized
# dataset by the function "categorize_dataset" into test(0.25) and train(0.75).
X_train, X_test = vectorize_sentences(sentences_train, sentences_test)  # creates feature vectors for all the sentences
# by using Bag-of-Words(BoW) model.

from sklearn.linear_model import LogisticRegression  # using LogisticRegression from scikit-learn library. It is simple
# and powerful linear model that does mathematical regression between 0 and 1 based on input feature vector and with
# cutoff value(by default 0.5) which is used for classification task.
classifier = LogisticRegression()
classifier.fit(X_train, y_train)  # Fits the model according to the given training data
score = classifier.score(X_test, y_test)  # Returns the mean accuracy on the given test data and labels
print("Accuracy:", score)  # Prints the accuracy
