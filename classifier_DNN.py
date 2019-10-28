from input_fn.input_fn_text_classifier import categorize_dataset
from input_fn.input_fn_text_classifier import splitting_data
from util.util_text_classifier import vectorize_sentences
from model_fn.model_fn_text_classifiers import DNN_basic
from util.util_text_classifier import plot_history

dataset_dictionary ={'Dataset':  'data/dataset.txt'}   # creating a dictionary
# with key as source of data. And the value part of dictionary is
# the address of the dataset.
categorized_dataset = categorize_dataset(dataset_dictionary)  # returns data organized in class, text and
# source of dataset.
sentences_train, sentences_test, y_train, y_test = splitting_data(categorized_dataset)
# Function splitting_data, takes this organized data and returns splitted data test(0.25) and train(0.75)
# and respective labels(class).
X_train, X_test = vectorize_sentences(sentences_train, sentences_test)  # the function vectorizes the sentences by
# creating feature vector of the count of the words.


input_dim = X_train.shape[1]  # input_dim takes the shape of the X_train(in this case it is 4814)
model = DNN_basic(input_dim)  # DNN_basic is a sequential two layer Neural network, with relu activation in first layer
# and sigmoid in second. It calculates loss by binary cross entropy, uses adam optimizer and returns accuracy metrics.
training = model.fit(X_train, y_train,
                    epochs=5,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    batch_size=10)  # model.fit(),Trains the model for a fixed number of epochs (iterations on dataset).
loss_train, accuracy_train = model.evaluate(X_train, y_train, verbose=False)  # Returns the loss value & metrics values
# for the model in train mode.
print("Training Accuracy: {:.4f}".format(accuracy_train))  # Prints the accuracy of the model on train dataset.
loss_test, accuracy_test = model.evaluate(X_test, y_test, verbose=False)  # Returns the loss value & metrics values for
# the model in test mode.
print("Testing Accuracy:  {:.4f}".format(accuracy_test))  # Prints the accuracy of the model on test dataset.
plot_history(training)  # this function displays the accuracy and loss during training using matplotlib.


