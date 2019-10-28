from input_fn.input_fn_text_classifier import categorize_dataset
from input_fn.input_fn_text_classifier import splitting_data
from util.util_text_classifier import tokenizer
from util.util_text_classifier import pad_sequence
from model_fn.model_fn_text_classifiers import CNN_text_classifier
from util.util_text_classifier import plot_history

dataset_dictionary ={'Market Logic Software':  'data/Market_Logic_Software_dataset.txt'}  # creating a dictionary
# with key as source of data (in this case, it is Market Logic Software). And the value part of dictionary is
# the address of the dataset.
categorized_dataset = categorize_dataset(dataset_dictionary)  # returns data organized in class, text and
# source of dataset
sentences_train, sentences_test, y_train, y_test = splitting_data(categorized_dataset)
# the function splitting_data, takes this organized data and returns splitted data test(0.25) and train(0.75)
# and respective labels(class).
X_train, X_test, vocab_size, _ = tokenizer(sentences_train, sentences_test)  # Function tokenizer, vectorizes the texts
# by turning each text into sequence of integers.
X_train, X_test, maxlen = pad_sequence(X_train, X_test)  # Function pad_sequence, takes the vectorized sentences
# and returns numpy arrays for X_train, X_test and also maximum length of the sequence.


embedding_dim = 100  # using embedding dimension of 100.
model = CNN_text_classifier(embedding_dim, vocab_size, maxlen)  # CNN_text_classifier is a sequential five layer
# neural network where first layer is an Embedding layer, second layer is 1-dimensional convolution layer using
# relu activation function, third layer is maxpooling layer followed by a dense layer using relu activation
# and finally the last dense layer using sigmoid activation function
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

