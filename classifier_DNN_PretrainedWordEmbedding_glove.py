from input_fn.input_fn_text_classifier import categorize_dataset
from input_fn.input_fn_text_classifier import splitting_data
from util.util_text_classifier import tokenizer
from util.util_text_classifier import pad_sequence
from model_fn.model_fn_text_classifiers import DNN_PretrainedWordEmbeddingGlove
from util.util_text_classifier import plot_history

import numpy as np

#Todo: To run this code it is needed to download Glove embeddings in 'data/glove_word_embeddings' from
# http://nlp.stanford.edu/data/glove.6B.zip

dataset_dictionary ={'Dataset':  'data/dataset.txt'}   # creating a dictionary
# with key as source of data. And the value part of dictionary is
# the address of the dataset.
categorized_dataset = categorize_dataset(dataset_dictionary)  # returns data organized in class, text and
# source of dataset
sentences_train, sentences_test, y_train, y_test = splitting_data(categorized_dataset)
# the function splitting_data, takes this organized data and returns splitted data test(0.25) and train(0.75)
# and respective labels(class).
X_train, X_test, vocab_size, tokenizer_word = tokenizer(sentences_train, sentences_test)  # Function tokenizer,
# vectorizes the texts by turning each text into sequence of integers.
X_train, X_test, maxlen = pad_sequence(X_train, X_test)  # Function pad_sequence, takes the vectorized sentences
# and returns numpy arrays for X_train, X_test and also maximum length of the sequence.


def create_embedding_matrix(embedding_address, word_index, embedding_dim):
    '''this function skips those words from the pretrained word embedding which is not present in our data vocabulary,
    as we don't need all the words from embeddings and can concentrate on only words present in our vocabulary. It
    returns embedding_matrix for words in our vocabulary'''
    vocab_size = len(word_index) + 1  # Adding 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(embedding_address) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


embedding_dim = 50  # using embedding dimension of 50.
embedding_matrix = create_embedding_matrix('data/glove_word_embeddings/glove.6B.50d.txt',tokenizer_word.word_index,
                                           embedding_dim)

model = DNN_PretrainedWordEmbeddingGlove(embedding_dim, vocab_size, maxlen, embedding_matrix)
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
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print('Vocabulary covered of the dataset by Pretrained GloVe word embedding:',(nonzero_elements / vocab_size))
plot_history(training)  # this function displays the accuracy and loss during training using matplotlib.
