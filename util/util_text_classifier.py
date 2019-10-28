from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt


def vectorize_sentences(sentences_train, sentences_test):
    '''this function vectorizes the sentences. i.e. it takes the words of each sentence and creates a vocabulary of all
    unique words in sentences which is used to create feature vector of the count of the words.(Bag-of-Word Model).
    it returns vectorized train and test sentences.'''
    vectorizer = CountVectorizer()  # used to convert a collection of text docs to a matrix of token counts.
    vectorizer.fit(sentences_train)  # Learns a vocab dictionary of all tokens in sentence_train.
    X_train = vectorizer.transform(sentences_train)  # Transforms sentences_train into document-term matrix.
    X_test = vectorizer.transform(sentences_test)  # Transforms sentences_test into document-term matrix.
    return X_train, X_test


def tokenizer(sentences_train, sentences_test):
    '''this function vectorizes the texts by turning each text into sequence of integers. It returns the vectorized
     sentences of train and test, the vocab size and Tokenizer class'''
    tokenizer_word = Tokenizer(num_words=5000)  # maximum num of words(5000) to keep based on word frequency.
    tokenizer_word.fit_on_texts(sentences_train)  # fits the tokenizer on the sentence_train
    X_train = tokenizer_word.texts_to_sequences(sentences_train)  # converts the text into sequence of numbers
    X_test = tokenizer_word.texts_to_sequences(sentences_test)  # converts the text into sequence of numbers
    vocab_size = len(tokenizer_word.word_index) + 1  # Adding 1 because of reserved 0 index
    return X_train, X_test, vocab_size, tokenizer_word


def pad_sequence(X_train, X_test):
    '''this function takes the vectorized sentences and returns numpy arrays for X_train, X_test
    of shape(len(sequences), maxlen) and maxlen'''
    maxlen = 100  # maximum length of all the sequences, set to 100 in this case
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)  # removes the values from the X_train sequences
    # larger than "maxlen" from the end of the sequences.
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)  # removes the values from the X_test sequences
    # larger than "maxlen" from the end of the sequences.
    return X_train, X_test, maxlen


def plot_history(training):
    '''this function displays the accuracy and loss during training using matplotlib.'''
    plt.style.use('ggplot')  # using ggplot style
    acc = training.history['accuracy']  # history is an Object returned by fit() method and it is a dictionary
    # recording training loss values and metrics values at successive epochs, as well as validation loss values and
    # validation metrics values.
    val_acc = training.history['val_accuracy']
    loss = training.history['loss']
    val_loss = training.history['val_loss']
    x = range(1, len(acc) + 1)  # setting range of the x-axis.

    plt.figure(figsize=(12, 5))  # setting the figure size of the plot display.
    plt.subplot(1, 2, 1)  # creates subplot with number of row = 1,number of column = 2,
    # and index of the plot to be created = 1.
    plt.plot(x, acc, 'b', label='Training acc')  # plots Training accuracy labelled as Training acc
    plt.plot(x, val_acc, 'r', label='Validation acc')  # plots Validation accuracy labelled as Validation acc
    plt.title('Training and validation accuracy')  # sets title of the subplot
    plt.legend()  # places legend inside the plot
    plt.subplot(1, 2, 2)  # creates subplot with number of row = 1,number of column = 2,
    # and index of the plot to be created = 2.
    plt.plot(x, loss, 'b', label='Training loss')  # plots the Training loss.
    plt.plot(x, val_loss, 'r', label='Validation loss')  # plots the Validation loss.
    plt.title('Training and validation loss')  # sets title of the subplot
    plt.legend()   # places legend inside the plot
    plt.show()  # Displays the figure
