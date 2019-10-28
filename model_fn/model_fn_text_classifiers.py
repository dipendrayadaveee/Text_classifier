from keras.models import Sequential
from keras import layers


def CNN_text_classifier(embedding_dim, vocab_size, maxlen):
    '''It is a sequential five layer neural network where first layer is an Embedding layer, second layer is
    1-dimensional convolution layer using relu activation function, third layer is maxpooling layer followed by
    a dense layer using relu activation and finally the last dense layer using sigmoid activation function'''

    model = Sequential()  # Sequential is a Keras API which allows us to create models layer-by-layer piecewise.
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))  # It is the first embedding layer of
    # the model which turns positive integers(indexes) into dense vectors of fixed size. input_dim = vocab_size(4933 in
    # this case). Output_dimension = embedding_dim(100 in this case), is the dimension of dense embedding.
    # input_length = maxlen(100 in this case), is the length of input sequence.
    model.add(layers.Conv1D(128, 5, activation='relu'))  # adding 1-dimensional convolution layer as second layer with
    # filters = 128, output filters in the convolution(dimensionality of the output space). kernel_size = 5, which
    # signifies the length of the 1D convolution window.
    model.add(layers.GlobalMaxPooling1D())  # using maxpooling layer to reduce the spatial size of the representation.
    model.add(layers.Dense(10, activation='relu'))  # Dense is used to create densely-connected
    # Neural Network layers. we have taken units:10, which means output array shape will be(*, 10). input_dim is the
    # dimension of input fed to the layer and activation:relu, basically uses rectified linear unit to convert input
    # signal into output signal at a A-NN node. It has been seen that it provides better convergence and also it
    # also rectifies vanishing gradient problem.
    model.add(layers.Dense(1, activation='sigmoid'))  # it is the last layer whose output shape array shape
    # will be(*, 1). The activation function used is Sigmoid whose range is between 0 and 1.
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    #  compile is a method of sequential class which configures the model for training. We are using binary_crossentropy
    #  loss function to calculate loss, and adam optimizer. we want to evaluate just the accuracy metric.
    model.summary()  # prints a summary representation of the model.

    return model


def DNN_basic(input_dim):
    '''It is a sequential two layer Neural network, with relu activation in first layer and sigmoid in second.
    It calculates loss by binary cross entropy, uses adam optimizer and returns accuracy metrics.'''

    model = Sequential()  # Sequential is a Keras API which allows us to create models layer-by-layer piecewise.
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))  # Dense is used to create densely-connected
    # Neural Network layers. we have taken units:10, which means output array shape will be(*, 10). input_dim is the
    # dimension of input fed to the layer and activation:relu, basically uses rectified linear unit to convert input
    # signal into output signal at a A-NN node. It has been seen that it provides better convergence and also it
    # also rectifies vanishing gradient problem.
    model.add(layers.Dense(1, activation='sigmoid'))  # it is the last layer whose output shape array shape
    # will be(*, 1). The activation function used is Sigmoid whose range is between 0 and 1.
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    #  compile is a method of sequential class which configures the model for training. We are using binary_crossentropy
    #  loss function to calculate loss, and adam optimizer. we want to evaluate just the accuracy metric.
    model.summary()  # prints a summary representation of the model.
    #  if you run the model.summary() for this model, you get: output shape for first layer as (None, 10) and
    #  that of second (None, 1). Params for first layer=48150(number_parameters= output_size * (input_size + 1);
    #  input_size = input_dim = 4814) and for second layer is 11.
    return model


def DNN_MaxPooling1D(embedding_dim, vocab_size, maxlen):
    '''It is a sequential four layer neural network where first layer is an Embedding layer, second layer
    is maxpooling layer followed by a dense layer using relu activation and finally the last dense layer using
    sigmoid activation function.'''

    model = Sequential()  # Sequential is a Keras API which allows us to create models layer-by-layer piecewise.
    model.add(layers.Embedding(input_dim=vocab_size,
                               output_dim=embedding_dim,
                               input_length=maxlen))  # It is the first, embedding layer of the model which turns
    # positive integers(indexes) into dense vectors of fixed size. input_dim = vocab_size(4933 in
    # this case). Output_dimension = embedding_dim(50 in this case), is the dimension of dense embedding.
    # input_length = maxlen(100 in this case), is the length of input sequence.
    model.add(layers.GlobalMaxPool1D())  # using maxpooling layer to reduce the spatial size of the representation.
    model.add(layers.Dense(10, activation='relu'))  # Dense is used to create densely-connected
    # Neural Network layers. we have taken units:10, which means output array shape will be(*, 10). input_dim is the
    # dimension of input fed to the layer and activation:relu, basically uses rectified linear unit to convert input
    # signal into output signal at a A-NN node. It has been seen that it provides better convergence and also it
    # also rectifies vanishing gradient problem.
    model.add(layers.Dense(1, activation='sigmoid'))  # it is the last layer whose output shape array shape
    # will be(*, 1). The activation function used is Sigmoid whose range is between 0 and 1.
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    #  compile is a method of sequential class which configures the model for training. We are using binary_crossentropy
    #  loss function to calculate loss, and adam optimizer. we want to evaluate just the accuracy metric.
    model.summary()  # prints a summary representation of the model.
    return model


def DNN_PretrainedWordEmbeddingGlove(embedding_dim, vocab_size, maxlen, embedding_matrix):
    '''It is a sequential four layer neural network where first layer is an Embedding layer which gets weights from
     pretrained GloVe word-embeddings, second layer is maxpooling layer followed by a dense layer using relu activation
     and finally the last dense layer using sigmoid activation function.'''

    model = Sequential()  # Sequential is a Keras API which allows us to create models layer-by-layer piecewise.
    # TODO: vary trainable between True and False to additionally train the word embeddings.
    model.add(layers.Embedding(vocab_size, embedding_dim,
                               weights=[embedding_matrix],
                               input_length=maxlen,
                               trainable=True))# It is the first, embedding layer of the model which uses pretrained
    # GloVe word embeddings. We can choose to train the word embeddings additionally by setting trainable=True.
    # input_length = maxlen(100 in this case), is the length of input sequence.
    model.add(layers.GlobalMaxPool1D())  # using maxpooling layer to reduce the spatial size of the representation.
    model.add(layers.Dense(10, activation='relu'))  # Dense is used to create densely-connected
    # Neural Network layers. we have taken units:10, which means output array shape will be(*, 10). input_dim is the
    # dimension of input fed to the layer and activation:relu, basically uses rectified linear unit to convert input
    # signal into output signal at a A-NN node. It has been seen that it provides better convergence and also it
    # also rectifies vanishing gradient problem.
    model.add(layers.Dense(1, activation='sigmoid'))  # it is the last layer whose output shape array shape
    # will be(*, 1). The activation function used is Sigmoid whose range is between 0 and 1.
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    #  compile is a method of sequential class which configures the model for training. We are using binary_crossentropy
    #  loss function to calculate loss, and adam optimizer. we want to evaluate just the accuracy metric.
    model.summary()  # prints a summary representation of the model.
    return model


def DNN_WordEmbedding(embedding_dim, vocab_size, maxlen):
    '''It is a sequential four layer neural network where first layer is an Embedding layer, second layer
    is used to flatten the output of first layer followed by a dense layer using relu activation and
    finally the last dense layer using sigmoid activation function.'''

    model = Sequential()  # Sequential is a Keras API which allows us to create models layer-by-layer piecewise.
    model.add(layers.Embedding(input_dim=vocab_size,
                               output_dim=embedding_dim,
                               input_length=maxlen))  # It is the first, embedding layer of the model which turns
    # positive integers(indexes) into dense vectors of fixed size. input_dim = vocab_size(4933 in
    # this case). Output_dimension = embedding_dim(50 in this case), is the dimension of dense embedding.
    # input_length = maxlen(100 in this case), is the length of input sequence.
    model.add(layers.Flatten())  # removes all of the dimensions except one, i.e. in this case output shape from last
    # layer (None, 100, 50) is flattened to (None, 5000).
    model.add(layers.Dense(10, activation='relu'))  # Dense is used to create densely-connected
    # Neural Network layers. we have taken units:10, which means output array shape will be(*, 10). input_dim is the
    # dimension of input fed to the layer and activation:relu, basically uses rectified linear unit to convert input
    # signal into output signal at a A-NN node. It has been seen that it provides better convergence and also it
    # also rectifies vanishing gradient problem.
    model.add(layers.Dense(1, activation='sigmoid'))  # it is the last layer whose output shape array shape
    # will be(*, 1). The activation function used is Sigmoid whose range is between 0 and 1.
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    #  compile is a method of sequential class which configures the model for training. We are using binary_crossentropy
    #  loss function to calculate loss, and adam optimizer. we want to evaluate just the accuracy metric.
    model.summary()  # prints a summary representation of the model.
    return model
