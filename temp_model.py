#coding=utf-8

try:
    import load_data_script as ld
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.layers import Dense, Dropout
except:
    pass

try:
    from keras.layers import BatchNormalization, Conv1D
except:
    pass

try:
    from keras.utils import to_categorical
except:
    pass

try:
    from keras.optimizers import sgd
except:
    pass

try:
    from keras import regularizers
except:
    pass

try:
    from models.attention import AttentionWithContext
except:
    pass

try:
    import random
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    from hyperas.distributions import uniform, choice, quniform
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from hyperas import optim
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperas.distributions import conditional

# Properties of the data
max_length = 530
data_dim = 4
training_rate = 0.5
validation_rate = 0.1
detect_pca = False

# Load the dataset
(training_data, validation_data, testing_data) = ld.read_dataset(training_rate=training_rate,
                                                                 validation_rate=validation_rate,
                                                                 max_length=max_length, data_dim=data_dim,
                                                                 detect_pca=detect_pca)
x_train = training_data._normalised_input_data
y_train = training_data._output_data
x_val = validation_data._normalised_input_data
y_val = validation_data._output_data



def keras_fmin_fnct(space):


    epochs = 80
    max_length = 530
    data_dim = 4

    # Instantiate model for eye movement analysis
    model = Sequential()
    model.add(Conv1D(space['Conv1D'], space['Conv1D_1'],
                     activation=space['activation'],
                     input_shape=(max_length, data_dim), kernel_regularizer=regularizers.l2(space['l2'])))
    model.add(BatchNormalization())
    model.add(Conv1D(space['Conv1D_2'], space['Conv1D_3'],
                     activation=space['activation_1'], kernel_regularizer=regularizers.l2(space['l2_1'])))
    model.add(BatchNormalization())
    model.add(Conv1D(space['Conv1D_4'], space['Conv1D_5'],
                     activation=space['activation_2'], kernel_regularizer=regularizers.l2(space['l2_2'])))
    model.add(BatchNormalization())
    model.add(AttentionWithContext(W_regularizer=regularizers.l2(space['l2_3']),
                                   u_regularizer=regularizers.l2(space['l2_4']),
                                   b_regularizer=regularizers.l2(space['l2_5'])))
    model.add(Dropout(space['Dropout']))

    best_score = 0

    optimizer = sgd(lr=space['lr'], decay=1e-4, momentum=space['momentum'])
    if(detect_pca):
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        for epoch in range(0, epochs):
            print('Epoch:', epoch)
            model.fit(x_train, to_categorical(y_train),
                      validation_split=0.1, batch_size=16)
            score, acc = model.evaluate(x_val, y_val, verbose=0)
            if(acc > best_score):
                print('Validation accuracy:', acc)
                best_score = acc

    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        for epoch in range(0, epochs):
            print('Epoch:', epoch)
            model.fit(x_train, y_train,
                      validation_data=[x_val, y_val],
                      batch_size=16)
            score, acc = model.evaluate(x_val, y_val, verbose=0)
            if(acc > best_score):
                print('Validation accuracy:', acc)
                best_score = acc

    return {'loss': -best_score, 'status': STATUS_OK, 'model': model}

def get_space():
    return {
        'Conv1D': hp.choice('Conv1D', [2, 4, 8, 16, 32, 48, 64, 78, 100]),
        'Conv1D_1': hp.choice('Conv1D_1', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]),
        'activation': hp.choice('activation', ['relu', 'sigmoid']),
        'l2': hp.uniform('l2', 0.01, 1),
        'Conv1D_2': hp.choice('Conv1D_2', [2, 4, 8, 16, 32, 48, 64, 78, 100]),
        'Conv1D_3': hp.choice('Conv1D_3', [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]),
        'activation_1': hp.choice('activation_1', ['relu', 'sigmoid']),
        'l2_1': hp.uniform('l2_1', 0.01, 1),
        'Conv1D_4': hp.choice('Conv1D_4', [2, 4, 8, 16, 32, 48, 64, 78, 100]),
        'Conv1D_5': hp.choice('Conv1D_5', [3, 4, 5, 6, 7, 8, 9, 10, 15]),
        'activation_2': hp.choice('activation_2', ['relu', 'sigmoid']),
        'l2_2': hp.uniform('l2_2', 0.01, 1),
        'l2_3': hp.uniform('l2_3', 0.01, 1),
        'l2_4': hp.uniform('l2_4', 0.01, 1),
        'l2_5': hp.uniform('l2_5', 0.01, 1),
        'Dropout': hp.uniform('Dropout', 0, 1),
        'lr': hp.uniform('lr', 0.003, 0.01),
        'momentum': hp.uniform('momentum', 0.65, 0.99),
    }
