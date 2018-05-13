import load_data_script as ld
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, BatchNormalization, Activation, Bidirectional
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import sgd, adam
import glob
from models.attention import AttentionWithContext
import random

random.seed(42)

# Properties of the data
max_length=530
data_dim=4
training_rate=0.5
validation_rate=0.1
num_hidden = 10
epochs=100
detect_pca=False

# Load the dataset
(training_data, validation_data, testing_data) = ld.read_dataset(training_rate=training_rate, validation_rate=validation_rate,
                                                                 max_length=max_length, data_dim=data_dim, detect_pca=detect_pca)

# Instantiate LSTM model for eye movement analysis
model = Sequential()
model.add(Bidirectional(LSTM(num_hidden, return_sequences=True), input_shape=(max_length, data_dim)))
model.add(BatchNormalization())
# model.add(LSTM(32,return_sequences=True))
# model.add(BatchNormalization())
model.add(AttentionWithContext())
model.add(Dropout(0.5))

filepath_model = "weights-eye-{val_acc:.2f}-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath_model, monitor='val_acc',
                             verbose=1, save_best_only=True, mode='max')


if(detect_pca):
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(training_data._normalised_input_data, to_categorical(training_data._output_data),
              validation_split=0.1, batch_size=16, epochs=epochs, callbacks=[checkpoint])
else:
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(training_data._normalised_input_data, training_data._output_data,
              validation_data =[testing_data._normalised_input_data, testing_data._output_data],
              batch_size=16, epochs=epochs, callbacks=[checkpoint])


best_validation = load_model(sorted(glob.glob('./weights-eye*'))[-1], custom_objects={'AttentionWithContext':AttentionWithContext})
score = model.evaluate(testing_data._normalised_input_data, testing_data._output_data, batch_size=16)

print(score[1])

# # Train and evaluate the model
# keep_prob = 0.5
# batch_size = 15
# normalise = True
# num_epochs = 500
# print_flag = True
#
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#
#     num_batches = int(num_training_files / batch_size) + int(bool(num_training_files % batch_size))
#
#     feed_dict = ld.testing_data.feed_dict(model, 1.0, normalise)
#     diag_pred = isinstance(model, DiagnosisPrediction)
#     incorrect = sess.run(model.error,feed_dict)
#     best_test_error = utils.mean_error(incorrect, diag_pred, ld.testing_data.sequence_lengths)
#
#
#     train_errors, test_errors = train_model(ld.training_data, ld.testing_data, model, sess, keep_prob, batch_size, normalise, num_epochs, print_flag)
#     #train_errors, test_errors = train_lstm_diag(ld.training_data, ld.testing_data, model, sess, keep_prob, batch_size, normalise, num_epochs, print_flag)
#     best_test_error = min(best_test_error, min(test_errors))
#
#     plt.plot(train_errors)
#     plt.plot(test_errors)
#     plt.show()
#     print('Best test error was {:4.2f}%'.format(100*best_test_error))
#
#     vals, preds = utils.vals_preds(model, ld.testing_data, sess, normalise)
#     #vals = ld.testing_data.diagnosis_labels
#     #max_val = vals.shape[1]
#     #feed_dict = ld.testing_data.feed_dict(model, 1.0, normalise)
#     #feed_dict[model.diag_labels] = ld.testing_data.diagnosis_labels
#     #preds = sess.run(model.diag_prediction, feed_dict)
#     #preds = np.concatenate([np.reshape(preds == i, [preds.size, 1]).astype(int) for i in range(max_val)], axis = 1)
#     cm = utils.calc_cm(vals, preds)
#     utils.print_cm(cm)
#     print("Matthew's correlation coeffecient: {:4.3f}".format(utils.mcc(vals, preds)))
#     print("Youden's J statistic: {:4.3f}".format(utils.youdens_j(cm)))
#     print("F1 score: {:4.3f}".format(utils.f1(cm)))

