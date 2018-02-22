import numpy as np
from DiagnosisModel import DiagnosisPrediction
from utils import mean_error

def train_model(training_data, testing_data, model, sess, keep_prob, batch_size, normalise, num_epochs, print_flag):
    train_errors = np.zeros(num_epochs, dtype=float)
    test_errors = np.zeros(num_epochs, dtype=float)
    diag_pred = isinstance(model, DiagnosisPrediction)
    num_training_files = training_data.input_data().shape[0]
    num_batches = int(num_training_files / batch_size) + int(bool(num_training_files % batch_size))
    sl_train = training_data.sequence_lengths
    sl_test = testing_data.sequence_lengths
    for epoch in range(num_epochs):
        if print_flag:
            print('Training epoch {:2d}'.format(epoch + 1))
        ptr = 0
        ''' shuffled_order = np.array(range(num_training_files))
        np.random.shuffle(shuffled_order) '''
        training_data.random_shuffle()
        for batch in range(num_batches):
            this_batch_size = min(num_training_files-ptr, batch_size)
            #this_batch_idx = shuffled_order[range(ptr, ptr+this_batch_size)]
            this_batch_idx = range(ptr, ptr+this_batch_size)
            feed_dict = training_data.feed_dict(model, keep_prob, normalise, this_batch_idx)
            ptr += this_batch_size
            sess.run(model.minimize, feed_dict)
            if print_flag:
                feed_dict = training_data.feed_dict(model, 1.0, normalise, this_batch_idx)
                incorrect = sess.run(model.error, feed_dict)
                incorrect = mean_error(incorrect, diag_pred, sl_train[this_batch_idx])
                print('     Training data batch {:2d} error {:3.1f}%'.format(batch + 1, 100 * incorrect))
        feed_dict = training_data.feed_dict(model, 1.0, normalise)
        incorrect = sess.run(model.error, feed_dict)
        train_errors[epoch] = mean_error(incorrect, diag_pred, sl_train)
        feed_dict = testing_data.feed_dict(model, 1.0, normalise)
        incorrect = sess.run(model.error, feed_dict)
        test_errors[epoch] = mean_error(incorrect, diag_pred, sl_test)
            
    return train_errors, test_errors

def train_lstm_diag(training_data, testing_data, model, sess, keep_prob, batch_size, normalise, num_epochs, print_flag):
    train_errors = np.zeros(num_epochs, dtype=float)
    test_errors = np.zeros(num_epochs, dtype=float)
    #diag_pred = isinstance(model, DiagnosisPrediction)
    diag_pred = True
    num_training_files = training_data.input_data().shape[0]
    num_batches = int(num_training_files / batch_size) + int(bool(num_training_files % batch_size))
    print num_training_files
    print batch_size
    print num_training_files / batch_size
    print num_batches
    sl_train = training_data.sequence_lengths
    sl_test = testing_data.sequence_lengths
    for epoch in range(num_epochs):
        if print_flag:
            print('Training epoch {:2d}'.format(epoch + 1))
        ptr = 0
        ''' shuffled_order = np.array(range(num_training_files))
        np.random.shuffle(shuffled_order) '''
        training_data.random_shuffle()
        for batch in range(num_batches):
            this_batch_size = min(num_training_files-ptr, batch_size)
            #this_batch_idx = shuffled_order[range(ptr, ptr+this_batch_size)]
            this_batch_idx = range(ptr, ptr+this_batch_size)
            feed_dict = training_data.feed_dict(model, keep_prob, normalise, this_batch_idx)
            feed_dict[model.diag_labels] = training_data.diagnosis_labels[this_batch_idx]
            ptr += this_batch_size
            sess.run(model.diag_minimize, feed_dict)
            if print_flag:
                feed_dict = training_data.feed_dict(model, 1.0, normalise, this_batch_idx)
                feed_dict[model.diag_labels] = training_data.diagnosis_labels[this_batch_idx]
                incorrect = sess.run(model.diag_error, feed_dict)
                incorrect = mean_error(incorrect, diag_pred, sl_train[this_batch_idx])
                print('     Training data batch {:2d} error {:3.1f}%'.format(batch + 1, 100 * incorrect))
        feed_dict = training_data.feed_dict(model, 1.0, normalise)
        feed_dict[model.diag_labels] = training_data.diagnosis_labels
        incorrect = sess.run(model.diag_error, feed_dict)
        train_errors[epoch] = mean_error(incorrect, diag_pred, sl_train)
        feed_dict = testing_data.feed_dict(model, 1.0, normalise)
        feed_dict[model.diag_labels] = testing_data.diagnosis_labels
        incorrect = sess.run(model.diag_error, feed_dict)
        test_errors[epoch] = mean_error(incorrect, diag_pred, sl_test)

    return train_errors, test_errors