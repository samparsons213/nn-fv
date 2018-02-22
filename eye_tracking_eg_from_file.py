import tensorflow as tf
sess = tf.Session()

filename_queue = tf.train.string_input_producer(["diff_diff_data_and_labels.csv"], num_epochs=1)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.0], [0.0], [0]]
col1, col2, col3 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack([col1, col2])

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

eof = False
i = 0
#example, label = sess.run([features, col3])
#indata = [example]
#outdata = [label]
#i += 1
indata = []
outdata = []
while not eof:
	try:
		i += 1
		example, label = sess.run([features, col3])
		indata.append(example)
		outdata.append([1-label, label])
		#print('%d' % i)
	except tf.errors.OutOfRangeError:
		eof = True
	finally:
		coord.request_stop()

coord.join(threads)

indata = [indata]
outdata = [outdata]
print len(indata[0])
data = tf.placeholder(tf.float32, [None, 2606, 2])
targets = tf.placeholder(tf.float32, [None, 2606, 2])

num_hidden = 24
cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)

val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
#print val.get_shape(), cell.output_size

weight = tf.Variable(tf.truncated_normal([num_hidden, int(targets.get_shape()[2])]))
bias = tf.Variable(tf.constant(0.1, shape=[targets.get_shape()[2]]))
pred_mul = tf.einsum('ijk,kl->ijl', val, weight)
#print pred_mul.get_shape(), weight.get_shape()

#prediction = tf.nn.softmax(pred_mul + bias)
#print prediction.get_shape()

#cross_entropy = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
pred_logits = pred_mul + bias
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=pred_logits))

optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

#mistakes = tf.not_equal(tf.argmax(targets, 2), tf.argmax(prediction, 2))
mistakes = tf.not_equal(tf.argmax(targets, 2), tf.argmax(pred_logits, 2))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

batch_size = 1
no_of_batches = int(len([indata])/batch_size)

epoch = 10
init_op = tf.global_variables_initializer()
sess.run(init_op)
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = indata[ptr:ptr+batch_size], outdata[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, targets: out})
    print "Epoch - ",str(i)

#print sess.run(mistakes,{data: indata, targets: outdata}).shape
incorrect = sess.run(error,{data: indata, targets: outdata})
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()


