import tensorflow as tf
sess = tf.Session()

filename_queue = tf.train.string_input_producer(["diff_diff_data.csv"], num_epochs=1)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.0], [0.0]]
col1, col2 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack([col1, col2])

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

eof = False
i = 0
example = [sess.run(features)]
i += 1
while not eof:
	try:
		i += 1
		example.append(sess.run(features))#example = sess.run(features)
		print('%d' % i)

	except tf.errors.OutOfRangeError:
		eof = True
	finally:
		coord.request_stop()
	

for p in example: print(p)

coord.join(threads)
sess.close()

