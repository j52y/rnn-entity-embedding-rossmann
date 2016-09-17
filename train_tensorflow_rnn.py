import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

from model import *


def fetch_data(x, y, t_size, step, all=False):
    xd_1, xd_2 = x.shape

    if all:
        indices = np.arange(t_size)
    else:
        indices = np.random.randint(0, xd_1 - step, t_size)

    diff_stores = [i for i, j in enumerate(indices) if x[j, 1] != x[j + step - 1, 1]]
    # print('ds %s %i' % (arr, len(arr)))

    indices = np.delete(indices, diff_stores)
    t_size = len(indices)

    data_x = np.zeros((t_size * step, xd_2), dtype='int8')
    for i in range(step):
        data_x[i * t_size:(i + 1) * t_size] = x[indices + i]

    print(t_size, indices.shape, y.shape)
    return data_x, y[indices + step - 1]


np.random.seed(123)

r_range = 0.05  # random range

y_label = tf.placeholder("float", (None, 1), name='y_label')

n_store, v_store = 1115, 10  # the number of stores, vector dim of stores
n_dow, v_dow = 7, 6
n_promo, v_promo = 1, 1
n_year, v_year = 3, 2
n_month, v_month = 12, 6
n_day, v_day = 31, 10
n_germanstate, v_germanstate = 12, 6

x_store = tf.placeholder("float", (None, n_store), name='x_store')
x_dow = tf.placeholder("float", (None, n_dow), name='x_dow')
x_promo = tf.placeholder("float", (None, n_promo), name='x_promo')
x_year = tf.placeholder("float", (None, n_year), name='x_year')
x_month = tf.placeholder("float", (None, n_month), name='x_month')
x_day = tf.placeholder("float", (None, n_day), name='x_day')
x_germanstate = tf.placeholder("float", (None, n_germanstate), name='x_germanstate')

emb_store = tf.matmul(x_store, tf.Variable(tf.random_uniform((n_store, v_store), -r_range, r_range)))
emb_dow = tf.matmul(x_dow, tf.Variable(tf.random_uniform((n_dow, v_dow), -r_range, r_range)))
emb_promo = tf.matmul(x_promo, tf.Variable(tf.random_uniform((n_promo, v_promo), -r_range, r_range)))
emb_year = tf.matmul(x_year, tf.Variable(tf.random_uniform((n_year, v_year), -r_range, r_range)))
emb_month = tf.matmul(x_month, tf.Variable(tf.random_uniform((n_month, v_month), -r_range, r_range)))
emb_day = tf.matmul(x_day, tf.Variable(tf.random_uniform((n_day, v_day), -r_range, r_range)))
emb_germanstate = tf.matmul(x_germanstate,
                            tf.Variable(tf.random_uniform((n_germanstate, v_germanstate), -r_range, r_range)))

emb_all = tf.concat(1, [emb_store, emb_dow, emb_promo, emb_year, emb_month, emb_day, emb_germanstate])
v_all = v_store + v_dow + v_promo + v_year + v_month + v_day + v_germanstate

n_steps = 7
n_lstm_out = 1000

x = tf.split(0, n_steps, emb_all)

lstm_cell = rnn_cell.BasicLSTMCell(n_lstm_out, forget_bias=1.0, state_is_tuple=True)
outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

w1 = tf.Variable(tf.random_uniform((n_lstm_out, 500), -r_range, r_range))
b1 = tf.Variable(tf.random_uniform((500,), -r_range, r_range))
h1 = tf.nn.relu(tf.matmul(outputs[-1], w1) + b1)

w2 = tf.Variable(tf.random_uniform((500, 1), -r_range, r_range))
b2 = tf.Variable(tf.random_uniform((1,), -r_range, r_range))
t1 = tf.matmul(h1, w2) + b2
pred = tf.nn.sigmoid(t1)

cost = tf.reduce_sum(tf.abs(tf.sub(pred, y_label)))
learning_rate = 0.0001  # 0.003일 때 발산
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

train_ratio = 0.9
iterations = 1000

sorted_data_file = 'sorted_feature_train_data.pickle'
f = open(sorted_data_file, 'rb')
(X, y) = pickle.load(f)

num_records = len(X)

train_size = int(train_ratio * num_records)

X_train = X[:train_size]
X_val = X[train_size:]
y_train = y[:train_size]
y_val = y[train_size:]

max_log_y = max(np.max(np.log(y_train)), np.max(np.log(y_val)))

sess = tf.InteractiveSession()
saver = tf.train.Saver()

file_model = "./tmp/model_rnn.ckpt"

if os.path.isfile(file_model):
    saver.restore(sess, file_model)
else:
    sess.run(tf.initialize_all_variables())


def val_for_fit(val):
    val = np.log(val) / max_log_y
    return val


def val_for_pred(val):
    return numpy.exp(val * max_log_y)


def one_hot_multiple(size, targets):
    return np.eye(size)[targets]


def feed_dict(train_x, train_y):
    x_store_v = train_x[:, 1]
    x_dow_v = train_x[:, 2]
    x_promo_v = train_x[:, 3]
    x_year_v = train_x[:, 4]
    x_month_v = train_x[:, 5]
    x_day_v = train_x[:, 6]
    x_germanstate_v = train_x[:, 7]
    y_v = val_for_fit(train_y)[:, None]
    return {x_store: one_hot_multiple(1115, x_store_v),
            x_dow: one_hot_multiple(7, x_dow_v),
            x_promo: x_promo_v[:, None],
            x_year: one_hot_multiple(3, x_year_v),
            x_month: one_hot_multiple(12, x_month_v),
            x_day: one_hot_multiple(31, x_day_v),
            x_germanstate: one_hot_multiple(12, x_germanstate_v),
            y_label: y_v}


def feed_dict_x(train_x):
    x_store_v = train_x[:, 1]
    x_dow_v = train_x[:, 2]
    x_promo_v = train_x[:, 3]
    x_year_v = train_x[:, 4]
    x_month_v = train_x[:, 5]
    x_day_v = train_x[:, 6]
    x_germanstate_v = train_x[:, 7]
    return {x_store: one_hot_multiple(1115, x_store_v),
            x_dow: one_hot_multiple(7, x_dow_v),
            x_promo: x_promo_v[:, None],
            x_year: one_hot_multiple(3, x_year_v),
            x_month: one_hot_multiple(12, x_month_v),
            x_day: one_hot_multiple(31, x_day_v),
            x_germanstate: one_hot_multiple(12, x_germanstate_v)}


def evaluate_models(x, y):
    assert (min(y) > 0)
    guessed_sales = val_for_pred(pred.eval(feed_dict=feed_dict_x(x)))
    mean_sales = guessed_sales.mean(axis=0)
    relative_err = np.absolute((y - mean_sales) / y)
    result = np.sum(relative_err) / len(y)
    return result


i = 0
while i <= iterations:
    batch_x, batch_y = fetch_data(X_train, y_train, 10000, n_steps)
    sess.run(optimizer, feed_dict=feed_dict(batch_x, batch_y))
    if i % 10 == 0:
        save_path = saver.save(sess, file_model)
        print("Model saved in file: %s" % save_path)

        print("Evaluate combined model...")
        # print("Training error...")
        # eval_x, eval_y = fetch_data_all(X_train, y_train, X_train.shape[0], n_steps)
        # r_train = evaluate_models(eval_x, eval_y)
        # print(r_train)

        print("Validation error...")
        eval_x, eval_y = fetch_data(X_val, y_val, X_val.shape[0] - n_steps, n_steps, all=True)
        r_val = evaluate_models(eval_x, eval_y)
        print(r_val)

    i += 1
    print(i)
