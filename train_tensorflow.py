import os
import pickle
import numpy as np
import tensorflow as tf

from model import *


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

w1 = tf.Variable(tf.random_uniform((v_all, 1000), -r_range, r_range))
h1 = tf.nn.relu(tf.matmul(emb_all, w1))

w2 = tf.Variable(tf.random_uniform((1000, 500), -r_range, r_range))
h2 = tf.nn.relu(tf.matmul(h1, w2))

w3 = tf.Variable(tf.random_uniform((500, 1), -r_range, r_range))
h3 = tf.nn.sigmoid(tf.matmul(h2, w3))

# cost = tf.reduce_mean(tf.abs(tf.sub(h3, y_label)))
cost = tf.reduce_sum(tf.abs(tf.sub(h3, y_label)))

learning_rate = 0.005  # 0.1일 때 발산
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

train_ratio = 0.9
iterations = 1000

sorted_data_file = 'sorted_feature_train_data.pickle'
f = open(sorted_data_file, 'rb')
(X, y) = pickle.load(f)

num_records = len(X)

train_size = int(train_ratio * num_records)

X_train = X[:train_size]  # (759904,8)
X_val = X[train_size:]
y_train = y[:train_size]
y_val = y[train_size:]

max_log_y = max(np.max(np.log(y_train)), np.max(np.log(y_val)))


def sample(x, y, n):
    num_row = x.shape[0]
    indices = np.random.randint(num_row, size=n)
    return x[indices, :], y[indices]


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
    guessed_sales = val_for_pred(h3.eval(feed_dict=feed_dict_x(x)))
    mean_sales = guessed_sales.mean(axis=0)
    relative_err = np.absolute((y - mean_sales) / y)
    result = np.sum(relative_err) / len(y)
    return result


sess = tf.InteractiveSession()
saver = tf.train.Saver()

file_model = "./tmp/model.ckpt"

if os.path.isfile(file_model):
    saver.restore(sess, file_model)
else:
    sess.run(tf.initialize_all_variables())

i = 0
while i <= iterations:
    batch_x, batch_y = sample(X_train, y_train, 200000)
    sess.run(optimizer, feed_dict=feed_dict(batch_x, batch_y))
    if i % 10 == 0:
        # save_path = saver.save(sess, file_model)
        # print("Model saved in file: %s" % save_path)

        # print("Evaluate combined model...")
        # print("Training error...")
        # r_train = evaluate_models(X_train, y_train)
        # print(r_train)

        print("Validation error...")
        r_val = evaluate_models(X_val, y_val)
        print(r_val)

    i += 1
    print(i)
