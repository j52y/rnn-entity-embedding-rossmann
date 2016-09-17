import numpy

numpy.random.seed(123)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Reshape
from keras.layers.embeddings import Embedding


class NN_with_EntityEmbedding(object):
    def __init__(self, X_train, y_train, X_val, y_val):
        self.nb_epoch = 10
        self.max_log_y = max(numpy.max(numpy.log(y_train)), numpy.max(numpy.log(y_val)))
        self.__build_keras_model()
        self.fit(X_train, y_train, X_val, y_val)

    def evaluate(self, X_val, y_val):
        assert (min(y_val) > 0)
        guessed_sales = self.guess(X_val)
        relative_err = numpy.absolute((y_val - guessed_sales) / y_val)
        result = numpy.sum(relative_err) / len(y_val)
        return result

    def split_features(self, X):
        X_list = []

        # 7, 30개씩 잘라서 학습시켜보자.. LSTM 7 or 30 짜리로
        # X_train을 소팅하여 적용하면서 validation error rate가 0.14 -> 0.3으로 증가함

        store_index = X[:, [1]]  # 첫번 째 칼럼, X의 모양 유지하면서 꺼낸다.
        X_list.append(store_index)

        day_of_week = X[:, [2]]
        X_list.append(day_of_week)

        promo = X[:, [3]]
        X_list.append(promo)

        year = X[:, [4]]
        X_list.append(year)

        month = X[:, [5]]
        X_list.append(month)

        day = X[:, [6]]
        X_list.append(day)

        State = X[:, [7]]
        X_list.append(State)

        return X_list

    def preprocessing(self, X):
        X_list = self.split_features(X)
        return X_list

    def __build_keras_model(self):
        models = []

        model_store = Sequential()
        model_store.add(Embedding(1115, 10, input_length=1))
        model_store.add(Reshape(target_shape=(10,)))
        models.append(model_store)

        model_dow = Sequential()
        model_dow.add(Embedding(7, 6, input_length=1))
        model_dow.add(Reshape(target_shape=(6,)))
        models.append(model_dow)

        model_promo = Sequential()
        model_promo.add(Dense(1, input_dim=1))
        models.append(model_promo)

        model_year = Sequential()
        model_year.add(Embedding(3, 2, input_length=1))
        model_year.add(Reshape(target_shape=(2,)))
        models.append(model_year)

        model_month = Sequential()
        model_month.add(Embedding(12, 6, input_length=1))
        model_month.add(Reshape(target_shape=(6,)))
        models.append(model_month)

        model_day = Sequential()
        model_day.add(Embedding(31, 10, input_length=1))
        model_day.add(Reshape(target_shape=(10,)))
        models.append(model_day)

        model_germanstate = Sequential()
        model_germanstate.add(Embedding(12, 6, input_length=1))
        model_germanstate.add(Reshape(target_shape=(6,)))
        models.append(model_germanstate)

        self.model = Sequential()
        self.model.add(Merge(models, mode='concat'))
        self.model.add(Dense(1000, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(500, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='mean_absolute_error', optimizer='adam')

    # 모델의 마지막이 sigmoid라서 y값 맞춰준다
    def _val_for_fit(self, val):
        val = numpy.log(val) / self.max_log_y
        return val

    # 실제 예측 값은 sigmoid가 아니므로 모델에서 나온 sigmoid 값을 실제 값으로 변환한다.
    def _val_for_pred(self, val):
        return numpy.exp(val * self.max_log_y)

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(self.preprocessing(X_train), self._val_for_fit(y_train),
                       validation_data=(self.preprocessing(X_val), self._val_for_fit(y_val)),
                       nb_epoch=self.nb_epoch, batch_size=128,
                       )
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, features):
        features = self.preprocessing(features)
        result = self.model.predict(features).flatten()
        return self._val_for_pred(result)
