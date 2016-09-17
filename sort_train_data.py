import pickle
import numpy as np

data_file = 'feature_train_data.pickle'
sorted_data_file = 'sorted_feature_train_data.pickle'

f = open(data_file, 'rb')
X, y = pickle.load(f)

num_records = len(X)

Xy = np.append(X, y.reshape(num_records, 1), axis=1)

print('sorting train data, size %s' % Xy.shape[0])

Xy = Xy[Xy[:, 6].argsort()]  # day로 sorting
Xy = Xy[Xy[:, 5].argsort(kind='mergesort')]  # month로 sorting
Xy = Xy[Xy[:, 4].argsort(kind='mergesort')]  # year로 sorting
Xy = Xy[Xy[:, 1].argsort(kind='mergesort')]  # store 로 sorting

X = Xy[:, :-1]  # 마지막 column을 제외한 나머지 모두
y = Xy[:, -1]  # 마지막 column만 고른다

print('check ')
for i in range(Xy.shape[0]):
    x = X[i]
    print(x[1], x[4], x[5], x[6])

print('saving %s' % sorted_data_file)

f = open(sorted_data_file, 'wb')
pickle.dump((X, y), f, -1)

