import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import ensemble

# Заметка на будущее: использовать Jupyter

plt.style.use('ggplot')
'exec(%matplotlib inline)'
pd.options.display.width = 0

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
data = pd.read_csv(url, header=None, na_values='?')

print(data.shape)
print(data.head())
print(data.tail())

data.columns = ['A' + str(i) for i in range(1, 16)] + ['class']
# print(data.head())

# print(data['A5'][687])
# print(data.at[687, 'A5'])

# print(data.describe())

categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
numerical_columns = [c for c in data.columns if data[c].dtype.name != 'object']
# print(categorical_columns)
# print(numerical_columns)

# print(data[categorical_columns].describe())
# print(data.describe(include=[object]))

# Определить полный перечень значений категориальных признаков
# for c in categorical_columns:
#     print(data[c].unique())

# scatter_matrix(data, alpha=0.05, figsize=(10, 10))
# plt.show()

# print(data.corr())

col1 = 'A2'
col2 = 'A11'

# Диаграмма рассеивания для пары А2, А11
# plt.figure(figsize=(10, 6))
#
#  plt.scatter(data[col1][data['class'] == '+'],
#              data[col2][data['class'] == '+'],
#              alpha=0.75,
#              color='red',
#              label='+')
#
#  plt.scatter(data[col1][data['class'] == '-'],
#              data[col2][data['class'] == '-'],
#              alpha=0.75,
#              color='blue',
#              label='-')
#
#  plt.xlabel(col1)
#  plt.ylabel(col2)
#  plt.legend(loc='best')
# plt.show()

###########################
# Подготовка данных

# print(data.count(axis=0))
# удалить данные:
# data.dropna(axis=1))
# data.dropna(axis=0))

# Заполнить значения количественных признаков медианным значением:
data = data.fillna(data.median(axis=0), axis=0)
# print(data.count(axis=0))

# Заполнить значения категориальных признаков самым популярным:
data_describe = data.describe(include=[object])
for c in categorical_columns:
    data[c] = data[c].fillna(data_describe[c]['top'])

# print(data.count(axis=0))

binary_columns = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
# print(binary_columns, nonbinary_columns)

# Замена значений бинарных признаков на 0 и 1:

# data.at[data['A1'] == 'b', 'A1'] = 0
# data.at[data['A1'] == 'a', 'A1'] = 1
# data['A1'] = data['A1'].astype('object')
# print(data['A1'].describe())


for c in binary_columns:
    top = data_describe[c]['top']
    top_items = data[c] == top
    data.loc[top_items, c] = 0
    data.loc[np.logical_not(top_items), c] = 1
    data[c] = data[c].astype('object')

# print(data[binary_columns].describe())

# Подготовка Небинарных признаков:
# print(data['A4'].unique())
data_nonbinary = pd.get_dummies(data[nonbinary_columns])
# print(data_nonbinary.columns)

# Нормализация количественных признаков:
data_numerical = data[numerical_columns]
data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()
# print(data_numerical.describe())

# Финальный сбор таблиц:
data = pd.concat((data_numerical, data[binary_columns], data_nonbinary), axis=1)
data = pd.DataFrame(data, dtype=float)
# print(data.shape)
# print(data.columns)

X = data.drop(('class'), axis=1)  # Выбрасываем столбец 'class'
y = data['class']
feature_names = X.columns
# print(feature_names)
#
# print(X.shape)
# print(y.shape)
N, d = X.shape

######################################
# Деление на тестовую и обучающую выборку:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 11)

N_train, _ = X_train.shape
N_test,  _ = X_test.shape
# print(N_train, N_test)

# # Обучаем по методу KNN
# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)
# # print(knn)
#
# # Делаем предсказание и проверяем ошибку:
# y_train_predict = knn.predict(X_train)
# y_test_predict = knn.predict(X_test)
#
# err_train = np.mean(y_train != y_train_predict)
# err_test = np.mean(y_test != y_test_predict)
# # print(err_train, err_test)
#
# # Ошибка слишком большая, поищим оптимальное значение параметров:
# n_neighbors_array = [1, 3, 5, 7, 10, 15]
# knn = KNeighborsClassifier()
# grid = GridSearchCV(knn, param_grid={'n_neighbors': n_neighbors_array})
# grid.fit(X_train, y_train)
#
# best_cv_err = 1 - grid.best_score_
# best_n_neighbors = grid.best_estimator_.n_neighbors
# # print(best_cv_err, best_n_neighbors)
#
# knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
# knn.fit(X_train, y_train)
#
# err_train = np.mean(y_train != knn.predict(X_train))
# err_test  = np.mean(y_test != knn.predict(X_test))
# # print(err_train, err_test)

###################################
# Пробуем другой метод - SVC:
# svc = SVC()
# svc.fit(X_train, y_train)
#
# err_train = np.mean(y_train != svc.predict(X_train))
# err_test  = np.mean(y_test  != svc.predict(X_test))
# print(err_train, err_test)

# Попробуем найти лучшие значения параметров для радиального ядра:
# C_array = np.logspace(-3, 3, num=7)
# gamma_array = np.logspace(-5, 2, num=8)
# svc = SVC(kernel='rbf')
# grid = GridSearchCV(svc, param_grid={'C': C_array, 'gamma': gamma_array})
# grid.fit(X_train, y_train)
# print('CV error    = ', 1 - grid.best_score_)
# print('best C      = ', grid.best_estimator_.C)
# print('best gamma  = ', grid.best_estimator_.gamma)
#
# svc = SVC(kernel='rbf', C=grid.best_estimator_.C, gamma=grid.best_estimator_.gamma)
# svc.fit(X_train, y_train)
#
# err_train = np.mean(y_train != svc.predict(X_train))
# err_test  = np.mean(y_test  != svc.predict(X_test))
# print(err_train, err_test)

# Попробуем найти лучшие значения параметров для линейного ядра:
# C_array = np.logspace(-3, 3, num=7)
# svc = SVC(kernel='linear')
# grid = GridSearchCV(svc, param_grid={'C': C_array})
# grid.fit(X_train, y_train)
# print('CV error    = ', 1 - grid.best_score_)
# print('best C      = ', grid.best_estimator_.C)
#
# svc = SVC(kernel='linear', C=grid.best_estimator_.C)
# svc.fit(X_train, y_train)
#
# err_train = np.mean(y_train != svc.predict(X_train))
# err_test  = np.mean(y_test  != svc.predict(X_test))
# print(err_train, err_test)

# Полиномиальное ядро:
# C_array = np.logspace(-5, 2, num=8)
# gamma_array = np.logspace(-5, 2, num=8)
# degree_array = [2, 3, 4]
# svc = SVC(kernel='poly')
# grid = GridSearchCV(svc, param_grid={'C': C_array, 'gamma': gamma_array, 'degree': degree_array})
# grid.fit(X_train, y_train)
# print('CV error    = ', 1 - grid.best_score_)
# print('best C      = ', grid.best_estimator_.C)
# print('best gamma  = ', grid.best_estimator_.gamma)
# print('best degree = ', grid.best_estimator_.degree)
#
# svc = SVC(kernel='poly', C=grid.best_estimator_.C,
#           gamma=grid.best_estimator_.gamma, degree=grid.best_estimator_.degree)
# svc.fit(X_train, y_train)
#
# err_train = np.mean(y_train != svc.predict(X_train))
# err_test  = np.mean(y_test  != svc.predict(X_test))
# print(err_train, err_test)

#############################
# # Модель Random Forest
# rf = ensemble.RandomForestClassifier(n_estimators=100, random_state=11)
# rf.fit(X_train, y_train)
#
# err_train = np.mean(y_train != rf.predict(X_train))
# err_test  = np.mean(y_test  != rf.predict(X_test))
# print(err_train, err_test)
#
# # Отбор значимых признаков
# importances = rf.feature_importances_
# indices = np.argsort(importances)[::-1]
#
# print("Feature importances:")
# for f, idx in enumerate(indices):
#     print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, feature_names[idx], importances[idx]))
#
# # Cтолбцовая диаграмма, представляющая значимость первых 20 признаков
# d_first = 20
# plt.figure(figsize=(8, 8))
# plt.title("Feature importances")
# plt.bar(range(d_first), importances[indices[:d_first]], align='center')
# plt.xticks(range(d_first), np.array(feature_names)[indices[:d_first]], rotation=90)
# plt.xlim([-1, d_first])
# plt.show()
#
# best_features = indices[:8]
# best_features_names = feature_names[best_features]
# print(best_features_names)

########################
# Метод GBT:
gbt = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=11)
gbt.fit(X_train, y_train)

err_train = np.mean(y_train != gbt.predict(X_train))
err_test = np.mean(y_test != gbt.predict(X_test))
print(err_train, err_test)

# Используем только значимые признаки
# gbt = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=11)
# gbt.fit(X_train[best_features_names], y_train)
#
# err_train = np.mean(y_train != gbt.predict(X_train[best_features_names]))
# err_test = np.mean(y_test != gbt.predict(X_test[best_features_names]))
# print(err_train, err_test)





