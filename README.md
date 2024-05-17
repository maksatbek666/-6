# -6
# Сравнение Логистической Регрессии и K-ближайших соседей (KNN) на Датасете Iris

Этот проект сравнивает два алгоритма машинного обучения - Логистическую Регрессию и K-ближайших соседей (KNN) - на датасете Iris. Оценка моделей проводится по метрике точности (precision).

## Описание файлов

### main.py
Скрипт выполняет следующие шаги:
1. Загрузка датасета Iris.
2. Разделение данных на обучающую и тестовую выборки.
3. Создание и обучение модели Логистической Регрессии.
4. Предсказание и оценка точности модели Логистической Регрессии.
5. Создание и обучение модели KNN.
6. Предсказание и оценка точности модели KNN.

python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Градиентный спуск
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Вычисление градиентов
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Обновление параметров
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

# Загрузка данных
iris = load_iris()
X, y = iris.data, iris.target

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели логистической регрессии
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Предсказание меток классов на тестовом наборе
y_pred_log_reg = log_reg.predict(X_test)

# Оценка точности модели логистической регрессии
precision_log_reg = precision_score(y_test, y_pred_log_reg, average='weighted')
print("Precision of Logistic Regression:", precision_log_reg)

# Создание и обучение модели метода k ближайших соседей
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Предсказание меток классов на тестовом наборе
y_pred_knn = knn.predict(X_test)

# Оценка точности модели метода k ближайших соседей
precision_knn = precision_score(y_test, y_pred_knn, average='weighted')
print("Precision of KNN:", precision_knn)


## Требования

Для запуска этого проекта вам потребуется установить Python и следующие библиотеки:
- numpy
- scikit-learn

Вы можете установить необходимые библиотеки, используя следующую команду:
bash
pip install numpy scikit-learn


## Запуск кода

Для запуска кода просто выполните скрипт `main.py`:


## Результаты

Скрипт выведет точность (precision) каждой модели на тестовом наборе данных:
- Точность Логистической Регрессии
- Точность KNN

