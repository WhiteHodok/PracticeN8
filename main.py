import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv('data.csv')
X = data.drop(['id', 'diagnosis'], axis=1) # Выделяем независимые переменные
y = data['diagnosis'] # Выделяем зависимую переменную
# Разбиваем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Создаем модель обучающего дерева с максимальной глубиной 3
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) # Вычисляем точность предсказаний с помощью метрики accuracy_score
print("Accuracy:", accuracy) # Выводим точность на экран

