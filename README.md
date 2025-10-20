# Класифікація виживання пацієнтів — *Haberman Survival Dataset*

## Опис проєкту
Цей проєкт демонструє використання методів машинного навчання — **логістичної регресії** та **методу опорних векторів (SVM)** — для класифікації виживання пацієнтів після операції на основі даних із **Haberman Survival Dataset**.

Мета проєкту — дослідити, як різні моделі машинного навчання можуть передбачати результат лікування пацієнта за такими параметрами:
- **Age** — вік пацієнта;
- **Year** — рік операції;
- **Nodes** — кількість уражених лімфатичних вузлів.

---

## Використані бібліотеки
Проєкт реалізовано мовою **Python** з використанням таких бібліотек:

- `pandas` — обробка даних;
- `numpy` — чисельні обчислення;
- `matplotlib` — побудова графіків;
- `scikit-learn` — моделі машинного навчання (Logistic Regression, SVM, метрики, поділ вибірки).

---

## Основні етапи роботи

### 1️⃣ Завантаження та підготовка даних
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

url = 'https://raw.githubusercontent.com/rikhuijzer/haberman-survival-dataset/main/haberman.csv'
data = pd.read_csv(url)

data.columns = ['Age', 'Year', 'Nodes', 'Survival']
data['Survival'] = data['Survival'].apply(lambda x: 1 if x == 1 else 0)


### 2️⃣ Формування вибірки
X = data[['Age', 'Nodes']]
y = data['Survival']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


### 3️⃣ Логістична регресія
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("Логістична регресія")
print("Точність (Accuracy):", accuracy_score(y_test, y_pred_log))
print("Матриця плутанини:\n", confusion_matrix(y_test, y_pred_log))
print("Звіт класифікації:\n", classification_report(y_test, y_pred_log))

### 4️⃣ Метод опорних векторів (SVM)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

print("SVM")
print("Точність (Accuracy):", accuracy_score(y_test, y_pred_svm))
print("Матриця плутанини:\n", confusion_matrix(y_test, y_pred_svm))
print("Звіт класифікації:\n", classification_report(y_test, y_pred_svm))


### 5️⃣ Контрольне розпізнавання нових об’єктів
new_objects = np.array([
[45, 0],
[60, 5],
[70, 20],
[30, 1],
[55, 3]
])

log_preds_new = log_model.predict(new_objects)
svm_preds_new = svm_model.predict(new_objects)

print("Прогнози логістичної регресії:", log_preds_new)
print("Прогнози SVM:", svm_preds_new)


### 6️⃣ Візуалізація результатів
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test['Age'], X_test['Nodes'], c=y_test, cmap='bwr', alpha=0.6, label='Тестові дані')
plt.scatter(new_objects[:, 0], new_objects[:, 1], c='green', marker='x', s=100, label='Нові об’єкти')
plt.title("Logistic Regression")
plt.xlabel("Age")
plt.ylabel("Nodes")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_test['Age'], X_test['Nodes'], c=y_test, cmap='bwr', alpha=0.6, label='Тестові дані')
plt.scatter(new_objects[:, 0], new_objects[:, 1], c='green', marker='x', s=100, label='Нові об’єкти')
plt.title("SVM")
plt.xlabel("Age")
plt.ylabel("Nodes")
plt.legend()

plt.tight_layout()
plt.show()


### 7️⃣ Порівняння результатів
results = pd.DataFrame({
'Age': new_objects[:, 0],
'Nodes': new_objects[:, 1],
'Logistic Regression': log_preds_new,
'SVM': svm_preds_new
})
print(results)


Кожна точка на графіку позначає пацієнта:

- 🟢 — вижив (1)  
- 🔴 — не вижив (0)  
- ❌ — нові об’єкти класифікації

---
