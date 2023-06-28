from matplotlib import pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  ConfusionMatrixDisplay, classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import time

inicio = time.time()
data = pd.read_csv('D:/IA/ml-2023-1-trabalho-final/Titanic-Dataset.csv')

data = data.dropna(axis=1, how='all')

for y in data.columns:
    if data[y].dtype == 'object': 
        lbl = LabelEncoder()
        lbl.fit(list(data[y].values))
        data[y] = lbl.transform(list(data[y].values))

colunas = ['PassengerId',  'Survived', 'Name', 'Ticket', 'Fare',
           'Cabin',
           'Embarked',
           ]

dt_copy = data.copy()
dt_copy.drop(colunas, axis=1, inplace=True)
print(dt_copy)

for column in dt_copy.columns:

    q1 = dt_copy[column].quantile(0.25)
    q3 = dt_copy[column].quantile(0.75)
    iqr = q3 - q1

    li = q1 - (1.5 * iqr)
    ls = q3 + (1.5 * iqr) 

    outliers = (dt_copy[column] < li) | (dt_copy[column] > ls)

    dt_copy.loc[outliers, column] = np.nan

dt_copy = dt_copy.join(data[colunas])

dt_copy = dt_copy[dt_copy.columns[dt_copy.isna().sum()/dt_copy.shape[0] < 0.9]]
dt_copy = dt_copy.fillna(dt_copy.median())


# exibindo correlações
correlations = dt_copy.corrwith(dt_copy['Survived'])
print(correlations.sort_values(ascending=False).head(10))
print('\n\n')
print(correlations.sort_values(ascending=True).head(10))
print('\n\n')
print(correlations.abs().sort_values(ascending=False).head(10))
print('\n\n')

# criando listas X (features) e y (resultado esperado)
X = dt_copy.drop(columns=['Survived'])

X = dt_copy[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]

scaler =StandardScaler().fit(X)
X = scaler.transform(X)

y = dt_copy['Survived']

# separando 66% do dataset para treinamento e 33% para validação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=101)

names = [
    "Decision Tree",
]

classifiers = [
    DecisionTreeClassifier(),
]

results = []

for model_names, model in zip(names,classifiers):
    
    model.fit(X_train, y_train)
    
    prds = model.predict(X_test)
    
    cn = confusion_matrix(y_test, prds).ravel()
    
    tn, fp, fn, tp = confusion_matrix(y_test, prds).ravel()
    acs = accuracy_score(y_test, prds)
     
    print(f'Model {model_names}:\n\n'
          f'tn {tn}, fp {fp}, fn {fn}, tp {tp}\n\n'
          f'Accuracy: {acs}\n\n'
      f'Classification Report:\n{classification_report(y_test, prds, target_names = ["morreu", "sobreviveu"])}\n')
    
    fim = time.time()
    tempo_de_processamento = fim - inicio
    print(f"O tempo de processamento foi de {tempo_de_processamento} segundos.")
    
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=['morreu', 'sobreviveu'], cmap='Blues', values_format='d')
    plt.show()
