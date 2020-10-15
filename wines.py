import pandas as pd

file = pd.read_csv('D:/Python Projects/Machine-Learning/wine_dataset.csv')

#print(file.head(50))

file['style'] = file['style'].replace('red', 0)

file['style'] = file['style'].replace('white', 1)

#separando variaveis entre preditoras e variaveis alvo

y = file['style']
x = file.drop('style', axis=1)

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste=train_test_split(x, y, test_size= 0.3)

from sklearn.ensemble import ExtraTreesClassifier
#criação do modelo
model = ExtraTreesClassifier()

model.fit(x_treino, y_treino)
#show results
results = model.score(x_teste, y_teste)

print('Accuracy: ', results)

print(y_teste[200:215])

print(x_teste[200:215])

prevision = model.predict(x_teste[200:215])

print(prevision)





