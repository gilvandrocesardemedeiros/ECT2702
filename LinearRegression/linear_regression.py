'''
@author: Gilvandro Cesar de Medeiros
Para o funcionamento da regressao linear, separa-se os dados em dois conjuntos: treino e teste.
Para nao viciar a regressao linear, utilizou-se do metodo shuffle para embaralhar os dados.
O conjunto de treino serah destinado para modelar a regressao linear, enquanto que o de 
teste serah utilizado para validar o funcionamento.
'''

import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
from sklearn.utils import shuffle
import time
from sklearn.metrics import mean_squared_error, r2_score

#Lendo a base de dados "Salary_Data.csv" e armazenando na variavel trabalhadores
trabalhadores = pd.read_csv("Salary_Data.csv")
#Separando a coluna "years of experience" do dataframe Salary_Data na variavel trabalhadores_xp
trabalhadores_xp = trabalhadores.iloc[:, :-1].values
#Separando a coluna "salary" do dataframe Salary_Data na variavel trabalhadores_salary 
trabalhadores_salary = trabalhadores.iloc[:,1].values
#Misturando os dados para tornar o mais aleatorio possivel
trabalhadores_xp, trabalhadores_salary = shuffle(trabalhadores_xp, trabalhadores_salary, random_state = int(time.time()))
#Pegando todos os dados de xp e salary, excluindo os 10 ultimos, e separando para conjunto de treino
trabalhadores_xp_train = trabalhadores_xp[:-10]
trabalhadores_salary_train = trabalhadores_salary[:-10]
#Pegando os 10 ultimos dados dos arrays de xp e salary, e separando-os para conjunto de teste
trabalhadores_xp_test = trabalhadores_xp[-10:]
trabalhadores_salary_test = trabalhadores_salary[-10:]
#Criando objeto para regressao linear
model = linear_model.LinearRegression()
#Treinando model com xp_train para o eixo x e salary_train para o eixo y da regressao linear
model.fit(trabalhadores_xp_train, trabalhadores_salary_train)
#Validando o desempenho da regressao linear para estimar os valores, utilizando o conjunto de teste
trabalhadores_salary_predict = model.predict(trabalhadores_xp_test)
#Esboco grafico para a modelagem do problema:
plt.scatter(trabalhadores_xp_test, trabalhadores_salary_test,  color='black')
plt.plot(trabalhadores_xp_test, trabalhadores_salary_predict, color='blue', linewidth=3)
plt.show()
#Agora, interagindo com o usuario, para que ele visualize o efeito da regressao
valor = 1
while(valor):
    print("Para parar, digite 0:")
    valor = float(input("Entre com os anos de experiencia do funcionario, para estimar quanto ele recebe:"))
    if valor != 0.0:
        print("O modelo previu o salario de ($): ")
        print(round(float(model.predict(valor)),2))

print("Programa encerrado!")