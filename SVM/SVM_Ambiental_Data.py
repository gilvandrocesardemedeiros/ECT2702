'''
@author: Gilvandro Cesar de Medeiros
Algoritmo de Support Vector Machine para estimar qual o mes do ano baseado em dados ambientais na cidade de Natal-RN.
Base de dados coletada a partir do Sistema de Monitoramento Agrometeorologico (AGRITEMPO) <www.agritempo.gov.br>
Sendo: 
    data (mes da medicao), tMin (temperatura minima), tMed (temperatura media), tMax (temperatura maxima), 
    precipitacao (volume de chuva), urMin (umidade relativa minima), urMax (umidade relativa maxima), evPot (evapotranspiracao potencial), 
    radSol (radiacao solar), velVen (velocidade do vento), poMin (ponto de orvalho minimo), poMax (ponto de orvalho maximo), 
    patmMin (pressao atmosferica minima), patmMax (pressao atmosferica maxima), evReal (evapotranspiracao real), daSolo (disponibilidade da agua no solo).
'''

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

#Leitura dos dados bruto: Separador eh ';' e as 8 primeiras linhas sao informacoes dispensaveis
dados = pd.read_csv("dadosClimaticos_216.csv", sep = ';', skiprows=8)
#Transformando o arquivo em um arquivo da classe Numpy Array
dados = np.array(dados)
#Extraindo os dados relevantes para arrays'
dataBruta = dados[:, 0]
tMinBruta = dados[:, 1]  
tMedBruta = dados[:, 3]
tMaxBruta = dados[:, 5]
precipitacaoBruta = dados[:, 7]
urMinBruta = dados[:, 11]
urMaxBruta = dados[:, 13]
evPotBruta = dados[:, 15]
radSolBruta = dados[:, 16]
velVenBruta = dados[:, 17]
poMinBruta = dados[:, 18]
poMaxBruta = dados[:, 19]
patmMinBruta = dados[:, 20]
patmMaxBruta = dados[:, 21]
evRealBruta = dados[:,22]
daSoloBruta = dados[:, 23]
#Formando um array bidimensional com os arrays extraidos acima 
matrizDados = np.array([dataBruta, tMinBruta, tMedBruta, tMaxBruta, precipitacaoBruta, urMinBruta, 
                        urMaxBruta, evPotBruta, radSolBruta, velVenBruta, poMinBruta, poMaxBruta, 
                        patmMinBruta, patmMaxBruta, evRealBruta, daSoloBruta])
#Transformando esse array em um dataframe, para se livrar dos NaN devido a imperfeicoes na base de dados
matrizDados = pd.DataFrame(matrizDados)
#Excluindo as linhas que possuem NaN nas variaveis
matrizDados = matrizDados.dropna(how = 'any', axis = 1)
#Transformando o dataframe de volta em array bidimensional
matrizDados = np.array(matrizDados)
#Extraindo novamente os dados relevantes para arrays, agora apenas com dados consistentes
data = matrizDados[0, :]
tMin = matrizDados[1, :]
tMed = matrizDados[2, :]
tMax = matrizDados[3, :]
precipitacao = matrizDados[4, :]
urMin = matrizDados[5, :]
urMax = matrizDados[6, :]
evPot = matrizDados[7, :]
radSol = matrizDados[8, :]
velVen = matrizDados[9, :]
poMin = matrizDados[10, :]
poMax = matrizDados[11, :]
patmMin = matrizDados[12, :]
patmMax = matrizDados[13, :]
evReal = matrizDados[14, :]
daSolo = matrizDados[15, :]

#Declarando um array ano, que vai guardar as informacoes sobre a que ano cada evento pertence
ano = [None] * len(data)
#Declarando um contador que vai armazenar quantas vezes determinado ano (intervalo de anos) se repete
contAno = 0

#Transformando dados de tempo do tipo dd/mm/aaaa em dados que so possuam o mes
#Convertendo os valores de string "xx,xx" para float xx.xx ou de string para int, quando for o caso
for i in range(len(data)):
    dt = datetime.strptime(data[i], '%d/%m/%Y')
    data[i] = dt.month
    ano[i] = dt.year
    if ano[i] > 2013:
        contAno = contAno + 1    
    tMin[i] = float(tMin[i].replace(",","."))
    tMed[i] = float(tMed[i].replace(",","."))
    tMax[i] = float(tMax[i].replace(",","."))
    precipitacao[i] = float(precipitacao[i].replace(",", "."))
    urMin[i] = int(urMin[i])
    urMax[i] = int(urMax[i])
    evPot[i] = float(evPot[i].replace(",","."))
    radSol[i] = float(radSol[i].replace(",","."))
    velVen[i] = float(velVen[i].replace(",","."))
    poMin[i] = float(poMin[i].replace(",","."))
    poMax[i] = float(poMax[i].replace(",","."))
    patmMin[i] = float(patmMin[i].replace(",","."))
    patmMax[i] = float(patmMax[i].replace(",","."))
    evReal[i] = float(evReal[i].replace(",","."))
    daSolo[i] = float(daSolo[i].replace(",","."))
    
#Declarando os arrays (inicializados como vazio) que irao ter informacoes referentes apenas ao intervalo de anos utilizado
dataDefinitiva = [None] * contAno
tMinDefinitiva = [None] * contAno
tMedDefinitiva = [None] * contAno
tMaxDefinitiva = [None] * contAno
precipitacaoDefinitiva = [None] * contAno
urMinDefinitiva = [None] * contAno
urMaxDefinitiva = [None] * contAno
evPotDefinitiva = [None] * contAno
radSolDefinitiva = [None] * contAno
velVenDefinitiva = [None] * contAno
poMinDefinitiva = [None] * contAno
poMaxDefinitiva = [None] * contAno
patmMinDefinitiva = [None] * contAno
patmMaxDefinitiva = [None] * contAno
evRealDefinitiva = [None] * contAno
daSoloDefinitiva = [None] * contAno

#Contador que vai guardar o indice dos arrays definitivos 
contArray = 0

#Separando os arrays definitivos no intervalo de tempo solicitado
for i in range(len(data)):
    if ano[i] > 2013:
        dataDefinitiva[contArray] = data[i]
        tMinDefinitiva[contArray] = tMin[i]
        tMedDefinitiva[contArray] = tMed[i]
        tMaxDefinitiva[contArray] = tMax[i]
        precipitacaoDefinitiva[contArray] = precipitacao[i]
        urMinDefinitiva[contArray] = urMin[i]
        urMaxDefinitiva[contArray] = urMax[i]
        evPotDefinitiva[contArray] = evPot[i]
        radSolDefinitiva[contArray] = radSol[i]
        velVenDefinitiva[contArray] = velVen[i]
        poMinDefinitiva[contArray] = poMin[i]
        poMaxDefinitiva[contArray] = poMax[i]
        patmMinDefinitiva[contArray] = patmMin[i]
        patmMaxDefinitiva[contArray] = patmMax[i]
        evRealDefinitiva[contArray] = evReal[i]
        daSoloDefinitiva[contArray] = daSolo[i]
        contArray = contArray + 1

#Obtendo uma matrizDados com todos os dados preprocessados, exceto dataDefinitiva, que serah utilizado posteriormente
matrizDados = np.array([tMinDefinitiva, tMedDefinitiva, tMaxDefinitiva, precipitacaoDefinitiva, urMinDefinitiva, 
                        urMaxDefinitiva, evPotDefinitiva, radSolDefinitiva, velVenDefinitiva, poMinDefinitiva, 
                        poMaxDefinitiva, patmMinDefinitiva, patmMaxDefinitiva, evRealDefinitiva, daSoloDefinitiva])
#Transpondo a matrizDados, para ficar com a mesma orientacao que o array dataDefinitiva
matrizDados = matrizDados.transpose()
#Separando os conjuntos de treino e teste com os dados embaralhados, sendo 90% para treino e 10% para teste
X_train, X_test, y_train, y_test = train_test_split(matrizDados, dataDefinitiva, test_size = 0.1, random_state = 0)

#Transformando os dados em escala, estabelecendo uma media para cada coluna e calculando quao longe da media determinado valor se encontra
sc = StandardScaler()
#fit_transform pois esta serah uma variavel usada para treinar 
X_train = sc.fit_transform(X_train)
#transform pois o unico intuito eh tornar X_train um array para usar na funcao fit, abaixo
X_test = sc.transform(X_test)

#Corrigindo erro relacionado ao tipo do y_train e do y_test, necessario para usar no fit
y_train = np.array(y_train).astype(int)
y_test = np.array(y_test).astype(int)

#Este processo faz parte da busca pelos melhores parametros C e Gamma para a SVM.
#Declaracao de variaveis para ajudar na escolha dos parametros que retornem o melhor numero de acertos
acertos_maior = 0
C_maior = 0
gamma_maior = 0
total_iteracoes_maior = 0
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
print("O numero de iteracoes necessarias sera de: ")
print(len(C_range)*len(gamma_range))
for i in range(len(C_range)):
    for j in range(len(gamma_range)):
        print(i*len(gamma_range)+j)
        #Numero de acertos locais
        acerto_local = 0
        #Numero de iteracoes na matriz de confusao
        total_iteracoes = 0
        #Implementacao do classificador
        classifier = SVC(C = C_range[i], kernel = 'rbf', random_state = 0, gamma = gamma_range[j])
        #Modelagem da SVM
        classifier.fit(X_train, y_train)
        #Verificando o desempenho da modelagem, usando X_test como parametro para predict no classificador
        y_pred = classifier.predict(X_test)
        for num in range(len(y_test)):
            if y_test[num] == y_pred[num]:
                acerto_local = acerto_local + 1
                continue
            if y_test[num] + 1 == y_pred[num]:
                acerto_local = acerto_local + 1
                continue
            if y_test[num] - 1 == y_pred[num]:
                acerto_local = acerto_local + 1
                continue
            if y_test[num] == 12 and y_pred[num] == 1:
                acerto_local = acerto_local + 1
                continue
            if y_test[num] == 1 and y_pred[num] == 12:
                acerto_local = acerto_local + 1
                continue
        #Matriz de confusao, cujos valores da diagonal principal sao os acertos, acima da diagonal principal falsos-positivos e abaixo, falso-negativos
        #cm = confusion_matrix(y_test, y_pred)
        #print(cm)
        #Soma de todas as iteracoes feitas exibidas na matriz de confusao
        #for l in range(len(cm)):
        #    for c in range(len(cm[0])):
        #        total_iteracoes = total_iteracoes + cm[l][c]
        #        if l == c:
        #            acerto_local = acerto_local + cm[l][l]
        #Verificando se o numero de acertos locais foi o maior ate entao
        if acerto_local > acertos_maior:
            acertos_maior = acerto_local
            total_iteracoes_maior = len(y_test)
            C_maior = C_range[i]
            gamma_maior = gamma_range[j]

#Mostrando os resultados obtidos:
print("O numero de acertos foi de: ")
print(acertos_maior)
print("Com uma taxa de acertos (%) de: ")
print(float(acertos_maior/total_iteracoes_maior)*100)
print("Com o parametro C = ")
print(C_maior)
print("E com o parametro gamma = ")
print(gamma_maior)

'''
#Classificador com os parametros que obtiveram melhor resultado
classifier = SVC(C = 10000000.0, kernel = 'rbf', random_state = 0, gamma = 0.0001)
'''

'''
#Para o valor EXATO:
Resultado obtido:
O numero de acertos foi de: 
101
Com uma taxa de acertos (%) de: 
79.52755905511812
Com o parametro C = 
10000000.0
E com o parametro gamma = 
0.0001

#Considerando um mês de margem de erro tolerável:
O numero de acertos foi de: 
122
Com uma taxa de acertos (%) de: 
96.06299212598425
Com o parametro C = 
1000000000.0
E com o parametro gamma = 
0.0001
'''

'''
CONSIDERAÇOES FINAIS:
    Este código ainda está em desenvolvimento!
    Embora a taxa de acerto nao tenha sido tao alta, o erro entre o valor esperado para o mes e a data real
    aparentou ser pequeno a partir de analises visuais feitas previamente. 
    Para desenvolver este codigo, visualiza-se possibilidade de aplicacao de algoritmos de cluster.
    Ja para medir melhor o "acerto", considera-se relevante usar de algum indice para medir o erro relativo.
'''