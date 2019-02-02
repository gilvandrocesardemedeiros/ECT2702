'''
Autoria: Gilvandro Cesar de Medeiros

    -> Este codigo tem por finalidade classificar base de dados em perfis politicos de 
    "extrema esquerda" (EE) e "extrema direita" (ED) utilizando KMeans.
    -> Dois centroides foram utilizados.
    -> A identificacao sobre qual centroide eh de EE e qual eh de ED se dah a partir da
    analise da base de dados, considerando as definicoes abaixo: 
    -> Perguntas que, quando a resposta for afirmativa, sao consideradas de EE:
        [:,1] Lula deve ser solto
        [:,2] O impeachment de Dilma foi um golpe
        [:,3] Legalização do aborto
        [:,4] Políticas de ações afirmativas (cotas)
        [:,5] A favor da reforma agrária
        [:,9] Apoia a laicidade do Estado
        [:,13] A favor de uma reforma política
    -> Perguntas que, quando a resposta for afirmativa, sao consideradas de ED:
        [:,0] Apoia privatizações das empresas brasileiras
        [:,6] Todos os brasileiro devem alcançar suas conquistas através da meritocracia
        [:,7] Apoia a legalização do porte de armas
        [:,8] Apoia a redução da maioridade penal
        [:,10] Apoia a diminuição do Estado
        [:,11] As Universidades são um gasto público
        [:,12] A favor do foro privilegiado
        [:,14] A favor de uma diminuição de regalias para os políticos
    -> A partir dessas consideracoes, define-se o perfil de EE e ED. 
    -> Com esse perfil definido, pretende-se julgar o posicionamento do usuario a partir 
    de valores lidos via teclado.
'''

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv("perfil_politico.csv")
opinioes = dataset.iloc[:,:].values
#Array que avalia numericamente quao "de direita" eh uma das pessoas da base de dados
direitomero = []

#Inicializando o "direitomero" com indice 0 para cada uma das pessoas da base de dados
for i in range(len(opinioes)):
    direitomero.append(0)
    
for i in range(len(opinioes)):
    for j in range(len(opinioes[0])):
        #Quando a resposta positiva significa que a pessoa eh de esquerda
        if j == 1 or j == 2 or j == 3 or j == 4 or j == 5 or j == 9 or j == 13:
            direitomero[i] = direitomero[i] - opinioes[i,j]
        #Quando a resposta positiva significa que a pessoa eh de direita
        else:
            direitomero[i] = direitomero[i] + opinioes[i,j]

#Escolhendo os perfis que representam EE e ED, juntamente com os indices destes
for i in range(len(direitomero)):
    if direitomero[i] == max(direitomero):
        indiceDireita = i
    elif direitomero[i] == min(direitomero):
        indiceEsquerda = i

#Modelagem computacional com KMeans
kmeans = KMeans(n_clusters = 2, init = 'random', random_state = 1)
#Resultado de para qual centroide cada uma das pessoas na base de dados converge
ResultadoCentroides = kmeans.fit_predict(opinioes)
#Estabelecendo qual o indice do centroide de esquerda e de direita
centroideDireita = ResultadoCentroides[indiceDireita]
centroideEsquerda = ResultadoCentroides[indiceEsquerda]

#Funcao que serah chamada para normalizar o valor da resposta
#Parametro: texto informado pelo usuario
#Retorno: valor 1 para SIM, 0 para NAO, 0.5 para TALVEZ e -1 para texto invalido.
def respostaNormalizada(texto):
    if texto == "SIM":
        return 1
    if texto == "NAO":
        return 0
    if texto == "TALVEZ":
        return 0.5
    else:
        print("ERRO!! TEXTO INVALIDO")
        return -1
    
#Funcao que serah chamada para fazer a pergunta
#Parametro: indice da pergunta
#Retorno: valor 1 para SIM, 0 para NAO, 0.5 para TALVEZ e -1 para texto invalido.
def pergunta(indice):
    print("Respostas possiveis: SIM, NAO, TALVEZ")
    if indice == 0:
        resposta = input("Apoia privatizações das empresas brasileiras?")
        return respostaNormalizada(resposta)
    if indice == 1:
        resposta = input("Lula deve ser solto?")
        return respostaNormalizada(resposta)
    if indice == 2:
        resposta = input("O impeachment de Dilma foi um golpe?")
        return respostaNormalizada(resposta)
    if indice == 3:
        resposta = input("Legalização do aborto?")
        return respostaNormalizada(resposta)
    if indice == 4:
        resposta = input("Políticas de ações afirmativas (cotas)?")
        return respostaNormalizada(resposta)
    if indice == 5:
        resposta = input("A favor da reforma agrária?")
        return respostaNormalizada(resposta)
    if indice == 6:
        resposta = input("Todos os brasileiro devem alcançar suas conquistas através da meritocracia?")
        return respostaNormalizada(resposta)
    if indice == 7:
        resposta = input("Apoia a legalização do porte de armas?")
        return respostaNormalizada(resposta)
    if indice == 8:
        resposta = input("Apoia a redução da maioridade penal?")
        return respostaNormalizada(resposta)
    if indice == 9:
        resposta = input("Apoia a laicidade do Estado?")
        return respostaNormalizada(resposta)
    if indice == 10:
        resposta = input("Apoia a diminuição do Estado?")
        return respostaNormalizada(resposta)
    if indice == 11:
        resposta = input("As Universidades são um gasto público?")
        return respostaNormalizada(resposta)
    if indice == 12:
        resposta = input("A favor do foro privilegiado?")
        return respostaNormalizada(resposta)
    if indice == 13:
        resposta = input("A favor de uma reforma política?")
        return respostaNormalizada(resposta)
    if indice == 14:
        resposta = input("A favor de uma diminuição de regalias para os políticos?")
        return respostaNormalizada(resposta)
    
def main():
    resultado = []
    #Fazer todas as perguntas
    for indice in range(len(direitomero)):
        prov = pergunta(indice)
        if prov == -1:
            break
        resultado.append(prov)
    #Verificacao se o usuario eh de esquerda ou de direita
    resultado = np.array(resultado)
    resultado = resultado.reshape(1,-1)
    CentroideResultado = kmeans.predict(resultado)
    if  CentroideResultado == centroideDireita:
        print("Voce eh relativamente de direita!!")
    elif CentroideResultado == centroideEsquerda:
        print("Voce eh relativamente de esquerda!!")

main()
print("FIM!")