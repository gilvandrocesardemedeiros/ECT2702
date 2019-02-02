"""
@author: Gilvandro Cesar de Medeiros
Problem: Dada uma rede do tipo Perceptron formada por um neurônio com três terminais de entrada
utilizando pesos iniciais w0 = 0.4, w1 = -0.6 e w2 = 0.6, limiar θ = 0.5 e uma taxa de aprendizado = 0.4. 
Considere que o limiar será sempre multiplicado por -1. responda os itens abaixo:
Ensinar a rede a gerar a saída -1 para o padrão 001 e a saída +1 para os padrão 110;
A que classe pertencem os padrões 111, 000, 100 e 011?
"""

#Pesos e valores de constantes iniciais
def pesos():
    return [0.4, -0.6, 0.6]
def limiar():
    return 0.5
def taxaAprend():
    return 0.4

#Leitura para os dados (int) retornando um vetor de int
def leitura():
    while(True):
        x = int(input("Entre com a sequencia de 3 digitos inteiros para fazer a previsao: \n"))
        num_valido = (x == 0) or (x == 1) or (x == 10) or (x == 11) or (x == 100) or (x == 101) or (x == 110) or (x == 111)
        if num_valido:
            if x < 10:
                return [0,0,x]
            if x < 100:
                return [0, int(x / 10), x % 10]
            else:
                return [1, int(x / 10) - 10, x % 10]
        else:
            print("Numero invalido!\n")
            continue

#Funcao para classificar os padroes
def classificador(x, w, limiar, taxaAprend):
    saida = 0
    for i in range(len(x)):
        saida += x[i]*w[i]
    saida -= limiar
    if saida >= 0:
        return 1
    else:
        return -1

def atualizarPesos(w, x, taxaAprend, d, y):
    for i in range(len(w)):
        w[i] = w[i] + taxaAprend*x[i]*(d-y)
    return w
        
def valorDesejado(x):
    if x == [0, 0, 1]:
        return -1
    if x == [1, 1, 0]:
        return 1
    else:
        return 0

#Determinacao das variaveis de treino para a rede
w = pesos()
limiar = limiar()
taxaAprend = taxaAprend()

#Bloco de codigo principal
continuar = True
while(continuar):
    x = leitura()
    d = valorDesejado(x)
    y = classificador(x, w, limiar, taxaAprend)
    w = atualizarPesos(w, x, taxaAprend, d, y)
    print("O resultado foi: ")
    print(y)
    continuar = int(input("Deseja continuar? Se sim, digite 1. Se não, digite 0: \n"))
    
print("Programa encerrado!")
