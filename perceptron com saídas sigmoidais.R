#Inteligência Computacional Aplicada
#Trabalho 4 - Redes Neurais Artificiais
#Autor: Felipe Pinto Marinho
#Data: 07/07/2022

#Carregando algumas bibliotecas relevantes
library(ggplot2)
library(plotly)
library(pracma)
library(MASS)
library(ks)

#Implmentando algumas funções
Logistica = function(u){
  return(1/(1 + exp(-u)))
}

derivada = function(u){
  return(Logistica(u) * (1 - Logistica(u)))
}

##Função para normalização dos dados
normaliza = function(D){
  D_normalizado = apply(D, 2, scale)
  return(D_normalizado)
}

##Função para determinar a saída da rede
Saída_Net = function(x, W_opt){
  u = W_opt %*% x
  u = apply(u, 2, Logistica)
  y = rep(0, nrow(W_opt))
  y[which.max(u)] = 1
  
  return(y)
}

##Regra delta para atualização dos pesos
LMS = function(w_0, alpha, d, x){
  
  #Inicialização do vetor de pesos
  w = w_0
  
  #Obtenção do vetor de erro
  u = dot(as.numeric(w), as.numeric(x))
  y = Logistica(u)
  phi = derivada(u)
  e = d - y
  
  #Atualização
  w = w + alpha*e*x*phi
  
  #Criação de lista
  resultados = list(w, e^2)
  names(resultados) = c("w", "e")
  return(resultados)
}

##Função para treinamento do Perceptron simples
treino_PS = function(Ne, X_treino, d_treino){
  
  #Inicialização aleatória dos pesos
  w_0 = rnorm(ncol(X_treino), mean = 0, sd = 1)
  w = w_0
  soma = rep(0, Ne)
  
  #Treinamento ao longo das épocas
  for (j in 1:Ne){
    
    #Inicialização do erro
    erro = 0
    
    #Permutação das linhas
    permut = sample(1:nrow(X_treino), nrow(X_treino))
    
    #Apresentação dos padrões de treino
    for (i in 1:nrow(X_treino)) {
      X_treino_aleatorio = X_treino[permut, ]
      d_treino_aleatorio = d_treino[permut]
      
      #atualização
      Resultados = LMS(w, 0.2, d_treino_aleatorio[i], X_treino_aleatorio[i,])
      w = as.numeric(Resultados$w)
      er = Resultados$e
      erro = cbind(erro, er)
    }
    
    #Soma dos erros
    soma[j] = sum(erro)
  }
  
  #criação de lista
  resultados_PS = list(w, soma)
  names(resultados_PS) = c("w", "soma")
  
  return(resultados_PS)
}

##Função para o treinamento da rede
Treino_Net = function(Ne, X_treino, D_treino, p, c){
  
  #Criando uma matriz para armazenar os pesos ótimos para cada camada
  W_opt = matrix(0, nrow = c, ncol = p)
  
  #Treinando cada neurônio na camada de saída
  for (k in 1:c) {
    W_opt[k, ] = treino_PS(Ne, X_treino, D_treino[k, ])$w
  }
  
  return(W_opt)
}

#Função para calcular as estatísticas
STATS = function(y){
  media = mean(y)
  print("a média é:")
  print(media)
  
  mediana = median(y)
  print("a mediana é:")
  print(mediana)
  
  maximo = y[which.max(y)]
  print("o máximo é:")
  print(maximo)
  
  minimo = y[which.min(y)]
  print("o mínimo é:")
  print(minimo)
  
  desvio = sd(y)
  print("o desvio padrão é:")
  print(desvio)
}

###########################################
#Conjunto de dados Iris
###Construindo a matriz de Rótulos para o conjunto Iris
names(iris) = c(c("sepal.length", "sepal.width", "petal.length", "petal.width", "specie"))
Y_iris = matrix(0, nrow = 3, ncol = nrow(iris))
J1 = which(iris$specie == "Iris-setosa")
J2 = which(iris$specie == "Iris-versicolor")
J3 = which(iris$specie == "Iris-virginica") 

Y_iris[, J1] = c(1, 0, 0)
Y_iris[, J2] = c(0, 1, 0)
Y_iris[, J3] = c(0, 0, 1)

#Apenas um pequeno teste para avaliar o código desenvolvido
set.seed(2)
treino = sample(1:nrow(iris), 0.7*nrow(iris))
iris_X_treino_normalizado = as.data.frame(normaliza(iris[treino, -ncol(iris)]))
iris_X_treino_normalizado = cbind(iris_X_treino_normalizado, rep(-1, nrow(iris_X_treino_normalizado)))

iris_X_teste_normalizado = as.data.frame(normaliza(iris[-treino, -ncol(iris)]))
iris_X_teste_normalizado = cbind(iris_X_teste_normalizado, rep(-1, nrow(iris_X_teste_normalizado)))
alvo_treino = Y_iris[, treino]
alvo_teste = Y_iris[, -treino]
Y_iris

#treino da rede
W_opt = Treino_Net(100, iris_X_treino_normalizado, alvo_treino, 5, 3)

#teste da rede
estimativa = matrix(0, nrow = 3, ncol = nrow(iris_X_teste_normalizado))
for (j in 1:nrow(iris_X_teste_normalizado)) {
  estimativa[, j] = Saída_Net(as.numeric(iris_X_teste_normalizado[j,]), W_opt)
}

vetor_estimativa = rep(0, ncol(estimativa))
for (j in 1:ncol(estimativa)) {
  vetor_estimativa[j] = which.max(estimativa[, j])
}

vetor_teste = rep(0, ncol(estimativa))
for (j in 1:ncol(estimativa)) {
  vetor_teste[j] = which.max(alvo_teste[, j])
}
mean(vetor_teste == vetor_estimativa)
table(vetor_estimativa, vetor_teste)

################################################
acerto_Iris = rep(0, 20)
confusao_Iris = list()

#Realização do Hold-Out
for (k in 1:20) {
  
  #Divisão treino-teste
  treino = sample(1:nrow(iris), 0.7 * nrow(iris))
  iris_X_treino_normalizado = as.data.frame(normaliza(iris[treino, -ncol(iris)]))
  iris_X_treino_normalizado = cbind(iris_X_treino_normalizado, rep(-1, nrow(iris_X_treino_normalizado)))
  
  iris_X_teste_normalizado = as.data.frame(normaliza(iris[-treino, -ncol(iris)]))
  iris_X_teste_normalizado = cbind(iris_X_teste_normalizado, rep(-1, nrow(iris_X_teste_normalizado)))
  alvo_treino = Y_iris[, treino]
  alvo_teste = Y_iris[, -treino]
  
  #Treino da ajuste
  W_opt = Treino_Net(100, iris_X_treino_normalizado, alvo_treino, 5, 3)
  
  #Avaliação no teste
  estimativa = matrix(0, nrow = 3, ncol = nrow(iris_X_teste_normalizado))
  for (j in 1:nrow(iris_X_teste_normalizado)) {
    estimativa[, j] = Saída_Net(as.numeric(iris_X_teste_normalizado[j,]), W_opt)
  }
  
  vetor_estimativa = rep(0, ncol(estimativa))
  for (j in 1:ncol(estimativa)) {
    vetor_estimativa[j] = which.max(estimativa[, j])
  }
  
  vetor_teste = rep(0, ncol(estimativa))
  for (j in 1:ncol(estimativa)) {
    vetor_teste[j] = which.max(alvo_teste[, j])
  }
  acerto_Iris[k] = mean(vetor_teste == vetor_estimativa)
  confusao_Iris = append(confusao_Iris, table(vetor_estimativa, vetor_teste))
}


########################################################
###Construindo a matriz de Rótulos para o conjunto Artificial
artificial = matrix(rnorm(300), nrow = 150)
rotulos = cbind(rep("circulo", 50), rep("triângulo", 50), rep("estrela", 50))
rotulos = vec(rotulos)
rotulos
artificial = cbind(artificial, rotulos)
artificial = as.data.frame(artificial)
names(artificial)


Y_artificial = matrix(0, nrow = 3, ncol = nrow(artificial))
J1 = which(artificial$rotulos == "circulo")
J2 = which(artificial$rotulos == "triângulo")
J3 = which(artificial$rotulos == "estrela")

Y_artificial[, J1] = c(1, 0, 0)
Y_artificial[, J2] = c(0, 1, 0)
Y_artificial[, J3] = c(0, 0, 1)


sum(is.na(artificial))
fix(artificial)
names(artificial)
dim(artificial)

acerto_artificial = rep(0, 20)
confusao_artificial = list()

#Realização do Hold-Out
for (k in 1:20) {
  
  #Divisão treino-teste
  treino = sample(1:nrow(artificial), 0.7 * nrow(artificial))
  artificial_X_treino_normalizado = as.data.frame(normaliza(artificial[treino, -ncol(artificial)]))
  artificial_X_treino_normalizado = cbind(artificial_X_treino_normalizado, rep(-1, nrow(artificial_X_treino_normalizado)))
  
  artificial_X_teste_normalizado = as.data.frame(normaliza(artificial[-treino, -ncol(artificial)]))
  artificial_X_teste_normalizado = cbind(artificial_X_teste_normalizado, rep(-1, nrow(artificial_X_teste_normalizado)))
  alvo_treino = Y_artificial[, treino]
  alvo_teste = Y_artificial[, -treino]
  
  #Treino da ajuste
  W_opt = Treino_Net(100, artificial_X_treino_normalizado, alvo_treino, ncol(artificial_X_treino_normalizado), 3)
  
  #Avaliação no teste
  estimativa = matrix(0, nrow = 3, ncol = nrow(artificial_X_teste_normalizado))
  for (j in 1:nrow(artificial_X_teste_normalizado)) {
    estimativa[, j] = Saída_Net(as.numeric(artificial_X_teste_normalizado[j, ]), W_opt)
  }
  
  vetor_estimativa = rep(0, ncol(estimativa))
  for (j in 1:ncol(estimativa)) {
    vetor_estimativa[j] = which.max(estimativa[, j])
  }
  
  vetor_teste = rep(0, ncol(estimativa))
  for (j in 1:ncol(estimativa)) {
    vetor_teste[j] = which.max(alvo_teste[, j])
  }
  
  acerto_artificial[k] = mean(vetor_teste == vetor_estimativa)
  confusao_artificial = append(confusao_artificial, table(vetor_estimativa, vetor_teste)) 
}

acerto_artificial

#Construção de gráficos
Valores = cbind(acerto_Iris, acerto_artificial)
Valores = as.data.frame(Valores)
figura = plot_ly(data = Valores, x = 1:20, y = Valores[, 1], type = "scatter", marker = list(size = 10, color = "rgba(125, 180, 240, .9)", line = list(color = "rgba(0, 152, 0, .8)", width = 2)), name = NULL) 
figura = figura %>% layout(xaxis = list(title = "Realizações", titlefont = list(size = 22), tickfont = list(size = 22)), yaxis = list(title = "Taxa de acerto", titlefont = list(size = 22), tickfont = list(size = 22))) %>% add_lines(x = 1:20, y = Valores[, 1], name = 'Iris', line = list(width = 4))
figura = figura %>% add_trace(x = 1:20, y = Valores[, 2], name = "Artificial", line = list(color = "red", width = 4), marker = list(size = 10, color = "rgba(240, 18, 24, .9)", line = list(color = "rgba(152, 0, 0, .8)", width = 2), symbol = 'x'))
figura

#Fronteira de decisão
#Gráfico mais complicado
conjunto = rep(0, nrow(iris))
conjunto[treino] = "Treino"
conjunto[-treino] = "Teste"
length(iris$specie)
Dados = cbind(iris$specie, conjunto)
Dados = cbind(iris, Dados)
Dados = Dados[, -6]
fix(Dados)
attach(Dados)

mesh_size = .02

margin = 0.25

x_min =  min(iris$petal.length) - margin

x_max = max(iris$petal.length) + margin

y_min = min(iris$petal.width) - margin

y_max = max(iris$petal.width) + margin

xrange = seq(x_min, x_max, mesh_size)

yrange = seq(y_min, y_max, mesh_size)

xy = meshgrid(x = xrange, y = yrange)

xx = xy$X

yy = xy$Y

#####################################
dim_val = dim(xx)

xx1 = matrix(xx, length(xx), 1)

yy1 = matrix(yy, length(yy), 1)

final = data.frame(xx1, yy1)
fix(final)

colnames(final) = c('petal.length','petal.width')

final_normalizado = as.data.frame(normaliza(final))
fix(final_normalizado)
predicted = matrix(0, nrow = 3, ncol = nrow(final_normalizado))
final_normalizado[1, ]


for (i in 1:nrow(final)) {
  predicted[, i] = Saída_Net(as.numeric(final_normalizado[i,]), W_opt[, c(3,4)])
}

vetor_estimativa = rep(0, ncol(predicted))
for (j in 1:ncol(predicted)) {
  vetor_estimativa[j] = which.max(predicted[, j])
}

Z = matrix(vetor_estimativa, dim_val[1], dim_val[2])
Z


fig1 = plot_ly(symbols = c('square','circle', 'star'))%>%
  
  add_trace(x = xrange, y= yrange, z = Z, colorscale='RdBu', type = "contour", opacity = 0.5) %>%
  
  add_trace(data = Dados, x = ~petal.length, y = ~petal.width, type = 'scatter', mode = 'markers', symbol = ~specie ,
            
            marker = list(size = 12,
                          
                          color = 'lightyellow',
                          
                          line = list(color = 'black',width = 1))) %>% layout(xaxis = list(titlefont = list(size = 18), tickfont = list(size = 18), title = "petal.length"),
                                                                              yaxis = list(titlefont = list(size = 18), tickfont = list(size = 18), title = "petal.width"), title = "Fronteira de decisão")

fig1

##Conjunto Artificial
treino = sample(1:nrow(artifcial), 0.7*nrow(artificial))
artificial_X_treino_normalizado = as.data.frame(normaliza(artificial[treino, -ncol(artificial)]))
artificial_X_treino_normalizado = cbind(artificial_X_treino_normalizado, rep(-1, nrow(artificial_X_treino_normalizado)))
alvo_treino = Y_artificial[, treino]
W_opt = Treino_Net(100, artificial_X_treino_normalizado, alvo_treino, 3, 3)
conjunto_artificial = rep(0, nrow(artificial))
conjunto_artificial[treino] = "Treino"
conjunto_artificial[-treino] = "Teste"
Dados_artificial = cbind(artificial$rotulos, conjunto_artificial)
Dados_artificial = cbind(artificial, Dados_artificial)
Dados_artificial = Dados_artificial[, -4]
Dados_artificial = as.data.frame(Dados_artificial)
fix(Dados_artificial)
attach(Dados_artificial)

mesh_size = .02

margin = 0.25

x_min =  min(artificial$V1) - margin

x_max = max(artificial$V1) + margin

y_min = min(artificial$V2) - margin

y_max = max(artificial$V2) + margin

xrange = seq(x_min, x_max, mesh_size)

yrange = seq(y_min, y_max, mesh_size)

xy = meshgrid(x = xrange, y = yrange)

xx = xy$X

yy = xy$Y

#####################################
dim_val = dim(xx)

xx1 = matrix(xx, length(xx), 1)

yy1 = matrix(yy, length(yy), 1)

final = data.frame(xx1, yy1)
fix(final)

colnames(final) = c('V1','V2')

final_normalizado = as.data.frame(normaliza(final))
fix(final_normalizado)
predicted = matrix(0, nrow = 3, ncol = nrow(final_normalizado))
final_normalizado[1, ]


for (i in 1:nrow(final)) {
  predicted[, i] = Saída_Net(as.numeric(final_normalizado[i,]), W_opt[, c(1,2)])
}

vetor_estimativa = rep(0, ncol(predicted))
for (j in 1:ncol(predicted)) {
  vetor_estimativa[j] = which.max(predicted[, j])
}

Z = matrix(vetor_estimativa, dim_val[1], dim_val[2])
Z


fig2 = plot_ly(symbols = c('square','circle', 'star'))%>%
  
  add_trace(x = xrange, y= yrange, z = Z, colorscale='RdBu', type = "contour", opacity = 0.5) %>%
  
  add_trace(data = Dados_artificial, x = ~V1, y = ~V2, type = 'scatter', mode = 'markers', symbol = ~rotulos,
            
            marker = list(size = 12,
                          
                          color = 'lightyellow',
                          
                          line = list(color = 'black',width = 1))) %>% layout(xaxis = list(titlefont = list(size = 18), tickfont = list(size = 18), title = "V1"),
                                                                              yaxis = list(titlefont = list(size = 18), tickfont = list(size = 18), title = "V2"), title = "Fronteira de decisão")

fig2

STATS(acerto_Iris)
STATS(acerto_artificial)
