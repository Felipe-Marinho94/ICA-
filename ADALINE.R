rm(list = ls())
#Inteligência Computacional
#Trabalho 2 - Redes Neurais
#Data: 01/07/2022
#Autor: Felipe Pinto Marinho

#Carregando alguma bibliotecas relevantes
library(caret)
library(ggplot2)
library(plotly)
library(MASS)
library(pracma)

#Modelo ADALINE
#Construção de algumas funções importantes
#Função para calcular a saída
saída = function(x, w){
  u = dot(as.numeric(w), as.numeric(x))
  y = u
  
  return(y)
}

#Função para normalização dos dados
normaliza = function(D){
  D_normalizado = apply(D, 2, scale)
  return(D_normalizado)
}

#Algoritmo Least Mean Squares (LMS)
LMS = function(w_0, alpha, d, x){
  
  #Inicialização do vetor de pesos
  w = w_0
  
  #Obtenção do vetor de erro
  y = dot(as.numeric(w), as.numeric(x))
  e = d - y
  
  #Atualização
  w = w + alpha*e*x
  
  #Criação de lista
  resultados = list(w, e^2)
  names(resultados) = c("w", "e")
  return(resultados)
}
ncol(Valores_X_treino_normalizado)
nrow(Valores_X_treino_normalizado)

#Função para treinamento da rede ADALINE
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
      X_treino_aleatorio = as.data.frame(X_treino[permut, ])
      d_treino_aleatorio = d_treino[permut]
      
      #atualização
      Resultados = LMS(w, 0.2, d_treino_aleatorio[i], X_treino_aleatorio[i,])
      w = Resultados$w
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

#Função para calcular diversas métricas de erro
métricas=function(Y_estimado,Y_real){
  MBE=mean(Y_real-Y_estimado)
  #print(MBE)
  MAE=mean(abs(Y_real-Y_estimado))
  #print(MAE)
  RMSE=sqrt(mean((Y_real-Y_estimado)^2))
  #print(RMSE)
  rRMSE=((RMSE)/mean(Y_real))*100
  #print(rRMSE)
  SSE=sum((Y_real-Y_estimado)^2)
  SSTO=sum((Y_real-mean(Y_real))^2)
  R2=1-(SSE/SSTO)
  #print(R2)
  
  erro = list(RMSE, R2)
  names(erro) = c("RMSE", "R2")
  return(erro)
  
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

##########################################################
#Geração do conjunto de dados
#Utilizando uma função afim
afim = function(x){
  
  #Setando os valores para a e b
  a = 3
  b=-20
  
  return(a*x + b)
  
}

#Laço para obtenção das saídas correspondentes
malha = seq(-100, 100, 1)
y = rep(0, length(malha))
for (i in malha) {
  y[i] = afim(malha[i])
}

#Adição de ruído gaussiano
y = y + rnorm(length(y), mean = 0, sd = 1.5)

#Representação gráfica dos dados
Valores = cbind(malha, y)
Valores = data.frame(Valores)
fix(Valores)
figura = plot_ly(data = Valores, x = Valores[,1], y = Valores[,2], type = "scatter", marker = list(size = 10, color = "rgba(125, 180, 240, .9)", line = list(color = "rgba(0, 152, 0, .8)", width = 2)), name = NULL) 
figura = figura %>% layout(xaxis = list(title = "Entradas", titlefont = list(size = 22), tickfont = list(size = 22)), yaxis = list(title = "Saída", titlefont = list(size = 22), tickfont = list(size = 22))) %>% add_lines(x = Valores[,1], y = Valores[,2], name = NULL, line = list(width = 4))
figura

#Verificação que o código está coerente
#Separação treino-teste
set.seed(2)
treino = sample(1:nrow(Valores), 0.7*nrow(Valores))
Valores_X_treino_normalizado = as.data.frame(scale(Valores$malha[treino])) 
Valores_X_teste_normalizado = as.data.frame(scale(Valores$malha[-treino]))
alvo_treino = Valores$y[treino]
alvo_teste = Valores$y[-treino]

#treino
w_opt = treino_PS(100, Valores_X_treino_normalizado, alvo_treino)$w

#Avaliação no teste
estimativa = rep(0, nrow(Valores_X_teste_normalizado))
for (j in 1:nrow(Valores_X_teste_normalizado)){
  estimativa[j] = saída(Valores_X_teste_normalizado[j,], w_opt)
}

métricas(estimativa, alvo_teste)

figura = plot_ly(data = Valores[-treino, ], x = Valores[-treino, 1], y = Valores[-treino, 2], type = "scatter", marker = list(size = 10, color = "rgba(125, 180, 240, .9)", line = list(color = "rgba(0, 152, 0, .8)", width = 2)), name = 'Saídas') 
figura = figura %>% layout(xaxis = list(title = "Entradas", titlefont = list(size = 22), tickfont = list(size = 22)), yaxis = list(title = "Saída", titlefont = list(size = 22), tickfont = list(size = 22))) %>% add_lines(x = Valores[-treino, 1], y = Valores[-treino, 2], name = 'Real', line = list(width = 4))
figura %>% add_trace(x = Valores[-treino, 1], y = estimativa, name = "Adaline", line = list(color = "black", width = 4), marker = list(size = 10, color = "rgba(12, 18, 240, .9)", line = list(color = "rgba(0, 0, 152, .8)", width = 2), symbol = 'triangle-up'))  

#Avaliação do desempenho
MSE = rep(0, 50)
RMSE = rep(0, 50)
R2 = rep(0, 50)
w_opt_vetor =rep(0, 50)

#Realização do Hold-Out

for (k in 1:50){
  
  #Divisão treino-teste
  treino = sample(1:nrow(Valores), 0.7*nrow(Valores))
  Valores_X_treino_normalizado =  as.data.frame(scale(Valores[treino, -ncol(Valores)]))
  Valores_X_teste_normalizado = as.data.frame(scale(Valores[-treino, -ncol(Valores)]))
  alvo_treino = Valores$y[treino]
  alvo_teste = Valores$y[-treino]
  
  #Treino do ADALINE
  w_opt = treino_PS(100, Valores_X_treino_normalizado, alvo_treino)$w
  w_opt_vetor[k] = w_opt
  
  #Avaliação no teste
  estimativa_AD = rep(0, nrow(Valores_X_teste_normalizado))
  
  #Estmativa ADALINE
  for (j in 1:nrow(Valores_X_teste_normalizado)) {
    estimativa_AD[j] = saída(Valores_X_teste_normalizado[j,], w_opt)
  }
  
  #Desempenho ADALINE
  RMSE[k] = métricas(estimativa_AD, alvo_teste)$RMSE
  MSE[k] = (métricas(estimativa_AD, alvo_teste)$RMSE)^2
  R2[k] = métricas(estimativa_AD, alvo_teste)$R2
  
}

#################################################
#Conjunto II
plano = function(x1, x2){
  
  #Setando os parâmetros da função
  a = 3
  b = 20
  c = 30
  return(a*x1 + b*x2 + c)
}

#Laço para criação do conjunto de dados
ruido_x1 = rnorm(200, mean = 0, sd = 1)
ruido_x2 = rnorm(200, mean = 0, sd = 1)
ruido_x3 = rnorm(200, mean = 0, sd =1)

malha_x1 = seq(-99, 100, 1)
malha_x2 = seq(-99, 100, 1)
malha_x3 = rep(1, 200)

#Adição de ruído
malha_x1 = malha_x1 + ruido_x1
malha_x2 = malha_x2 + ruido_x2
malha_x3 = malha_x3 + ruido_x3
y = rep(0, 200)

for (i in 1:length(y)){
  y[i] = plano(malha_x1[i], malha_x2[i])
}

dados = cbind(malha_x1, malha_x2, malha_x3)
dados= as.data.frame(dados)
fix(dados)
sum(is.na(dados))

#Avaliação do desempenho
MSE_II = rep(0, 50)
RMSE_II = rep(0, 50)
R2_II = rep(0, 50)
w_opt_vetor_II = matrix(0, nrow = 50, ncol = 3)

dados_X_normalizado = as.data.frame(normaliza(dados))
fix(dados_X_normalizado)
dados = cbind(dados_X_normalizado, y)
fix(dados)

#Realização do Hold-Out

for (k in 1:50){
  
  #Divisão treino-teste
  treino = sample(1:nrow(dados), 0.7*nrow(dados))
  dados_X_treino_normalizado =  dados_X_normalizado[treino, ]
  dados_X_teste_normalizado = dados_X_normalizado[-treino, ]
  alvo_treino = dados$y[treino]
  alvo_teste = dados$y[-treino]
  
  #Treino do ADALINE
  w_opt = treino_PS(100, dados_X_treino_normalizado, alvo_treino)$w
  w_opt = as.numeric(w_opt)
  w_opt_vetor_II[k, ] = w_opt
  
  #Avaliação no teste
  estimativa_AD = rep(0, nrow(dados_X_teste_normalizado))
  
  #Estmativa ADALINE
  for (j in 1:nrow(dados_X_teste_normalizado)) {
    estimativa_AD[j] = saída(dados_X_teste_normalizado[j,], w_opt)
  }
  
  #Desempenho ADALINE
  RMSE_II[k] = métricas(estimativa_AD, alvo_teste)$RMSE
  MSE_II[k] = (métricas(estimativa_AD, alvo_teste)$RMSE)^2
  R2_II[k] = métricas(estimativa_AD, alvo_teste)$R2
  
}

w_opt_sup = w_opt_vetor_II[which.min(RMSE_II), ]

#Avaliação gráfica da superfície de decisão
y_sup = matrix(0, nrow = length(malha_x1), ncol = length(malha_x2))
y_estimado = matrix(0, nrow = length(malha_x1), ncol = length(malha_x2))
for (i in 1:length(malha_x1)) {
  for (j in 1:length(malha_x2)) {
    y_sup[i, j] = plano(malha_x1[i], malha_x2[j])
    y_estimado[i, j] = saída(c(malha_x1[i], malha_x2[j], malha_x3[j]), w_opt_sup)
  }
}

par(mfrow = c(1, 2))
persp(malha_x1, malha_x2, y_sup, xlab = "x1", ylab = "x2", zlab = "y" ,  main = "Plano", phi = 45, theta = 45, col = "yellow")
persp(malha_x1, malha_x2, y_estimado, xlab = "x1", ylab = "x2", zlab = "y" ,  main = "Fronteira de decisão", phi = 45, theta = 45, col = "blue")

#Estatísticas descritivas
STATS(MSE)
STATS(RMSE)
STATS(R2)

STATS(MSE_II)
STATS(RMSE_II)
STATS(R2_II)
