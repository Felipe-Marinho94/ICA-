rm(list = ls())
#Perceptron Simples
#author: Felipe Pinto Marinho
#Data: 05/05/2022

#Carregamento de alguma bibliotecas relevantes
library(pracma)
library(MASS)
library(ggplot2)
library(ggthemes)
library(ggforce)
library(plotly)
library(ISLR)
library(class)
library(caret)
library(ks)

#Defini��o de algumas fun��es
#Fun��o degrau
degrau = function(u){
  if (u >= 0){
    return(1)
  }
  
  if (u < 0){
    return(0)
  }
}

#Fun��o para calcular a sa�da
sa�da = function(x, w){
  u = dot(as.numeric(w), as.numeric(x))
  y = degrau(u)
  
  return(y)
}

#Fun��o para normaliza��o dos dados
normaliza = function(D){
  D_normalizado = apply(D, 2, scale)
  return(D_normalizado)
}


#Algoritmo Least Mean Squares (LMS)
LMS = function(w_0, alpha, d, x){
  
  #Inicializa��o do vetor de pesos
  w = w_0

  #Obten��o do vetor de erro
  u = dot(as.numeric(w), as.numeric(x))
  y = degrau(u)
  e = d - y
    
  #Atualiza��o
  w = w + alpha*e*x
  
  return(w)
}


#Fun��o para treinamento do Perc�ptron simples
treino_PS = function(Ne, X_treino, d_treino){
  
  #Inicializa��o aleat�ria dos pesos
  w_0 = rnorm(ncol(X_treino), mean = 0, sd = 1)
  w = w_0
  
  #Treinamento ao longo das �pocas
  for (j in 1:Ne){
    
    #Permuta��o das linhas
    permut = sample(1:nrow(X_treino), nrow(X_treino))
    
    #Apresenta��o dos padr�es de treino
    for (i in 1:nrow(X_treino)) {
      X_treino_aleatorio = X_treino[permut, ]
      d_treino_aleatorio = d_treino[permut]
      
      #atualiza��o
      w = LMS(w, 0.2, d_treino_aleatorio[i], X_treino_aleatorio[i,])
      
    }
  }
  
  return(w)
}

#Carregando um dataset
fix(Smarket)

#Pr�-processamento
Smarket$Direction = as.numeric(Smarket$Direction)
I1 = which(Smarket$Direction == 1)
I2 = which(Smarket$Direction == 2)
Smarket[I1, ncol(Smarket)] = 0
Smarket[I2, ncol(Smarket)] = 1

#Divis�o treino-teste
treino = sample(1:nrow(Smarket), 0.7*nrow(Smarket))
Smarket_X_treino_normalizado = normaliza(Smarket[treino, -ncol(Smarket)])
Smarket_X_teste_normalizado = normaliza(Smarket[-treino, -ncol(Smarket)])
alvo_treino = Smarket$Direction[treino]
alvo_teste = Smarket$Direction[-treino]


#Treino do PS
w_opt = treino_PS(50, Smarket_X_treino_normalizado, alvo_treino)

#Avalia��o no teste
estimativa = rep(0, nrow(Smarket_X_teste_normalizado))
for (j in 1:nrow(Smarket_X_teste_normalizado)) {
  estimativa[j] = sa�da(Smarket_X_teste_normalizado[j,], w_opt)
}

mean(estimativa == alvo_teste)
table(estimativa, alvo_teste)


acerto_PS = rep(0, 100)
confus�o_PS = array(0, dim = c(2, 2, 100))

acerto_Logistica = rep(0, 100)
confus�o_Logistica = array(0, dim = c(2, 2, 100))

acerto_LDA = rep(0, 100)
confus�o_LDA = array(0, dim = c(2, 2, 100))

acerto_KNN = rep(0, 100)
confus�o_KNN = array(0, dim = c(2, 2, 100))

Smarket_treino_normalizado = as.data.frame(cbind(Smarket_X_treino_normalizado, alvo_treino))
Logistica = glm(Smarket_treino_normalizado$alvo_treino~., data = Smarket_treino_normalizado, family = "binomial")
fix(Smarket_treino_normalizado)
attach(Smarket)

#Realiza��o do Hold-out
for (k in 1:10){
  
  #Divis�o treino-teste
  treino = sample(1:nrow(Smarket), 0.7*nrow(Smarket))
  Smarket_X_treino_normalizado =  as.data.frame(normaliza(Smarket[treino, -ncol(Smarket)]))
  Smarket_X_teste_normalizado = as.data.frame(normaliza(Smarket[-treino, -ncol(Smarket)]))
  #Smarket_treino_normalizado = as.data.frame(cbind(Smarket_X_treino_normalizado, alvo_treino))
  #Smarket_teste_normalizado = as.data.frame(cbind(Smarket_X_teste_normalizado, alvo_teste))
  alvo_treino = Smarket$Direction[treino]
  alvo_teste = Smarket$Direction[-treino]
  
  #Treino do PS
  w_opt = treino_PS(100, Smarket_X_treino_normalizado, alvo_treino)
  
  #Treino da Regressao Log�stica
  #Logistica = glm(Smarket_treino_normalizado$alvo_treino~., data = Smarket_treino_normalizado, family = "binomial")
  
  #Treino LDA
  #LDA = lda(Smarket_treino_normalizado$alvo_treino~., data = Smarket_treino_normalizado)
  
  
    
  #Avalia��o no teste
  estimativa_PS = rep(0, nrow(Smarket_X_teste_normalizado))
  #estimativa_Logistica = rep(0, nrow(Smarket_X_teste_normalizado))
  #estimativa_LDA = rep(0, nrow(Smarket_X_teste_normalizado))
  estimativa_KNN = rep(0, nrow(Smarket_X_teste_normalizado))
  
  #Estmativa PS
  for (j in 1:nrow(Smarket_X_teste_normalizado)) {
    estimativa[j] = sa�da(Smarket_X_teste_normalizado[j,], w_opt)
  }
  
  #Estimativa Log�stica
  #glm_pred = predict(Logistica, Smarket_teste_normalizado, type = "response")
  #estimativa_Logistica = rep(0, nrow(Smarket_teste_normalizado))
  #estimativa_Logistica[glm_pred > .5] = 1
  
  #Estimativa LDA
  #LDA_pred = predict(LDA,Smarket_teste_normalizado)
  #estimativa_LDA = LDA_pred$class
  
  #Estimativa KNN
  estimativa_KNN = knn(Smarket_X_treino_normalizado, Smarket_X_teste_normalizado, alvo_treino, k = k)
  
  #Desempenho PS
  acerto_PS[k] = mean(estimativa == alvo_teste)
  confus�o_PS[, , k] = table(estimativa, alvo_teste)
  
  #Desempenho Logistica
  #acerto_Logistica[k] = mean(estimativa_Logistica == alvo_teste)
  #confus�o_Logistica[, , k] = table(estimativa_Logistica, alvo_teste)
  
  #Desempenho LDA
  #acerto_LDA[k] = mean(estimativa_LDA == alvo_teste)
  #confus�o_LDA[, , k] = table(estimativa_LDA, alvo_teste)
  
  #Desempenho Logistica
  acerto_KNN[k] = mean(estimativa_KNN == alvo_teste)
  confus�o_KNN[, , k] = table(estimativa_KNN, alvo_teste)
}

acerto_LDA
acerto_KNN
acerto_Logistica
acerto_PS
length(acerto_PS)
par(mfrow = c(2,2))
plot(x = 1:51, y = acerto_PS[1:51], type = "b", lwd = 3, cex.lab = 1.8, cex.axis = 1.8, cex.main = 1.8, col = "darkgreen", xlab = "Realiza��es", ylab = "Taxa de Acerto", cex = 1.3, main = "Taxa de acerto para PS")
points(x = which.max(acerto_PS), y = acerto_PS[which.max(acerto_PS)], cex = 2, col = "red", pch = 18)

plot(x = 1:51, y = acerto_Logistica[1:51], type = "b", lwd = 3, cex.lab = 1.8, cex.axis = 1.8, cex.main = 1.8, col = "grey", xlab = "Realiza��es", ylab = "Taxa de Acerto", cex = 1.3, main = "Taxa de acerto para Log�stica")
points(x = which.max(acerto_Logistica), y = acerto_Logistica[which.max(acerto_Logistica)], cex = 2, col = "red", pch = 18)

plot(x = 1:51, y = acerto_LDA[1:51], type = "b", lwd = 3, cex.lab = 1.8, cex.axis = 1.8, cex.main = 1.8, col = "darkblue", xlab = "Realiza��es", ylab = "Taxa de Acerto", cex = 1.3, main = "Taxa de acerto para o LDA")
points(x = which.max(acerto_LDA), y = acerto_LDA[which.max(acerto_LDA)], cex = 2, col = "red", pch = 18)

plot(x = 1:51, y = acerto_KNN[1:51], type = "b", lwd = 3, cex.lab = 1.8, cex.axis = 1.8, cex.main = 1.8, col = "purple", xlab = "Realiza��es", ylab = "Taxa de Acerto", cex = 1.3, main = "Taxa de acerto para PS")
points(x = which.max(acerto_KNN), y = acerto_KNN[which.max(acerto_KNN)], cex = 2, col = "red", pch = 18)


confus�o_PS[, , which.max(acerto_PS)]
confus�o_PS[, , which.min(acerto_PS[1:51])]
confus�o_Logistica[, , which.max(acerto_Logistica[1:51])]
confus�o_Logistica[, , which.min(acerto_Logistica[1:51])]

#MATRIZ DE CONFUS�O para PS
par(mfrow = c(1, 2))

#Melhor caso
ctable = as.table(matrix(c(191, 1, 0, 183), nrow = 2, byrow = TRUE))
colnames(ctable) = c("Real:Eleva��o", "Real;Queda")
rownames(ctable) = c("Predi��o:Eleva��o", "Predi��o;Queda")
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Matriz de confus�o PS: Melhor caso")

#Pior caso
ctable = as.table(matrix(c(175, 0, 22, 178), nrow = 2, byrow = TRUE))
colnames(ctable) = c("Real:Eleva��o", "Real;Queda")
rownames(ctable) = c("Predi��o:Eleva��o", "Predi��o;Queda")
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Matriz de confus�o PS: Pior caso")

#MATRIZ DE CONFUS�O para lOG�STICA
par(mfrow = c(1, 2))

#Melhor caso
ctable = as.table(matrix(c(88, 98, 2, 187), nrow = 2, byrow = TRUE))
colnames(ctable) = c("Real:Eleva��o", "Real;Queda")
rownames(ctable) = c("Predi��o:Eleva��o", "Predi��o;Queda")
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Matriz de confus�o Log�.: Melhor caso")

#Pior caso
ctable = as.table(matrix(c(21, 158, 76, 120), nrow = 2, byrow = TRUE))
colnames(ctable) = c("Real:Eleva��o", "Real;Queda")
rownames(ctable) = c("Predi��o:Eleva��o", "Predi��o;Queda")
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Matriz de confus�o Log�.: Pior caso")

confus�o_LDA[, , which.max(acerto_LDA[1:51])]
confus�o_LDA[, , which.min(acerto_LDA[1:51])]

#MATRIZ DE CONFUS�O para LDA
par(mfrow = c(1, 2))

#Melhor caso
ctable = as.table(matrix(c(88, 98, 2, 187), nrow = 2, byrow = TRUE))
colnames(ctable) = c("Real:Eleva��o", "Real;Queda")
rownames(ctable) = c("Predi��o:Eleva��o", "Predi��o;Queda")
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Matriz de confus�o LDA: Melhor caso")

#Pior caso
ctable = as.table(matrix(c(35, 149, 102, 89), nrow = 2, byrow = TRUE))
colnames(ctable) = c("Real:Eleva��o", "Real;Queda")
rownames(ctable) = c("Predi��o:Eleva��o", "Predi��o;Queda")
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Matriz de confus�o LDA: Pior caso")


confus�o_KNN[, , which.max(acerto_KNN[1:51])]
confus�o_KNN[, , which.min(acerto_KNN[1:51])]

#MATRIZ DE CONFUS�O para LDA
par(mfrow = c(1, 2))

#Melhor caso
ctable = as.table(matrix(c(164, 17, 6, 188), nrow = 2, byrow = TRUE))
colnames(ctable) = c("Real:Eleva��o", "Real;Queda")
rownames(ctable) = c("Predi��o:Eleva��o", "Predi��o;Queda")
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Matriz de confus�o KNN: Melhor caso")

#Pior caso
ctable = as.table(matrix(c(147, 37, 37, 154), nrow = 2, byrow = TRUE))
colnames(ctable) = c("Real:Eleva��o", "Real;Queda")
rownames(ctable) = c("Predi��o:Eleva��o", "Predi��o;Queda")
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Matriz de confus�o KNN: Pior caso")

#Avalia��o gr�fica
conjunto = rep(0, nrow(Smarket))
conjunto[treino] = "Treino"
conjunto[-treino] = "Teste"
Dados = cbind(Smarket$Direction, conjunto)
Dados = cbind(Smarket, Dados)
Dados = Dados[, -10]
fix(Dados)
attach(Dados)

fig = plot_ly(data = Dados, x = ~Lag1, y = ~Today, type = 'scatter', mode = 'markers',alpha = 0.5, symbol = ~Direction, symbols = c('square','circle'),
               
               marker = list(size = 12,
                             
                             color = 'darkcyan',
                             
                             line = list(color = 'black',width = 1))) %>% layout(xaxis = list(titlefont = list(size = 18), tickfont = list(size = 18), title = "Lag 1"),
                                                                                 yaxis = list(titlefont = list(size = 18), tickfont = list(size = 18), title = "Retorno atual"))
fig

#Gr�fico mais complicado
mesh_size = .02

margin = 0.25

x_min =  min(Dados$Lag1) - margin

x_max = max(Dados$Lag1) + margin

y_min = min(Dados$Today) - margin

y_max = max(Dados$Today) + margin

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

colnames(final) = c('Lag1','Atual')

predicted = rep(0, nrow(final))
for (i in 1:nrow(final)) {
  predicted[i] = sa�da(as.numeric(final[i,]), w_opt[c(2,8)])
}


Z = matrix(predicted, dim_val[1], dim_val[2])



fig1 = plot_ly(symbols = c('square','circle'))%>%
  
  add_trace(x = xrange, y= yrange, z = Z, colorscale='RdBu', type = "contour", opacity = 0.5) %>%
  
  add_trace(data = Dados, x = ~Lag1, y = ~Today, type = 'scatter', mode = 'markers', symbol = ~Direction ,
            
            marker = list(size = 12,
                          
                          color = 'lightyellow',
                          
                          line = list(color = 'black',width = 1))) %>% layout(xaxis = list(titlefont = list(size = 18), tickfont = list(size = 18), title = "Lag 1"),
                                                                                       yaxis = list(titlefont = list(size = 18), tickfont = list(size = 18), title = "Retorno atual"), title = "Fronteira de decis�o para o PS")

fig1

#Fun��o para calcular as estat�sticas
STATS = function(y){
  media = mean(y)
  print("a m�dia �:")
  print(media)
  
  mediana = median(y)
  print("a mediana �:")
  print(mediana)
  
  maximo = y[which.max(y)]
  print("o m�ximo �:")
  print(maximo)
  
  minimo = y[which.min(y)]
  print("o m�nimo �:")
  print(minimo)
  
  desvio = sd(y)
  print("o desvio padr�o �:")
  print(desvio)
}

STATS(acerto_PS[1:51])
STATS(acerto_Logistica[1:51])
STATS(acerto_LDA[1:51])
STATS(acerto_KNN[1:51])


#Box plot
acuracia = cbind(acerto_PS[1:51], acerto_Logistica[1:51], acerto_LDA[1:51], acerto_KNN[1:51])
acuracia = vec(acuracia)
fix(acuracia)
Modelos = rep(0, 51*4)
Modelos[1:51] = "PS"
Modelos[52:102] = "Log�stica"
Modelos[103:153] = "LDA"
Modelos[153:204] = "KNN"
acuracia = cbind(acuracia, Modelos)
acuracia = as.data.frame(acuracia)
fix(acuracia)
attach(acuracia)
ggplot(acuracia, aes(x = Modelos, y = Acerto, fill = Modelos )) + geom_boxplot()+
  labs(title = "Taxa de acerto ao longo das realiza��es", x = "Modelos", y = "Taxa de Acerto") + theme_minimal() +
  theme(axis.title=element_text(size=14,face="bold"), plot.title = element_text(size=14,face="bold"), axis.text = element_text(size = 16)) + scale_fill_brewer(palette="RdBu")