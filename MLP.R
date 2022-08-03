#Inteligência Computacional Aplicada - ICA
#Redes Neurais Artificiais  - Rede Perceptron Múltiplas Camadas (MLP
#Autor: Felipe Pinto Marinho

#Carregando algumas bibliotecas
library(ggplot2)
library(plotly)
library(pracma)
library(MASS)
library(ks)
library(caret)

#Implementação de algumas funções
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

##Função para determinar a saída da rede para a tarefa de classificação
Saída_Net_Classifica = function(x, W_opt){
  u = W_opt %*% x
  u = apply(u, 2, Logistica)
  y = rep(0, nrow(W_opt))
  y[which.max(u)] = 1
  
  return(y)
}

##Função para determinar a saída da rede para a tarefa de regressão
Saída_Net_Regressão = function(x, W_opt){
  u = W_opt %*% x
  u = apply(u, 2, Logistica)
  
  return(u)
}


##Função para o treinamento da MLP
Treino_MLP_classificação = function(Ne, X_treino, D_treino, q, c){
  
  ##Passo para frente (foward)
  #Inicialização aleatória dos pesos da camada oculta
  W = as.data.frame(matrix(rnorm(ncol(X_treino) * q), nrow = q, ncol = ncol(X_treino)))
  
  #Inicialização aleatória dos pesos da camada de saída
  M = matrix(rnorm(c * q), nrow = c)
  
  #Treinamento ao longo das épocas
  for (j in 1:Ne){
    
    #Permutação das linhas
    permut = sample(1:nrow(X_treino), nrow(X_treino))
    
    for (k in 1:nrow(X_treino)) {
      X_treino_aleatorio = X_treino[permut, ]
      D_treino_aleatorio = D_treino[, permut]
      
      #Inicialização dos campo locais induzidos para os neurônios ocultos
      u_oculta = rep(0, q)
      
      #Inicializaçao das saídas da camada oculta
      y_oculta = rep(0, q)
      
      for (i in 1:q) {
        
        #Campo local induzido para q-ésimo neurônio oculto
        u_oculta[i] = dot(as.numeric(W[i, ]), as.numeric(X_treino_aleatorio[k, ]))
        
        #Saída da camada oculta para um j-ésimo padrão de entrada
        y_oculta[i] = Logistica(u_oculta[i])
        
      }
      
      #Inicialização dos campo locais induzidos para os neurônios de saída
      u_saída = rep(0, c)
      
      #Inicializaçao das saídas da camada oculta
      y_saída = rep(0, c)
      
      for (i in 1:c) {
        
        #Campo local induzido para q-ésimo neurônio oculto
        u_saída[i] = dot(as.numeric(M[i,]), as.numeric(y_oculta))
        
        #Saída da camada oculta para um j-ésimo padrão de entrada
        y_saída[i] = Logistica(u_saída[i])
        
      }
      
      y_saída_final = rep(0, c)
      y_saída_final[which.max(y_saída)] = 1
      
      #Determinação do erro
      erro = D_treino_aleatorio[, k] - y_saída_final
      
      #Incialização dos deltas
      delta = rep(0, c)
      
      #Etapa de backpropagation
      for (i in 1:c) {
        #Atuaização dos pesos da camada de saída
        delta[i] =  erro[i] * derivada(u_saída[i])
        M[i, ] = M[i, ] + 0.2 * delta[i] * y_oculta
      }
      
      for (i in 1:q) {
        
        #Atuaização dos pesos da camada oculta
        W[i, ] = W[i, ] + 0.2 * derivada(u_oculta[i]) * dot(as.numeric(delta), 
                                                            as.numeric(M[, i])) * X_treino_aleatorio[k, ]
      }
      
    }
    
  }
  
  resultado = list(as.matrix(W), as.matrix(M))
  names(resultado) = c("W", "M")
  return(resultado)
}

resultado = Treino_MLP_classificação(100, iris_X_treino_normalizado, alvo_treino, 2, 3)
names(resultado)
W_opt = resultado$W
M_opt = resultado$M

#Funçao para determinação da saída para a tarefa de classificação
Saída_MLP_classifica = function(W, M, x){
  
  #Campo local induzido para a primeira camada
  u_oculta = W %*% x
  
  #Ativação na camada oculta
  y_oculta = apply(u_oculta, 2, Logistica)
  
  #Campo local induzido na camada de saída
  u_saída = M %*% y_oculta
  
  #Ativação na camada de saída
  y_saida = apply(u_saída, 2, Logistica)
  
  #Determinação da saída
  y = rep(0, nrow(M))
  y[which.max(y_saida)] = 1
  
  return(y)
}


#Função para realização de grid search com validação cruzada
valida = function(K, malha, X_treino, D_treino){
  
  #Particionamento do conjunto em k folds
  index = createFolds(1:nrow(X_treino), k = K, list = T, returnTrain = F)
  acerto_media = NULL
  
  for (j in malha) {
    
    acerto = rep(0, K)
    
    for (i in 1:K) {
      teste = as.numeric(index[[i]])
      conjunto_treino = X_treino[-teste, ]
      alvo_treino = D_treino[, -teste]
      
      conjunto_validação = X_treino[teste, ]
      alvo_validação = D_treino[, teste]
      
      #Ajuste no treino
      resultado = Treino_MLP_classificação(100, conjunto_treino, alvo_treino, j, nrow(D_treino))
      W_opt = resultado$W
      M_opt = resultado$M
      
      #Obtenção dos resultados na validação
      estimativa = matrix(0, nrow = nrow(alvo_treino), ncol = nrow(conjunto_validação))
      
      
      for (j in 1:nrow(conjunto_validação)) {
        estimativa[, j] = Saída_MLP_classifica(W_opt, M_opt, as.numeric(conjunto_validação[j, ]))
      }
      
      vetor_estimativa = rep(0, ncol(estimativa))
      for (j in 1:ncol(estimativa)) {
        vetor_estimativa[j] = which.max(estimativa[, j])
      }
      
      vetor_validação = rep(0, ncol(estimativa))
      for (j in 1:ncol(estimativa)) {
        vetor_validação[j] = which.max(alvo_validação[, j])
      }
      acerto[i] = mean(vetor_validação == vetor_estimativa)
    }
    
    acerto_media = cbind(acerto_media, mean(acerto))
  }
  
  valida = list(acerto_media, malha[which.max(acerto_media)])
  names(valida) = c("medias", "opt")
  return(valida)
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


#Pré-processamento dos conjuntos de dados
## Iris
###Construindo a matriz de Rótulos para o conjunto Iris
names(iris) = c(c("sepal.length", "sepal.width", "petal.length", "petal.width", "specie"))
Y_iris = matrix(0, nrow = 3, ncol = nrow(iris))
J1 = which(iris$specie == "Iris-setosa")
J2 = which(iris$specie == "Iris-versicolor")
J3 = which(iris$specie == "Iris-virginica") 

Y_iris[, J1] = c(1, 0, 0)
Y_iris[, J2] = c(0, 1, 0)
Y_iris[, J3] = c(0, 0, 1)


Acerto_Iris = rep(0, 20)
Confusao_Iris =list()
malha_neuronios = c(1, 2)

#Realização de Hold-Out
for (k in 1:20){
  
  #Divisão treino-teste
  treino = sample(1:nrow(iris), 0.7 * nrow(iris))
  iris_X_treino_normalizado = as.data.frame(normaliza(iris[treino, -ncol(iris)]))
  iris_X_treino_normalizado = cbind(iris_X_treino_normalizado, rep(-1, nrow(iris_X_treino_normalizado)))
  
  iris_X_teste_normalizado = as.data.frame(normaliza(iris[-treino, -ncol(iris)]))
  iris_X_teste_normalizado = cbind(iris_X_teste_normalizado, rep(-1, nrow(iris_X_teste_normalizado)))
  alvo_treino = Y_iris[, treino]
  alvo_teste = Y_iris[, -treino]
  
  #Realização do grid search com validação cruzada 5 folds
  q_opt = valida(5, malha_neuronios, iris_X_treino_normalizado, alvo_treino)$opt
  
  #Ajuste da rede no treino
  resultado = Treino_MLP_classificação(100, iris_X_treino_normalizado, alvo_treino, q_opt, 3)
  W_opt = resultado$W
  M_opt = resultado$M
  
  #Determinação da saída no conjunto de teste
  estimativa = matrix(0, nrow = nrow(alvo_treino), ncol = nrow(iris_X_teste_normalizado))
  
  
  for (j in 1:nrow(iris_X_teste_normalizado)) {
    estimativa[, j] = Saída_MLP_classifica(W_opt, M_opt, as.numeric(iris_X_teste_normalizado[j, ]))
  }
  
  vetor_estimativa = rep(0, ncol(estimativa))
  for (j in 1:ncol(estimativa)) {
    vetor_estimativa[j] = which.max(estimativa[, j])
  }
  
  vetor_teste = rep(0, ncol(estimativa))
  for (j in 1:ncol(estimativa)) {
    vetor_teste[j] = which.max(alvo_teste[, j])
  }
  
  Acerto_Iris[k] = mean(vetor_teste == vetor_estimativa)
  Confusao_Iris[[k]] = table(vetor_estimativa, vetor_teste)

}

Acerto_Iris

#Construindo rótulos para o conjunto coluna
names(coluna)
Y_coluna = matrix(0, nrow = 3, ncol = nrow(coluna))
J1 = which(coluna$Classe == "Hernia de disco")
J2 = which(coluna$Classe == "Espondilolistese")
J3 = which(coluna$Classe == "Normal") 

Y_coluna[, J1] = c(1, 0, 0)
Y_coluna[, J2] = c(0, 1, 0)
Y_coluna[, J3] = c(0, 0, 1)

acerto_Coluna = rep(0, 20)
confusao_Coluna = list()

#Realização de Hold-Out
for (k in 1:20){
  
  #Divisão treino-teste
  treino = sample(1:nrow(coluna), 0.7 * nrow(coluna))
  coluna_X_treino_normalizado = as.data.frame(normaliza(coluna[treino, -ncol(coluna)]))
  coluna_X_treino_normalizado = cbind(coluna_X_treino_normalizado, rep(-1, nrow(coluna_X_treino_normalizado)))
  
  coluna_X_teste_normalizado = as.data.frame(normaliza(coluna[-treino, -ncol(coluna)]))
  coluna_X_teste_normalizado = cbind(coluna_X_teste_normalizado, rep(-1, nrow(coluna_X_teste_normalizado)))
  alvo_treino = Y_coluna[, treino]
  alvo_teste = Y_coluna[, -treino]
  
  #Realização do grid search com validação cruzada 5 folds
  q_opt = valida(5, malha_neuronios, coluna_X_treino_normalizado, alvo_treino)$opt
  
  #Ajuste da rede no treino
  resultado = Treino_MLP_classificação(100, coluna_X_treino_normalizado, alvo_treino, q_opt, 3)
  W_opt = resultado$W
  M_opt = resultado$M
  
  #Determinação da saída no conjunto de teste
  estimativa = matrix(0, nrow = nrow(alvo_treino), ncol = nrow(coluna_X_teste_normalizado))
  
  
  for (j in 1:nrow(coluna_X_teste_normalizado)) {
    estimativa[, j] = Saída_MLP_classifica(W_opt, M_opt, as.numeric(coluna_X_teste_normalizado[j, ]))
  }
  
  vetor_estimativa = rep(0, ncol(estimativa))
  for (j in 1:ncol(estimativa)) {
    vetor_estimativa[j] = which.max(estimativa[, j])
  }
  
  vetor_teste = rep(0, ncol(estimativa))
  for (j in 1:ncol(estimativa)) {
    vetor_teste[j] = which.max(alvo_teste[, j])
  }
  
  acerto_Coluna[k] = mean(vetor_teste == vetor_estimativa)
  confusao_Coluna[[k]] = table(vetor_estimativa, vetor_teste)
  
}

###Construindo a matriz de Rótulos para o conjunto Breast
breast = breast.cancer.wisconsin
fix(breast)
Y_breast = matrix(0, nrow = 2, ncol = nrow(breast))
J1 = which(breast$X2.1 == 2)
J2 = which(breast$X2.1 == 4)

Y_breast[, J1] = c(1, 0)
Y_breast[, J2] = c(0, 1)


breast = apply(breast, 2, as.numeric)
sum(is.na(breast))
breast = na.omit(breast)
breast = as.data.frame(breast)

acerto_breast = rep(0, 20)
confusao_breast= list()

#Realização do Hold-Out
for (k in 1:20) {
  
  #Divisão treino-teste
  treino = sample(1:nrow(breast), 0.7 * nrow(breast))
  breast_X_treino_normalizado = as.data.frame(normaliza(breast[treino, -ncol(breast)]))
  breast_X_treino_normalizado = cbind(breast_X_treino_normalizado, rep(-1, nrow(breast_X_treino_normalizado)))
  
  breast_X_teste_normalizado = as.data.frame(normaliza(breast[-treino, -ncol(breast)]))
  breast_X_teste_normalizado = cbind(breast_X_teste_normalizado, rep(-1, nrow(breast_X_teste_normalizado)))
  alvo_treino = Y_breast[, treino]
  alvo_teste = Y_breast[, -treino]
  
  #Realização d0 grid search com validação cruzada 5 folds
  q_opt = valida(5, malha_neuronios, breast_X_treino_normalizado, alvo_treino)$opt
  
  #Ajuste da rede no treino
  resultado = Treino_MLP_classificação(100, breast_X_treino_normalizado, alvo_treino, q_opt, 2)
  W_opt = resultado$W
  M_opt = resultado$M
  
  #Determinação da saída no conjunto de teste
  estimativa = matrix(0, nrow = nrow(alvo_teste), ncol = nrow(breast_X_teste_normalizado))
  
  
  for (j in 1:nrow(breast_X_teste_normalizado)) {
    estimativa[, j] = Saída_MLP_classifica(W_opt, M_opt, as.numeric(breast_X_teste_normalizado[j, ]))
  }
  
  vetor_estimativa = rep(0, ncol(estimativa))
  for (j in 1:ncol(estimativa)) {
    vetor_estimativa[j] = which.max(estimativa[, j])
  }
  
  vetor_teste = rep(0, ncol(estimativa))
  for (j in 1:ncol(estimativa)) {
    vetor_teste[j] = which.max(alvo_teste[, j])
  }
  
  acerto_breast[k] = mean(vetor_teste == vetor_estimativa)
  confusao_breast[[k]] = table(vetor_estimativa, vetor_teste)
  
}

STATS(acerto_breast)
STATS(acerto_Coluna)
STATS(Acerto_Iris)

#Matrizes de confusão
confusao_breast[[which.max(acerto_breast)]]
confusao_breast_max = matrix(c(139, 0, 66, 0 ), nrow = 2, ncol = 2 )
confusao_breast_max
ctable = as.table(matrix(c(139, 0, 66, 0), nrow = 2, byrow = TRUE))
colnames(ctable) = c("Real:Benigno", "Real:Maligno")
rownames(ctable) = c("Predição:Benigno", "Predição:Maligno")
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Matriz de confusão MLP para Breast: Melhor caso")

plt_best = matrix(Confusao_Iris[[which.max(Acerto_Iris)]], nrow = 3, ncol = 3)
plt_best = as.data.frame(plt_best)
rownames(plt_best) = c("Estimativa Setosa", "Estimativa Versicolour", "Estimativa Virginica")
colnames(plt_best) = c("Alvo Setosa", "Alvo Versicolour", "Alvo Virginica")
fix(plt_best)

mapa_calor = plot_ly(x = rownames(plt_best), y = colnames(plt_best), z = as.matrix(plt_best), type = "heatmap", colors = colorRamp(c("gray", "blue")))
mapa_calor = mapa_calor %>% layout(title = "Melhor Caso para IRIS")
mapa_calor

plt_best_coluna = matrix(confusao_Coluna[[which.max(acerto_Coluna)]], nrow = 3, ncol = 3)
plt_best_coluna = as.data.frame(plt_best_coluna)
rownames(plt_best_coluna) = c("Estimativa Hérnia", "Estimativa Espondiolistese", "Estimativa Normal")
colnames(plt_best_coluna) = c("Alvo Hérnia", "Alvo Espondiolistese", "Alvo Normal")
fix(plt_best_coluna)

mapa_calor_coluna = plot_ly(x = rownames(plt_best_coluna), y = colnames(plt_best_coluna), z = as.matrix(plt_best_coluna), type = "heatmap", colors = colorRamp(c("gray", "blue")))
mapa_calor_coluna = mapa_calor_coluna %>% layout(title = "Melhor Caso para COLUNA")
mapa_calor_coluna

subplot(mapa_calor, mapa_calor_coluna, margin = 0.15) %>% layout(title = "Melhor caso IRIS e COLUNA") 
