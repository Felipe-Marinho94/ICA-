#Algoritmos de busca aleatória local e global - RSL e RSG
#Algoritmos Evolucionários - PSO e Genéticos
#Inteligência Computacional - Trabalho 3
#Autor - Felipe Pinto Marinho
#Data:03/08/2022

#Carregando algumas bibliotecas relevantes
library(ggplot2)
library(plotly)
library(pracma)
library(MASS)
library(ggforce)
library(ggExtra)
library(ggthemes)
library(gridExtra)
library(latex2exp)

#Definindo algumas funções
#Construção do polinômio
poli = function(betas, x_entrada, d){
  
  soma = 0
  for(i in 0:d){
    soma = soma + betas[i+1] * (x_entrada)^(i)
  }
  
  return(soma)
}

#Função objetivo (função erro)
custo = function(Y_real, X_entrada, betas, d){
  
  valor = dot(as.matrix(Y_real - poli(betas, X_entrada, d)), as.matrix(Y_real - poli(betas, X_entrada, d)))
  
  return(valor)
}

Y_real = rep(0, 2)
X_entrada = seq(1, 2)
custo(Y_real, X_entrada, betas, 3)

#Busca Aleatória Global (GRS)
GRS = function(Ng, beta_l, beta_b, Y_real, X_entrada, d){
  
  #Gera solução dentro do intervalo estabelecido (Inicialização da solução)
  betas_best = runif(d + 1, min = beta_l, max = beta_b)
  
  #Avalia a solução inicial
  F_best = custo(Y_real, X_entrada, betas_best, d)
  
  #Inicialização da aptidão
  aptidao = rep(0, Ng)
  
  #Rodando o GRS por Ng iterações
  for (i in 1:Ng) {
    
    #Gera solução candidata
    betas_cand = runif(d + 1, min = beta_l, max = beta_b)
    
    #Avalia solução candidata
    F_cand = custo(Y_real, X_entrada, betas_cand, d)
    
    #Condição
    if (F_cand < F_best){
      
      betas_best = betas_cand
      F_best = F_cand
    }
    
    #armazenamento da aptidão
    aptidao[i] = F_best
  }
  
  #Resultados
  resultados = list(betas_best, F_best, aptidao)
  names(resultados) = c("betas_best", "F_best", "Aptidão")
  return(resultados)
}

#Busca Aleatória Local (LRS)
LRS = function(Ng, beta_l, beta_b, Y_real, X_entrada, d){
  
  #Gera solução dentro do intervalo estabelecido (Inicialização da solução)
  betas_best = runif(d + 1, min = beta_l, max = beta_b)
  
  #Avalia a solução inicial
  F_best = custo(Y_real, X_entrada, betas_best, d)
  
  #Inicialização da aptidão
  aptidao = rep(0, Ng)
  
  #Rodando o GRS por Ng iterações
  for (i in 1:Ng) {
    
    #Gera ruído gausiano branco
    ruido = rnorm(d + 1, mean = 0, sd = 1)
    
    #Gera solução candidata
    betas_cand = betas_best + ruido
    
    #Avalia solução candidata
    F_cand = custo(Y_real, X_entrada, betas_cand, d)
    
    #Condição
    if (F_cand < F_best){
      
      betas_best = betas_cand
      F_best = F_cand
    }
    
    #armazenamento da aptidão
    aptidao[i] = F_best
  }
  
  #Resultados
  resultados = list(betas_best, F_best, aptidao)
  names(resultados) = c("betas_best", "F_best", "Aptidão")
  return(resultados)
}

#Avaliação
#Pré-processamento
#Pré-processamento
fix(gauss)
names(gauss) = c("Saída", "Entrada")
sum(is.na(gauss))

#Representação gráfica
#Gráfico de dispersão
attach(gauss)
p = ggplot(gauss, aes(x=Entrada, y=Saída, color = Saída)) +
  geom_point() +
  theme(legend.position="none") +xlab("Entrada") + ylab("Saída") + theme(axis.title = element_text(size=30,face="bold"), plot.title = element_text(size=20, face="bold"), axis.text = element_text(size = 30))+ theme_classic()

p
p1 =  ggMarginal(p, type="histogram", fill = "slateblue")
p1

X_entrada = gauss$Entrada
Y_real = gauss$Saída

#Estimação dos parâmetros via GRS e LRS
resultados_GRS = GRS(100, -50, 50, Y_real, X_entrada, 8)
resultados_LRS = LRS(100, -50, 50, Y_real, X_entrada, 8)

Valores = data.frame(resultados_GRS$Aptidão, resultados_LRS$Aptidão)
fix(Valores)
figura = plot_ly(data = Valores, x = 1:100, y = Valores[, 1], type = "scatter", marker = list(size = 10, color = "rgba(125, 180, 240, .9)", line = list(color = "rgba(0, 152, 0, .8)", width = 2)), name = NULL) 
figura = figura %>% layout(xaxis = list(title = "Iterações", titlefont = list(size = 22), tickfont = list(size = 22)), yaxis = list(title = "Aptidão", titlefont = list(size = 22), tickfont = list(size = 22))) %>% add_lines(x = 1:100, y = Valores[, 1], name = 'GRS', line = list(width = 4))
figura = figura %>% add_trace(x = 1:100, y = Valores[, 2], name = "LRS", line = list(color = "black", width = 4), marker = list(size = 10, color = "rgba(12, 18, 240, .9)", line = list(color = "rgba(0, 0, 152, .8)", width = 2), symbol = 'triangle-up'))

figura
resultados_GRS$betas_best
resultados_LRS$betas_best

#Realizando a avaliação para 50 realizações
F_best_realizacao_GRS = rep(0, 50)
F_best_realizacao_LRS = rep(0, 50)

betas_best_realizacao_GRS = matrix(0, nrow = 10, ncol = 50)
betas_best_realizacao_LRS = matrix(0, nrow = 10, ncol = 50)

aptidao_realizacao_GRS = matrix(0, nrow = 1000, ncol = 50)
aptidao_realizacao_LRS = matrix(0, nrow = 1000, ncol = 50)

for (k in 1:50) {
  
  #Aplicação do GRS
  resultados_GRS = GRS(1000, -50, 50, Y_real, X_entrada, 9)
  
  #Aplicação do LRS
  resultados_LRS = LRS(1000, -50, 50, Y_real, X_entrada, 9)
  
  #Armazenando o F_best
  F_best_realizacao_GRS[k] = resultados_GRS$F_best
  F_best_realizacao_LRS[k] = resultados_LRS$F_best
  
  #Armazenando os betas_best
  betas_best_realizacao_GRS[, k] = resultados_GRS$betas_best
  betas_best_realizacao_LRS[, k] = resultados_LRS$betas_best
  
  #Armazenando a aptidão
  aptidao_realizacao_GRS[, k] = resultados_GRS$Aptidão
  aptidao_realizacao_LRS[, k] = resultados_LRS$Aptidão
  
}


betas_best_realizacao_GRS[, which.min(F_best_realizacao_GRS)]
betas_best_realizacao_LRS[, which.min(F_best_realizacao_LRS)]

F_best_realizacao_GRS[which.min(F_best_realizacao_GRS)]
F_best_realizacao_LRS[which.min(F_best_realizacao_LRS)]

#Convergência para as três melhores realizações
Dados_GRS = data.frame(aptidao_realizacao_GRS[, order(F_best_realizacao_GRS, decreasing = T)[1:3]])
Dados_LRS = data.frame(aptidao_realizacao_LRS[, order(F_best_realizacao_LRS, decreasing = T)[1:3]])

figura_GRS = plot_ly(data = Dados_GRS, x = 1:1000, y = Dados_GRS[, 1], type = "scatter", marker = list(size = 10, color = "rgba(125, 180, 240, .9)", line = list(color = "rgba(0, 152, 0, .8)", width = 2)), name = NULL) 
figura_GRS = figura_GRS %>% layout(xaxis = list(title = "Iterações", titlefont = list(size = 22), tickfont = list(size = 22)), yaxis = list(title = "Aptidão", titlefont = list(size = 22), tickfont = list(size = 22))) %>% add_lines(x = 1:1000, y = Dados_GRS[, 1], name = '3° Realização', line = list(width = 4))
figura_GRS = figura_GRS %>% add_trace(x = 1:1000, y = Dados_GRS[, 2], name = "2° Realização", line = list(color = "black", width = 4), marker = list(size = 10, color = "rgba(12, 18, 240, .9)", line = list(color = "rgba(0, 0, 152, .8)", width = 2), symbol = 'triangle-up'))
figura_GRS = figura_GRS %>% add_trace(x = 1:1000, y = Dados_GRS[, 3], name = "1° Realização", line = list(color = "red", width = 4), marker = list(size = 10, color = "rgba(240, 18, 12, .9)", line = list(color = "rgba(152, 0, 0, .8)", width = 2), symbol = 'cross'))
figura_GRS %>% layout(title = "Convergência do GRS para as 3 melhores realizações")

figura_LRS = plot_ly(data = Dados_LRS, x = 1:1000, y = Dados_LRS[, 1], type = "scatter", marker = list(size = 10, color = "rgba(125, 180, 240, .9)", line = list(color = "rgba(0, 152, 0, .8)", width = 2)), name = NULL) 
figura_LRS = figura_LRS %>% layout(xaxis = list(title = "Iterações", titlefont = list(size = 22), tickfont = list(size = 22)), yaxis = list(title = "Aptidão", titlefont = list(size = 22), tickfont = list(size = 22))) %>% add_lines(x = 1:1000, y = Dados_LRS[, 1], name = '3° Realização', line = list(width = 4))
figura_LRS = figura_LRS %>% add_trace(x = 1:1000, y = Dados_LRS[, 2], name = "2° Realização", line = list(color = "black", width = 4), marker = list(size = 10, color = "rgba(12, 18, 240, .9)", line = list(color = "rgba(0, 0, 152, .8)", width = 2), symbol = 'triangle-up'))
figura_LRS = figura_LRS %>% add_trace(x = 1:1000, y = Dados_LRS[, 3], name = "1° Realização", line = list(color = "red", width = 4), marker = list(size = 10, color = "rgba(240, 18, 12, .9)", line = list(color = "rgba(152, 0, 0, .8)", width = 2), symbol = 'cross'))
figura_LRS %>% layout(title = "Convergência do LRS para as 3 melhores realizações")

################################################################
#Algoritmos Evolucionários
#Particle Swarm Optimization (PSO)
PSO = function(Ng, Np, c1, c2, w, betasl, betasu, Y_real, X_entrada, d){
  
  #Inicialização das velocidades
  V = matrix(0, nrow = Np, ncol = d +1)
  
  #Inicialização da posição das partículas
  X = matrix(runif(Np * (d+1), max = betasu, min = betasl), nrow = Np, ncol = d + 1)
  
  #Avaliação das posições iniciais
  F_cand = custo(Y_real, X_entrada, X, d)
  
  #Inicialização das melhores posições e aptidões
  X_best = X
  F_best = F_cand
  
  #Determinação da melhor solução no enxame
  xg_best = X_best[which.min(F_best), ]
  
  #Inicialização da aptidão
  aptidao = rep(0, Ng)
  
  #Atualização ao longo das Ng iterações
  for (i in 1:Ng) {
    
    #Atualização das velocidades das particulas
    V_cog = matrix(runif(Np * (d+1), max = betasu, min = betasl), nrow = Np, ncol = d + 1) * (X_best - X) #Componente Cognitiva
    V_soc = matrix(runif(Np * (d+1), max = betasu, min = betasl), nrow = Np, ncol = d + 1) * (matrix(rep(xg_best, Np), nrow = Np, ncol = d + 1, byrow = T) - X) #Componente Social
    
    #Atualização da velocidade das particulas
    V = w * V + c1 * V_cog + c2 * V_soc
    
    #Atualização da posição das particulas
    X = X + V #Geração de novas soluções candidatas
    
    #Verificação se há soluções candidatas fora dos limites
    M_l = X < betasl
    M_u = X > betasu
    
    X[which(M_l == T)] = betasl
    X[which(M_u == T)] = betasu
    
    #Avaliação das soluções candidatas
    F_cand = custo(Y_real, X_entrada, X, d)
    DF = F_cand - F_best
    
    #Atualização das particulas que melhoraram a performance
    X_best[which(DF <= 0), ] = X[which(DF <= 0), ]
    F_best[which(DF <= 0)] = F_cand[which(DF <= 0)]
    
    #Conservação da posição das particulas que pioraram a performance
    X_best[which(DF > 0), ] = X_best[which(DF > 0), ]
    F_best[which(DF > 0)] = F_best[which(DF > 0)]
    
    #Determinação da melhor solução corrente no enxame
    xg_best = X_best[which.min(F_best), ]
    
    #Armazena a aptidão
    aptidao[i] = F_best[which.min(F_best)]
    
  }
  
  #Resultados
  resultados = list(xg_best, F_best[which.min(F_best)], aptidao)
  names(resultados) = c("betas", "F_best", "Aptidão")
  return(resultados)
}

#Inicialização dos parâmetros no PSO
Ng = 1000  #Número de iterações
Np = 200 #Número de particulas
c1 = 2.05 #constante e aceleração 1
c2 = c1 #constante de aceleração 2
w = 0.6 #constante de inércia
d = 9 #grau ótimo do polinômio

resultados = PSO(Ng, Np, c1, c2, w, -50, 50, Y_real, X_entrada, 9)
plot(x= 1:100, y = resultados$Aptidão, xlab = "Iterações", ylab = "Aptidão", lwd =3, pch = 19, type = "l", cex.lab = 1.5, cex.main = 1.5, main = "Aptidão vs Iterações para PSO" )
resultados$Aptidão
resultados$betas

#Aplicando o PSO ao longo de 50 realizações
F_best_realizacao_PSO = rep(0, 50)

betas_best_realizacao_PSO = matrix(0, nrow = 10, ncol = 50)

aptidao_realizacao_PSO = matrix(0, nrow = 1000, ncol = 50)

for (k in 1:50) {
  
  #Aplicação do PSO
  resultados_PSO = PSO(Ng, Np, c1, c2, w, -50, 50, Y_real, X_entrada, 9)
  
  #Armazenando o F_best
  F_best_realizacao_PSO[k] = resultados_PSO$F_best
  
  #Armazenando os betas_best
  betas_best_realizacao_PSO[, k] = resultados_PSO$betas
  
  #Armazenando a aptidão
  aptidao_realizacao_PSO[, k] = resultados_PSO$Aptidão
  
}

betas_best_realizacao_PSO[, which.min(F_best_realizacao_PSO)]
F_best_realizacao_PSO[which.min(F_best_realizacao_PSO)]
F_best_realizacao_PSO
order(F_best_realizacao_PSO, decreasing = T)[1:3]

#Convergência para as três melhores realizações
Dados_PSO = data.frame(aptidao_realizacao_PSO[, order(F_best_realizacao_PSO, decreasing = T)[1:3]])

figura_PSO = plot_ly(data = Dados_PSO, x = 1:1000, y = Dados_PSO[, 1], type = "scatter", marker = list(size = 10, color = "rgba(125, 180, 240, .9)", line = list(color = "rgba(0, 152, 0, .8)", width = 2)), name = NULL) 
figura_PSO = figura_PSO %>% layout(xaxis = list(title = "Iterações", titlefont = list(size = 22), tickfont = list(size = 22)), yaxis = list(title = "Aptidão", titlefont = list(size = 22), tickfont = list(size = 22))) %>% add_lines(x = 1:1000, y = Dados_PSO[, 1], name = '3° Realização', line = list(width = 4))
figura_PSO = figura_PSO %>% add_trace(x = 1:1000, y = Dados_PSO[, 2], name = "2° Realização", line = list(color = "black", width = 4), marker = list(size = 10, color = "rgba(12, 18, 240, .9)", line = list(color = "rgba(0, 0, 152, .8)", width = 2), symbol = 'triangle-up'))
figura_PSO = figura_PSO %>% add_trace(x = 1:1000, y = Dados_PSO[, 3], name = "1° Realização", line = list(color = "red", width = 4), marker = list(size = 10, color = "rgba(240, 18, 12, .9)", line = list(color = "rgba(152, 0, 0, .8)", width = 2), symbol = 'cross'))
figura_PSO %>% layout(title = "Convergência do PSO para as 3 melhores realizações")

#Representação gráfica
#Criação de grids para os beta_zero e beta_um
beta_zero = seq(-50, 50)
beta_um = seq(-50, 50)

J = matrix(0, nrow = length(beta_zero), ncol = length(beta_um))
for (i in 1:length(beta_zero)) {
  for (j in 1:length(beta_um)) {
    J[i, j] = custo(Y_real, X_entrada, c(beta_zero[i], beta_um[j]), 1)
  }
}

res = persp(beta_zero, beta_um, J, xlab = "B0", ylab = "B1", main = "Superfície de erro", phi = 45, theta = 60, col = "yellow")

res
betas_GRS = GRS(1000, -50, 50, Y_real, X_entrada, 1)$betas
betas_LRS = LRS(1000, -50, 50, Y_real, X_entrada, 1)$betas
betas_PSO = PSO(Ng, Np, c1, c2, w, -50, 50, Y_real, X_entrada, 1)$betas
points(trans3d(c(betas_GRS[1], betas_LRS[1], betas_PSO[1]), c(betas_GRS[2], betas_LRS[2], betas_PSO[2]), c(custo(Y_real, X_entrada, betas_GRS, 1), custo(Y_real, X_entrada, betas_LRS, 1), custo(Y_real, X_entrada, betas_PSO, 1)), pmat=res), col = c("red", "blue", "green"), cex = 1.5, pch = 19)
custo(Y_real, X_entrada, betas_PSO, 1)
legend("topright", legend = c("GRS", "LRS", "PSO"), col = c("red", "blue", "green"), pch = c(19, 19, 19), bty = "n", pt.cex = 2, cex = 1.2, text.col = "black", horiz = F , inset = c(0.1, 0.1, 0.1))

#Nova função custo
SEA = function(Y_real, X_entrada, betas, d){
  erro = abs(Y_real - poli(betas, X_entrada, d))
  return(sum(erro))
}

GRS_SEA = function(Ng, beta_l, beta_b, Y_real, X_entrada, d){
  
  #Gera solução dentro do intervalo estabelecido (Inicialização da solução)
  betas_best = runif(d + 1, min = beta_l, max = beta_b)
  
  #Avalia a solução inicial
  F_best = SEA(Y_real, X_entrada, betas_best, d)
  
  #Inicialização da aptidão
  aptidao = rep(0, Ng)
  
  #Rodando o GRS por Ng iterações
  for (i in 1:Ng) {
    
    #Gera solução candidata
    betas_cand = runif(d + 1, min = beta_l, max = beta_b)
    
    #Avalia solução candidata
    F_cand = SEA(Y_real, X_entrada, betas_cand, d)
    
    #Condição
    if (F_cand < F_best){
      
      betas_best = betas_cand
      F_best = F_cand
    }
    
    #armazenamento da aptidão
    aptidao[i] = F_best
  }
  
  #Resultados
  resultados = list(betas_best, F_best, aptidao)
  names(resultados) = c("betas_best", "F_best", "Aptidão")
  return(resultados)
}

#Busca Aleatória Local (LRS)
LRS_SEA = function(Ng, beta_l, beta_b, Y_real, X_entrada, d){
  
  #Gera solução dentro do intervalo estabelecido (Inicialização da solução)
  betas_best = runif(d + 1, min = beta_l, max = beta_b)
  
  #Avalia a solução inicial
  F_best = SEA(Y_real, X_entrada, betas_best, d)
  
  #Inicialização da aptidão
  aptidao = rep(0, Ng)
  
  #Rodando o GRS por Ng iterações
  for (i in 1:Ng) {
    
    #Gera ruído gausiano branco
    ruido = rnorm(d + 1, mean = 0, sd = 1)
    
    #Gera solução candidata
    betas_cand = betas_best + ruido
    
    #Avalia solução candidata
    F_cand = SEA(Y_real, X_entrada, betas_cand, d)
    
    #Condição
    if (F_cand < F_best){
      
      betas_best = betas_cand
      F_best = F_cand
    }
    
    #armazenamento da aptidão
    aptidao[i] = F_best
  }
  
  #Resultados
  resultados = list(betas_best, F_best, aptidao)
  names(resultados) = c("betas_best", "F_best", "Aptidão")
  return(resultados)
}

PSO_SEA = function(Ng, Np, c1, c2, w, betasl, betasu, Y_real, X_entrada, d){
  
  #Inicialização das velocidades
  V = matrix(0, nrow = Np, ncol = d +1)
  
  #Inicialização da posição das partículas
  X = matrix(runif(Np * (d+1), max = betasu, min = betasl), nrow = Np, ncol = d + 1)
  
  #Avaliação das posições iniciais
  F_cand = SEA(Y_real, X_entrada, X, d)
  
  #Inicialização das melhores posições e aptidões
  X_best = X
  F_best = F_cand
  
  #Determinação da melhor solução no enxame
  xg_best = X_best[which.min(F_best), ]
  
  #Inicialização da aptidão
  aptidao = rep(0, Ng)
  
  #Atualização ao longo das Ng iterações
  for (i in 1:Ng) {
    
    #Atualização das velocidades das particulas
    V_cog = matrix(runif(Np * (d+1), max = betasu, min = betasl), nrow = Np, ncol = d + 1) * (X_best - X) #Componente Cognitiva
    V_soc = matrix(runif(Np * (d+1), max = betasu, min = betasl), nrow = Np, ncol = d + 1) * (matrix(rep(xg_best, Np), nrow = Np, ncol = d + 1, byrow = T) - X) #Componente Social
    
    #Atualização da velocidade das particulas
    V = w * V + c1 * V_cog + c2 * V_soc
    
    #Atualização da posição das particulas
    X = X + V #Geração de novas soluções candidatas
    
    #Verificação se há soluções candidatas fora dos limites
    M_l = X < betasl
    M_u = X > betasu
    
    X[which(M_l == T)] = betasl
    X[which(M_u == T)] = betasu
    
    #Avaliação das soluções candidatas
    F_cand = SEA(Y_real, X_entrada, X, d)
    DF = F_cand - F_best
    
    #Atualização das particulas que melhoraram a performance
    X_best[which(DF <= 0), ] = X[which(DF <= 0), ]
    F_best[which(DF <= 0)] = F_cand[which(DF <= 0)]
    
    #Conservação da posição das particulas que pioraram a performance
    X_best[which(DF > 0), ] = X_best[which(DF > 0), ]
    F_best[which(DF > 0)] = F_best[which(DF > 0)]
    
    #Determinação da melhor solução corrente no enxame
    xg_best = X_best[which.min(F_best), ]
    
    #Armazena a aptidão
    aptidao[i] = F_best[which.min(F_best)]
    
  }
  
  #Resultados
  resultados = list(xg_best, F_best[which.min(F_best)], aptidao)
  names(resultados) = c("betas", "F_best", "Aptidão")
  return(resultados)
}

#Realizando a avaliação para 50 realizações
F_best_realizacao_GRS_SEA = rep(0, 50)
F_best_realizacao_LRS_SEA = rep(0, 50)

betas_best_realizacao_GRS_SEA = matrix(0, nrow = 10, ncol = 50)
betas_best_realizacao_LRS_SEA = matrix(0, nrow = 10, ncol = 50)

aptidao_realizacao_GRS_SEA = matrix(0, nrow = 1000, ncol = 50)
aptidao_realizacao_LRS_SEA = matrix(0, nrow = 1000, ncol = 50)

for (k in 1:50) {
  
  #Aplicação do GRS
  resultados_GRS_SEA = GRS_SEA(1000, -50, 50, Y_real, X_entrada, 9)
  
  #Aplicação do LRS
  resultados_LRS_SEA = LRS_SEA(1000, -50, 50, Y_real, X_entrada, 9)
  
  #Armazenando o F_best
  F_best_realizacao_GRS_SEA[k] = resultados_GRS_SEA$F_best
  F_best_realizacao_LRS_SEA[k] = resultados_LRS_SEA$F_best
  
  #Armazenando os betas_best
  betas_best_realizacao_GRS_SEA[, k] = resultados_GRS_SEA$betas_best
  betas_best_realizacao_LRS_SEA[, k] = resultados_LRS_SEA$betas_best
  
  #Armazenando a aptidão
  aptidao_realizacao_GRS_SEA[, k] = resultados_GRS_SEA$Aptidão
  aptidao_realizacao_LRS_SEA[, k] = resultados_LRS_SEA$Aptidão
  
}


betas_best_realizacao_GRS_SEA[, which.min(F_best_realizacao_GRS_SEA)]
betas_best_realizacao_LRS_SEA[, which.min(F_best_realizacao_LRS_SEA)]

F_best_realizacao_GRS_SEA[which.min(F_best_realizacao_GRS_SEA)]
F_best_realizacao_LRS_SEA[which.min(F_best_realizacao_LRS_SEA)]

#Convergência para as três melhores realizações
Dados_GRS = data.frame(aptidao_realizacao_GRS_SEA[, order(F_best_realizacao_GRS_SEA, decreasing = T)[1:3]])
Dados_LRS = data.frame(aptidao_realizacao_LRS_SEA[, order(F_best_realizacao_LRS_SEA, decreasing = T)[1:3]])

figura_GRS = plot_ly(data = Dados_GRS, x = 1:1000, y = Dados_GRS[, 1], type = "scatter", marker = list(size = 10, color = "rgba(125, 180, 240, .9)", line = list(color = "rgba(0, 152, 0, .8)", width = 2)), name = NULL) 
figura_GRS = figura_GRS %>% layout(xaxis = list(title = "Iterações", titlefont = list(size = 22), tickfont = list(size = 22)), yaxis = list(title = "Aptidão", titlefont = list(size = 22), tickfont = list(size = 22))) %>% add_lines(x = 1:1000, y = Dados_GRS[, 1], name = '3° Realização', line = list(width = 4))
figura_GRS = figura_GRS %>% add_trace(x = 1:1000, y = Dados_GRS[, 2], name = "2° Realização", line = list(color = "black", width = 4), marker = list(size = 10, color = "rgba(12, 18, 240, .9)", line = list(color = "rgba(0, 0, 152, .8)", width = 2), symbol = 'triangle-up'))
figura_GRS = figura_GRS %>% add_trace(x = 1:1000, y = Dados_GRS[, 3], name = "1° Realização", line = list(color = "red", width = 4), marker = list(size = 10, color = "rgba(240, 18, 12, .9)", line = list(color = "rgba(152, 0, 0, .8)", width = 2), symbol = 'cross'))
figura_GRS %>% layout(title = "Convergência do GRS para as 3 melhores realizações")

figura_LRS = plot_ly(data = Dados_LRS, x = 1:1000, y = Dados_LRS[, 1], type = "scatter", marker = list(size = 10, color = "rgba(125, 180, 240, .9)", line = list(color = "rgba(0, 152, 0, .8)", width = 2)), name = NULL) 
figura_LRS = figura_LRS %>% layout(xaxis = list(title = "Iterações", titlefont = list(size = 22), tickfont = list(size = 22)), yaxis = list(title = "Aptidão", titlefont = list(size = 22), tickfont = list(size = 22))) %>% add_lines(x = 1:1000, y = Dados_LRS[, 1], name = '3° Realização', line = list(width = 4))
figura_LRS = figura_LRS %>% add_trace(x = 1:1000, y = Dados_LRS[, 2], name = "2° Realização", line = list(color = "black", width = 4), marker = list(size = 10, color = "rgba(12, 18, 240, .9)", line = list(color = "rgba(0, 0, 152, .8)", width = 2), symbol = 'triangle-up'))
figura_LRS = figura_LRS %>% add_trace(x = 1:1000, y = Dados_LRS[, 3], name = "1° Realização", line = list(color = "red", width = 4), marker = list(size = 10, color = "rgba(240, 18, 12, .9)", line = list(color = "rgba(152, 0, 0, .8)", width = 2), symbol = 'cross'))
figura_LRS %>% layout(title = "Convergência do LRS para as 3 melhores realizações")

#Realizando a avaliação para 50 realizações
F_best_realizacao_PSO_SEA = rep(0, 50)
betas_best_realizacao_PSO_SEA = matrix(0, nrow = 10, ncol = 50)
aptidao_realizacao_PSO_SEA = matrix(0, nrow = 1000, ncol = 50)

for (k in 1:50) {
  
  #Aplicação do PSO
  resultados_PSO_SEA = PSO_SEA(Ng, Np, c1, c2, w, -50, 50, Y_real, X_entrada, 9)
  
  #Armazenando o F_best
  F_best_realizacao_PSO_SEA[k] = resultados_PSO_SEA$F_best
  
  #Armazenando os betas_best
  betas_best_realizacao_PSO_SEA[, k] = resultados_PSO_SEA$betas
  
  #Armazenando a aptidão
  aptidao_realizacao_PSO_SEA[, k] = resultados_PSO_SEA$Aptidão
  
  
}


betas_best_realizacao_PSO_SEA[, which.min(F_best_realizacao_PSO_SEA)]

F_best_realizacao_PSO_SEA[which.min(F_best_realizacao_PSO_SEA)]

#Convergência para as três melhores realizações
Dados_PSO = data.frame(aptidao_realizacao_PSO_SEA[, order(F_best_realizacao_PSO, decreasing = T)[1:3]])

figura_PSO = plot_ly(data = Dados_PSO, x = 1:1000, y = Dados_PSO[, 1], type = "scatter", marker = list(size = 10, color = "rgba(125, 180, 240, .9)", line = list(color = "rgba(0, 152, 0, .8)", width = 2)), name = NULL) 
figura_PSO = figura_PSO %>% layout(xaxis = list(title = "Iterações", titlefont = list(size = 22), tickfont = list(size = 22)), yaxis = list(title = "Aptidão", titlefont = list(size = 22), tickfont = list(size = 22))) %>% add_lines(x = 1:1000, y = Dados_PSO[, 1], name = '3° Realização', line = list(width = 4))
figura_PSO = figura_PSO %>% add_trace(x = 1:1000, y = Dados_PSO[, 2], name = "2° Realização", line = list(color = "black", width = 4), marker = list(size = 10, color = "rgba(12, 18, 240, .9)", line = list(color = "rgba(0, 0, 152, .8)", width = 2), symbol = 'triangle-up'))
figura_PSO = figura_PSO %>% add_trace(x = 1:1000, y = Dados_PSO[, 3], name = "1° Realização", line = list(color = "red", width = 4), marker = list(size = 10, color = "rgba(240, 18, 12, .9)", line = list(color = "rgba(152, 0, 0, .8)", width = 2), symbol = 'cross'))
figura_PSO %>% layout(title = "Convergência do PSO para as 3 melhores realizações")
