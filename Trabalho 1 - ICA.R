rm(list = ls())
#Trabalho 1 - Inteligência Computacional
#Nome:Felipe Pinto Marinho
#Matrícula:502661
#Data:25/06/2022

#Carregando algumas bibliotecas relevantes
library(plotly)
library(ggplot2)
library(ggExtra)
library(ggthemes)
library(gridExtra)
library(caret)
library(glmnet)
library(MASS)
library(Renvlp)
library(ggforce)
library(pracma)


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

#Função para calcular diversas métricas de erro
métricas=function(Y_estimado,Y_real){
  MBE=mean(Y_real-Y_estimado)
  print(MBE)
  MAE=mean(abs(Y_real-Y_estimado))
  print(MAE)
  RMSE=sqrt(mean((Y_real-Y_estimado)^2))
  print(RMSE)
  rRMSE=((RMSE)/mean(Y_real))*100
  print(rRMSE)
  SSE=sum((Y_real-Y_estimado)^2)
  SSTO=sum((Y_real-mean(Y_real))^2)
  R2=1-(SSE/SSTO)
  print(R2)
  
  erro = list(RMSE, R2)
  names(erro) = c("RMSE", "R2")
  return(erro)
  
}

#Separação conjunto de treino e teste
set.seed(2)
treino = sample(1:nrow(gauss), 0.7*nrow(gauss))
gaussTreino = gauss[treino,]
gaussTeste = gauss[-treino,]
EntradaTreino = gauss$Entrada[treino]
EntradaTeste = gauss$Entrada[-treino]
SaídaTreino = gauss$Saída[treino]
SaídaTeste = gauss$Saída[-treino]

#Definição do treino
train.control = trainControl(method = "cv", number = 5)
names(train.control)
train.control$number

#Determinação das fórmula (dataset parede solar)
n = names(gauss)
f = as.formula(paste("Saída ~", paste(n[!n %in% "Saída"], collapse="+")))
f

#Ajuste de uma regressão linear múltipla padrão utilizando a biblioteca Caret
LM = train(f, gaussTreino, method = "lm", trControl=train.control, preProcess=c("center", "scale"))
gaussHat = predict(LM, gaussTeste)
métricas(gaussHat, SaídaTeste)

#Ajuste de uma regressão linear múltipla regularizada
#Ridge
#Regressão Ridge
Ridge = function(lambda, x, y){
  
  #Estimação da matriz dos coeficientes
  B = ginv(t(x) %*% x + lambda*diag(ncol(x))) %*% t(x) %*% y
  
  return(B)
}

#Estimação do parâmetro lambda por meio de hold-out
grid = 10^seq(-1, -3, length = 50)
RMSE = 0
R2 = 0
xT = cbind(rep(1, nrow(gaussTreino)), gaussTreino$Entrada)
xTest = cbind(rep(1, nrow(gaussTeste)), gaussTeste$Entrada)

for (l in grid) {
  
  #divisão treino e teste para hold-out
  tr = sample(1:nrow(gaussTreino), 0.7*nrow(gaussTreino))
  xtre = as.matrix(xT[tr, ])
  ytre = as.matrix(SaídaTreino[tr])
  
  #Obtenção do RMSE
  Bhat = Ridge(l, xtre, ytre)
  yhat = as.matrix(xT[-tr, ]) %*% Bhat
  RMSE = cbind(RMSE, métricas(yhat, SaídaTreino[-tr])$RMSE)
  R2 = cbind(R2, métricas(yhat, SaídaTreino[-tr])$R2)
}

RMSE = RMSE[-1]
R2 = R2[-1]
par(mfrow=c(1,2))
plot(x = grid, y = RMSE, type = "b", xlab = "Lambda", ylab = "RMSE", pch = 20, lwd = 3, cex = 1.5 ,bty = 'L', cex.lab = 1.5, col = "slateblue") 
plot(x = grid, y = R2, type = "b", xlab = "Lambda", ylab = "R²", pch = 6, lwd = 3, cex = 1.5 ,bty = 'L', cex.lab = 1.5, col = "black")
grid[which.min(RMSE)]
grid[which.max(R2)]
lambdaOtim = grid[which.max(R2)]

#Estimação dos coeficientes
Bhat = Ridge(lambdaOtim, xT, SaídaTreino)

#Estimação das saídas pelo Ridge
yhat = xTest %*% Bhat

métricas(yhat, SaídaTeste)

#Modelo de envelopes
dim = u.xenv(as.matrix(gaussTreino$Entrada), as.matrix(gaussTreino$Saída))

par(mfrow = c(1,2))
plot(1:length(dim$aic.seq), dim$aic.seq, type = "b", col = "deepskyblue4", main = "u x AIC", xlab = "u", ylab = "AIC", pch = 20, lwd = 4, cex = 1 ,bty = 'L', cex.lab = 1.5)
plot(1:length(dim$bic.seq), dim$bic.seq, type = "b", col = "red", main = "u x BIC", xlab = "u", ylab = "BIC", pch = 20, lwd = 4, cex = 1 ,bty = 'L', cex.lab = 1.5)
dim$u.bic
dim$u.aic
u_otimo = dim$u.bic

#Ajuste do modelo de envelope para os preditores
EP = xenv(gaussTreino$Entrada, SaídaTreino, u_otimo)

##Previsão no teste
Y_hat_envelope = rep(0, nrow(gaussTeste))
for (k in 1:nrow(gaussTeste)) {
  Y_hat_envelope[k] = pred.xenv(EP, gaussTeste[k, -2])$value
}

métricas(Y_hat_envelope, SaídaTeste)

#Regresso polinomial
attach(gauss)
RMSE_poli = 0
R2_poli = 0
debuggingState(on=FALSE)

for (i in 1:10) {
  
  #divisão treino e teste para hold-out
  tr = sample(1:nrow(gaussTreino), 0.7*nrow(gaussTreino))
  
  #Desempenho
  pol = lm(Saída~poly(Entrada, i), data = gaussTreino[tr, ])
  yhat = predict(pol,list(Entrada = EntradaTreino[-tr]))
  RMSE_poli = cbind(RMSE_poli, métricas(yhat, SaídaTreino[-tr])$RMSE)
  R2_poli = cbind(R2_poli, métricas(yhat, SaídaTreino[-tr])$R2)
}

RMSE_poli = RMSE_poli[-1] 
R2_poli = R2_poli[-1]

par(mfrow=c(1,2))
plot(x = 1:10, y = RMSE_poli, type = "b", xlab = "Grau", ylab = "RMSE", pch = 20, lwd = 3, cex = 1.5 ,bty = 'L', cex.lab = 1.5, col = "slateblue") 
plot(x = 1:10, y = R2_poli, type = "b", xlab = "Grau", ylab = "R²", pch = 6, lwd = 3, cex = 1.5 ,bty = 'L', cex.lab = 1.5, col = "black")
which.min(RMSE_poli) 

pol = lm(Saída~poly(Entrada, 9), data = gaussTreino)
yhat = predict(pol,list(Entrada = EntradaTeste))
métricas(yhat, SaídaTeste)

#Regressao local
grid_loc = 10^seq(0, -1, length = 29)
RMSE_loc = 0
R2_loc = 0

for (j in grid_loc) {
  
  #divisão treino e teste para hold-out
  tr = sample(1:nrow(gaussTreino), 0.7*nrow(gaussTreino))
  
  #Desempenho
  loc = loess(Saída~Entrada, span = j, data = gaussTreino[tr,])
  yhat_loc = predict(loc, gaussTreino[-tr,])
  RMSE_loc = cbind(RMSE_loc, métricas(yhat_loc, SaídaTreino[-tr])$RMSE)
  R2_loc = cbind(R2_loc, métricas(yhat_loc, SaídaTreino[-tr])$R2)
  
}

RMSE_loc_n = RMSE_loc[which(is.na(RMSE_loc) == F)]
R2_loc_n = R2_loc[which(is.na(R2_loc) == F)]
RMSE_loc_n = RMSE_loc_n[-1]
R2_loc_n = R2_loc_n[-1]

r = grid_loc[which(is.na(RMSE_loc) == F)]
r = r[-1]

par(mfrow=c(1,2))
plot(x = r, y = RMSE_loc_n, type = "b", xlab = "s", ylab = "RMSE", pch = 20, lwd = 3, cex = 1.5 ,bty = 'L', cex.lab = 1.5, col = "slateblue") 
plot(x = r, y = R2_loc_n, type = "b", xlab = "s", ylab = "R²", pch = 6, lwd = 3, cex = 1.5 ,bty = 'L', cex.lab = 1.5, col = "black")
s = r[which.min(RMSE_loc_n)]

loc = loess(Saída~Entrada, span = s, data = gaussTreino)
yhat_loc = predict(loc, gaussTeste)
métricas(yhat_loc[-1], SaídaTeste[-1])

################################################################################
#Sistema de Inferência Fuzzy - Mamdani
ordenamento = sort.int(Entrada, index.return = T) 
names(ordenamento)
x = ordenamento$x
y = Saída[ordenamento$ix]

#número de funções de pertinência
Nmf = 4

#Valores iniciais dos parâmetros das funções de pertinência
centers = c(24, 82, 150, 197, 226)
centers = c(69, 140, 175, 226)
saída= Saída[centers]
saída

par(mfrow = c(1, 1))
plot(x = Entrada, y = Saída, main = "Curva Simulada", lwd = 2, xlab = "Entrada", ylab = "Saída", cex.lab = 1.3, cex.axis = 1.3, xlim = c(0,350), ylim = c(0,150))
points(centers, saída, pch = 20, col = "red", cex = 2)
legend("topright", legend = c("observações", "pontos de suporte", "muito baixa", "baixa", "média", "alta", "muito alta"), col = c("black", "red", "darkslategrey", "green", "blue", "brown", "orange"), pch = c(1, 20, 20, 20, 20, 20, 20), bty = "n", pt.cex = 2, cex = 1.2, text.col = "black", horiz = F , inset = c(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))

#dispersão para cada gaussiana
spread = 5

#criação de um grid para a velocidade
xx = Entrada[-treino] #seq(from = 0.8*min(x), to = 1.2*max(x), by = 0.1)
mi = matrix(0, nrow = Nmf, ncol = length(xx))
ypred = rep(0, 4)
ypred_TS = rep(0, 4)
yhat_mamdani = rep(0, length(xx))
yhat_TS = rep(0, length(xx))

#Divisao do universo de discurso em intervalos de larguras iguais
t = c(0, 50, 100, 150, 200, 250)
t = c(0, 50, 70, 125, 250)
fix(gauss)

#funções de pertinência gaussianas para a velocidade
for (j in 1:length(xx)) {
  for (i in 1:Nmf) {
    mi[i, j] = exp(-(xx[j] - centers[i])^2/(2*(spread)^2)) #ativaçao da i-ésima regra
    ypred[i] = saída[i] #saída predita para i-ésima regra (singleton)
    I = which(Entrada >= t[i] & Entrada < t[i+1])
    linear = lm(Saída~Entrada, data = gauss[I, ])
    ypred_TS[i] = dot(linear$coefficients, c(1,xx[j]))
  }
  
  yhat_mamdani[j] = dot(ypred, mi[, j])/sum(mi[, j])
  yhat_TS[j] = dot(ypred_TS, mi[, j])/sum(mi[, j])
}


#Amplificação
K=50

#plot's das funções de pertinência
lines(xx, K*mi[1,], type = "l", col = "darkslategrey", lwd = 3)
lines(xx, K*mi[2,], type = "l", col = "green", lwd = 3)
lines(xx, K*mi[3,], type = "l", col = "blue", lwd = 3)
lines(xx, K*mi[4,], type = "l", col = "brown", lwd = 3)
lines(xx, K*mi[5,], type = "l", col = "orange", lwd = 3)

#Etapa de predição
métricas(yhat_mamdani, SaídaTeste)
métricas(yhat_TS, SaídaTeste)

#RMSE e R2 para fuzzy
RMSE_Mamdami = c(39.29, 23.34, 21.41)
RMSE_TS = c(20.37, 52.77, 94.57)

R2_Mamdami = c(0.06, 0.67, 0.72)
R2_TS = c(0.74, -0.69)
##############################################################

#plot
resultados = cbind(yhat_loc, SaídaTeste)
resultados = as.data.frame(resultados)
resultados = na.omit(resultados)
fix(resultados)

p = ggplot(resultados, aes(x=resultados$SaídaTeste, y=resultados$yhat_loc, color = SaídaTeste)) +
  geom_point() +
  theme(legend.position="none") +xlab("Saída Medida") + ylab("Saída Estimada") + theme(
    axis.title=element_text(size=14,face="bold"))

p = p + geom_smooth(method=lm , color="black", se=FALSE)
p

# with marginal histogram
p1 = ggMarginal(p, type="histogram", fill = "slateblue")
p1 

#radar plot
modelos = c("Linear", "Ridge", "Envelope", "Polinomial", "Local", "Mamdani", "Takagi-Sugeno")
RMSE = c(32.62, 32.62, 32.62, 5.91, 2.53, 25.31, 19.16)
MBE = c(5.01, 5.01, 5.01, 0.62, -0.45, 3.42, -0.45)
MAE = c(17.86, 17.87, 17.86, 10.50, 10.47)

erros = cbind(modelos, RMSE, MBE)
erros = data.frame(erros)
rownames(erros) = modelos
fix(erros)
attach(erros)

Modelo = erros$modelos
plotMBE_RMSE = ggplot(erros, aes(x = RMSE, y = MBE), group=Modelo) + 
  geom_point(aes(shape=Modelo, color=Modelo, 
                 size=Modelo))+
  scale_shape_manual(values=c(17, 19, 18, 15, 10, 9, 8))+
  scale_color_manual(values=c('#006600','red', '#cc9900', "blue", "black", "magenta", "darkslategrey"))+
  theme(legend.position="right")+
  scale_size_manual(values=c(3,4,3,3,3,3,3))+
  # horizontal
  geom_hline(yintercept=0, color="black", size=.5) + 
  # vertical
  geom_vline(xintercept=0, color="black", size=.5)+
  coord_cartesian(xlim=c(-40,40),ylim = c(-40,40))+
  scale_x_continuous(breaks = seq(-40,40,((40+40)/5)))+
  scale_y_continuous(breaks = seq(-40,40,((40+40)/5)))+
  coord_fixed()+
  geom_circle(linetype="dashed", color = "black" ,aes(x0 = 0, y0 = 0, r = seq(0, 40, length.out = 7)),inherit.aes = FALSE)+
  theme(
    axis.title=element_text(size=14,face="bold"), plot.title = element_text(size=14,face="bold"), axis.text = element_text(size = 16))
plotMBE_RMSE

#Análise de resíduos
yhat_loc_total = predict(loc, gauss)
residuos = yhat_loc_total[-1]-gauss$Saída[-1]
pacf(residuos, ylab = "PACF")

hist(residuos, col = "lightblue", ylab = "Frequência", xlab = "Resíduos", main = "Histograma Resíduos", breaks = 20)
curve(dnorm(x, mean = mean(residuos), sd = sd(residuos)), add = T, lwd = 2)

dados = data.frame(residuos)
fix(dados)
ggplot(dados) + aes(x = dados$residuos) + geom_histogram(fill = "lightblue", col = "black", alpha = 0.5, bins = 20, aes(y = ..density..)) +
stat_function(fun = dnorm, args = list(mean = mean(residuos), sd = sd(residuos)), size = 1.5, col = "slateblue")+
  theme(legend.position="none") +xlab("Resíduos") + ylab("Densidade") + theme(
    axis.title=element_text(size=14,face="bold"), plot.title = element_text(size=14,face="bold"), axis.text = element_text(size = 16)) 


classe = rep(0, nrow(gauss))
classe[treino] = "treino"
classe[-treino] = "teste"
aero = cbind(gauss, classe)
aero = data.frame(aero)[-1,]
attach(aero)

scatter <- ggplot(aero, aes(x = yhat_loc_total[-1], y = residuos, color = classe)) + 
  
  geom_point() +  labs( x ="predições") 


scatter <- ggplotly(p = scatter, type = 'scatter')
scatter

violin <- aero %>%
  
  plot_ly(x = ~classe, y = ~residuos, split = ~classe, type = 'violin' )


s <- subplot(
  
  scatter,
  
  violin,
  
  nrows = 1, heights = c(1), widths = c(0.65, 0.35), margin = 0.01,
  
  shareX = TRUE, shareY = TRUE, titleX = TRUE, titleY = TRUE
  
)


layout(s, showlegend = T)

