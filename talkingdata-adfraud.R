# Autor: Filipe de Paula Oliveira
# 
# Projeto 01 da Formação Cientista de Dados no curso Big Data Analytics com R
# e Microsoft Azure Machine Learning

# -- Importação de pacotes -- 

# Leitura e Manipulação de Dados
library(readr)
library(dplyr)
library(tidyr)
library(reshape2)
library(DMwR)
library(ROSE)

# Análise Gráfica
library(ggplot2)
library(corrplot)
library(GGally)

# Machine Learning
library(caret)
library(randomForest)
library(e1071)
library(ranger)
library(C50)
library(xgboost)

# -- Leitura do dataset --

colTypes <- cols(ip = 'i', app = 'i', device = 'i', os = 'i', channel = 'i', 
                 click_time = 'T', attributed_time = 'T', is_attributed = 'i')

# # Train dataset sample
# df_orig <- read_csv("talkingdata-adtracking-fraud-detection/train_sample.csv",
#                     col_types = colTypes)

# O dataset original tem uma proporção baixíssima da classe verdadeira (0.2%)
# O trecho abaixo gera um csv com uma proporção um pouco melhor (3%)

# # Lendo train dataset real e obtendo ~1.000.000 linhas aleatórias
# df_orig <- read_csv("talkingdata-adtracking-fraud-detection/train.csv",
#                     col_types = colTypes)
# df_total_true <- subset(df_orig, df_orig$is_attributed == 1)
# df_true <- df_total_true[sample(nrow(df_total_true), 30000), ]
# df_false <- df_orig[sample(nrow(df_orig), size = 970000), ]
# df_false <- filter(df_false, df_false$is_attributed == 0)
# df_orig <- rbind(df_false, df_true)

# # Gravando num arquivo para não ter que executar toda vez
# write_csv(df_orig, "trainTransf.csv")
# remove(df_total_true)
# remove(df_true)
# remove(df_false)
# remove(df_orig)

# Train dataset transformado final
df_inicial <- read_csv("trainTransf.csv", col_types = colTypes)

# -- Análise exploratória -- (pode ser comentada pra rodar apenas o modelo)

# Descrição do dataset

head(df_inicial)
str(df_inicial)
summary(df_inicial)

# Proporção da variável target em qtd e %

table(df_inicial$is_attributed)
round(prop.table(table(df_inicial$is_attributed))*100,3)

# Gráfico da contagem de valores unique

count_uniques <- df_inicial %>%
  summarise_all(n_distinct) %>%
  dplyr::select(1:5) %>%
  melt()
barplot(count_uniques$value,
        names.arg = count_uniques$variable,
        xlab = 'Variável',
        ylab = 'log(Count Uniques)',
        main = 'Valores Únicos por Variável',
        col = c('green','aquamarine','blue','chartreuse','purple')) %>%
  text(0, labels = round(count_uniques$value,2), pos=3, cex = 1, col = 'black')

# Gráfico de correlações

only_numeric <- dplyr::select(df_inicial,
                              c('ip','app','device','os','channel','is_attributed'))
corrplot(cor(only_numeric),
         method = 'color',
         diag = FALSE,
         addCoef.col = TRUE,
         title = 'Gráfico de Correlações',
         mar=c(0,0,1,0))

# Histogramas

ggplot(gather(only_numeric[1:5]), aes(x=value)) +
  geom_histogram(bins=10, fill = 'turquoise4') +
  facet_wrap(~key, scales = 'free_x')

# Pairplot (apenas primeiras 1000 linhas -> muito pesado)

ggpairs(only_numeric[0:1000,], mapping = aes(alpha = 0.02, pch = '.')) +
  ggtitle("Pairplot das variáveis numéricas (apenas 1000 linhas)") +
  theme_minimal()


# -- Pré-Processamento --

# Criando colunas de hora, minuto e segundo do clique; removendo target para normalização

df_transf <- df_inicial %>%
  mutate(click_hour = as.integer(lubridate::hour(df_inicial$click_time)),
         click_min = as.integer(lubridate::minute(df_inicial$click_time)),
         click_second = as.integer(lubridate::second(df_inicial$click_time)))
df_transf$click_time <- NULL
df_transf$attributed_time <- NULL
df_transf$is_attributed <- NULL

# Correlações com as novas variáveis

corrplot(cor(df_transf %>% mutate(is_attributed =df_inicial$is_attributed)),
         method = 'color',
         diag = FALSE,
         addCoef.col = TRUE,
         title = 'Gráfico de Correlações com Novas Variáveis',
         mar=c(0,0,1,0))

# Normalização e readicionando target

normalizar <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

df_norm <- as.data.frame(lapply(df_transf, normalizar)) %>%
  mutate(is_attributed = as.factor(df_inicial$is_attributed))

# Sampling 

splitIndex <- createDataPartition(y = df_norm$is_attributed, p = 0.7, list = FALSE)
df_train <- df_norm[splitIndex,]
df_test <- df_norm[-splitIndex,]

# Balanceamento de Classes (SMOTE ou ROSE)
# (smote) com o train_sample.csv -> train: perc.over = 30000, test: perc.over = 22000
# (smote) com o trainTransf.csv -> train: perc.over = 1600 ,test: perc.over = 1600

trainData <- SMOTE(is_attributed ~ ., df_train,
                   perc.over = 1600, perc.under=100)
testData <- SMOTE(is_attributed ~ ., df_test,
                  perc.over = 1600, perc.under=100)
# trainData <- ROSE(is_attributed ~ ., df_train)$data
# testData <- ROSE(is_attributed ~ ., df_test)$data
round(prop.table(table(trainData$is_attributed))*100,3)
round(prop.table(table(testData$is_attributed))*100,3)


# -- Machine Learning --

# Modelo #1 (Regressão Logistica)

modelo_lr <- glm(is_attributed ~ ., 
                 data = trainData,
                 family = 'binomial')
summary(modelo_lr)
predicoes_lr <- round(predict(modelo_lr, testData, type="response"))

confusionMatrix(table(data = predicoes_lr, 
                      reference = testData$is_attributed), 
                positive = '1')
auc_lr <- round(roc.curve(testData$is_attributed, predicoes_lr, plotit = F)$auc, 3)

# Modelo #2 (Naive Bayes)

modelo_nb <- naiveBayes(is_attributed ~ .,
                        data = trainData)
summary(modelo_nb)
predicoes_nb <- predict(modelo_nb, testData)

confusionMatrix(table(data = predicoes_nb, 
                      reference = testData$is_attributed), 
                positive = '1')
auc_nb <- round(roc.curve(testData$is_attributed, predicoes_nb, plotit = F)$auc, 3)

# Modelo #3 (Random Forest com pacote ranger)

modelo_rf <- ranger(is_attributed ~ .,
                    data = trainData)
modelo_rf
predicoes_rf <- predict(modelo_rf, testData)$predictions

confusionMatrix(table(data = predicoes_rf, 
                      reference = testData$is_attributed), 
                positive = '1')
auc_rf <- round(roc.curve(testData$is_attributed, predicoes_rf, plotit = F)$auc, 3)

# Modelo #4 (C5.0)

modelo_c50 <- C5.0(is_attributed ~ .,
                   data = trainData)
modelo_c50
predicoes_c50 <- predict(modelo_c50, testData)

confusionMatrix(table(data = predicoes_c50, 
                      reference = testData$is_attributed), 
                positive = '1')
auc_c50 <- round(roc.curve(testData$is_attributed, predicoes_c50, plotit = F)$auc, 3)

# Modelo #5 (XGBoost)

xgb_train <- xgb.DMatrix(data = data.matrix(trainData[,1:8]),
                         label = as.numeric(as.character(trainData$is_attributed)))
xgb_test <- xgb.DMatrix(data = data.matrix(testData[,1:8]),
                        label = as.numeric(as.character(testData$is_attributed)))

modelo_xgb <- xgboost(data=xgb_train,
                      max.depth=3,
                      nrounds=100,
                      objective = "binary:logistic")
modelo_xgb
predicoes_xgb <- round(predict(modelo_xgb, xgb_test))

confusionMatrix(table(data = predicoes_xgb, 
                      reference = testData$is_attributed), 
                positive = '1')
auc_xgb <- round(roc.curve(testData$is_attributed, predicoes_xgb, plotit = F)$auc, 3)

# Todas ROC Curves
legendLabels = c(sprintf('Logistic Regression (%s)', auc_lr),
                 sprintf('Naive Bayes (%s)', auc_nb),
                 sprintf('Random Forest (%s)', auc_rf),
                 sprintf('C5.0 (%s)', auc_c50),
                 sprintf('XGBoost (%s)', auc_xgb))
legendColors = c('blue','purple','green','red', 'turquoise')
roc.curve(testData$is_attributed, predicoes_lr, plotit = T, col = legendColors[1], lwd=2)
roc.curve(testData$is_attributed, predicoes_nb, add.roc = T, col = legendColors[2], lwd=2)
roc.curve(testData$is_attributed, predicoes_rf, add.roc = T, col = legendColors[3], lwd=2)
roc.curve(testData$is_attributed, predicoes_c50, add.roc = T, col = legendColors[4], lwd=2)
roc.curve(testData$is_attributed, predicoes_xgb, add.roc = T, col = legendColors[5], lwd=2)
rect(0, 1.1, 1, 1.7, xpd=TRUE, col="white", border="white") #apaga o titulo
title("ROC Curves") #coloca o novo titulo
legend("bottomright",legend = legendLabels, col = legendColors, lty=1, cex=0.8, lwd = 2, title = "Method (%)")