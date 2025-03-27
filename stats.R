install.packages("readxl")
install.packages("fastDummies")
install.packages("corrplot")
install.packages("pheatmap")
library("readxl")
library("fastDummies")
library("corrplot")
library("pheatmap")

data <- read.csv("heart_cleveland_upload.csv"); data
cor(data)

data$sex <- as.numeric(data$sex)
data$sex <- factor(data$sex, levels = c(0,1), labels = c("F","M"));data$sex

data$cp <- as.numeric(data$cp)
data$cp <- factor(data$cp, levels = c(0,1,2,3), labels = c("typical","atypical","non-anginal","asymptomatic"));data$cp

data$restecg <- as.numeric(data$restecg)
data$restecg <- factor(data$restecg, levels = c(0,1,2), labels = c("Normal","ST-T","hypertrophy"));data$restecg

data$slope <- as.numeric(data$slope)
data$slope <- factor(data$slope, levels = c(0,1,2), labels = c("upsloping","flat","downsloping"));data$slope

data$thal <- as.numeric(data$thal)
data$thal <- factor(data$thal, levels = c(0,1,2), labels = c("Normal","Fixed Defect", "Reversable"));data$thal

data_dum <- dummy_cols(data, select_columns = c("sex","cp","restecg","slope","thal"),remove_first_dummy = TRUE, remove_selected_columns = TRUE)
data_dum

cor_mat <- cor(data_dum)
COL2 <- colorRampPalette(c("blue", "white", "red"))(200)  # Azul (-1) → Branco (0) → Vermelho (+1)
corrplot(cor_mat, method = "color", type = "full", col = COL2, tl.col = "black", tl.srt = 45)
pheatmap(cor_mat, color = colorRampPalette(c("blue", "white", "red"))(50))
