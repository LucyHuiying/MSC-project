rm(list=ls())
PATH<-'G:/poject/New3'
setwd(PATH)
path<-paste0(PATH,'/Result/');DPI=600
library(rio)
library(dplyr)
library(prodlim)
library(survminer)
library(survival)
library(ggplot2)
library(MLmetrics)
library(ggsci)
library(scales)
library(caret)
library(tidyverse)
library(Hmisc)
library(tableone)
library(naniar)
library(lubridate)
library(simputation)
library(survminer)
library(survival)
library(ggsci)
library(ggpubr)
library(ggplot2)
library(rms)
library(survey)
library(gtsummary)
library(cowplot)
library(xtable)
library(flextable)
library(officer)
library(ggprism)
library(export)
library(reportROC)

setwd(paste0(PATH,'/Result'))


#----------------------------------------------------------------------
load('dtx-LR.RData')

var_LR=varn

df1=dtx
#----------------------------------------------------------------------
load('dtx-SVM.RData')
var_SVM=varn
xx_SVM=xx

df2=dtx %>% dplyr::select(NACCID,SVM)

#---------------------------------------------------------------------
load('dtx-randomForest.RData')
var_RF=varn
xx_RF=xx

df3=dtx %>% dplyr::select(NACCID,randomForest)

#---------------------------------------------------------------------
load('dtx-XGBOOST.RData')
var_XGB=varn
xx_XGB=xx

df4=dtx %>% dplyr::select(NACCID,XGBOOST)


dt=left_join(df1,df2,by='NACCID')
dt=left_join(dt,df3,by='NACCID')
dt=left_join(dt,df4,by='NACCID')


dt$status=dt$dementia



# Function to calculate F1 score
calculate_f1 <- function(conf_matrix) {
  TP <- conf_matrix$table[2, 2]  # True Positive
  FP <- conf_matrix$table[1, 2]  # False Positive
  FN <- conf_matrix$table[2, 1]  # False Negative
  
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  F1 <- 2 * (precision * recall) / (precision + recall)
  return(F1)
}

plot_confusion_matrix <- function(conf_matrix, model_name, path) {
  cm_table <- as.data.frame(conf_matrix$table)
  cm_table$Prediction <- factor(cm_table$Prediction, levels = rev(levels(cm_table$Prediction)))
  
  # Calculate proportions for confusion matrix
  cm_table$Freq <- cm_table$Freq / sum(cm_table$Freq)
  
  # Define a color palette similar to 'plasma'
  plasma_palette <- c("#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636", "#f0f921")
  
  # Plot confusion matrix using ggplot2 with a custom "plasma"-like colormap
  p <- ggplot(data = cm_table, aes(x = Reference, y = Prediction)) +
    geom_tile(aes(fill = Freq), color = "white") +
    geom_text(aes(label = sprintf("%.2f", Freq)), size = 6, fontface = "bold", color = "black") +
    scale_fill_gradientn(colors = plasma_palette, limits = c(0, 1), name = NULL) +
    labs(title = paste("Confusion Matrix -", model_name),
         x = "Actual",
         y = "Predicted") +
    theme_minimal() +
    theme(
      panel.background = element_rect(fill = "white", color = NA),  # Set the background to white
      panel.grid.major = element_blank(),  # Remove major gridlines
      panel.grid.minor = element_blank(),  # Remove minor gridlines
      axis.text.x = element_text(color = "black", size = 12),  # Set axis labels to be visible and black
      axis.text.y = element_text(color = "black", size = 12),
      plot.title = element_text(hjust = 0.5)  # Center the title
    )
  
  # Save plot to file
  ggsave(filename = paste0(path, "Confusion_Matrix_", model_name, ".png"), plot = p, width = 5, height = 4, bg = "white")
}


#-------------------------------------------------------------------------------
dtt=subset(dt,dt$setg=='Training')

rt=reportROC(gold = dtt$status,predictor = dtt$LR,important = "se")
rt1=cbind(data.frame(Model='LR'),rt)
rt1

rt=reportROC(gold = dtt$status,predictor = dtt$SVM,important = "se")
rt2=cbind(data.frame(Model='SVM'),rt)
rt2

rt=reportROC(gold = dtt$status,predictor = dtt$randomForest,important = "se")
rt3=cbind(data.frame(Model='RF'),rt)
rt3

rt=reportROC(gold = dtt$status,predictor = dtt$XGBOOST,important = "se")
rt4=cbind(data.frame(Model='XGB'),rt)
rt4

rst=rbind(rt1,rt2,rt3,rt4)
rst

# Extract Cutoff thresholds from reportROC results
threshold_LR <- rt1$Cutoff
threshold_SVM <- rt2$Cutoff
threshold_RF <- rt3$Cutoff
threshold_XGB <- rt4$Cutoff

# Calculate confusion matrix using extracted thresholds
conf_matrix_LR <- confusionMatrix(as.factor(ifelse(dtt$LR >= threshold_LR, 1, 0)), as.factor(dtt$status))
conf_matrix_SVM <- confusionMatrix(as.factor(ifelse(dtt$SVM > threshold_SVM, 1, 0)), as.factor(dtt$status))
conf_matrix_RF <- confusionMatrix(as.factor(ifelse(dtt$randomForest > threshold_RF, 1, 0)), as.factor(dtt$status))
conf_matrix_XGB <- confusionMatrix(as.factor(ifelse(dtt$XGBOOST > threshold_XGB, 1, 0)), as.factor(dtt$status))

# Plot and save confusion matrices directly using ggplot2
plot_confusion_matrix(conf_matrix_LR, "LR_Validation", path)
plot_confusion_matrix(conf_matrix_SVM, "SVM_Validation", path)
plot_confusion_matrix(conf_matrix_RF, "RF_Validation", path)
plot_confusion_matrix(conf_matrix_XGB, "XGB_Validation", path)

# Combine ROC results and export to Excel
rst <- rbind(
  cbind(data.frame(Model = 'LR'), rt1),
  cbind(data.frame(Model = 'SVM'), rt2),
  cbind(data.frame(Model = 'RF'), rt3),
  cbind(data.frame(Model = 'XGB'), rt4)
)

export(rst,'Traning.xlsx')


#------------------------------------------------------------------------------roc
roc1<-roc(dtt$status,dtt$LR,ci=T,smooth=F)
roc1

roc2<-roc(dtt$status,dtt$SVM,ci=T,smooth=F)
roc2

roc3<-roc(dtt$status,dtt$randomForest,ci=T,smooth=F)
roc3

roc4<-roc(dtt$status,dtt$XGBOOST,ci=T,smooth=F)
roc4

#--------------------------------------------------------------------------ROC
p1<-ggroc(list( "LR-AUC: 0.902"=roc1,
                "SVM-AUC: 0.967"=roc2,
                "RF-AUC: 1.000"=roc3,
                "XGB-AUC: 0.939"=roc4
),legacy.axes = TRUE,linetype=1,size=1)+
  scale_y_continuous(expand = c(0,0),limits = c(0,1),breaks = seq(0,1,0.2))+
  scale_x_continuous(expand = c(0,0),limits = c(0,1.02),breaks = seq(0,1,0.2))+
  theme_bw()+ geom_segment(aes(x = 0, xend = 1,
                               y = 0, yend = 1), color="black",   linetype=2)+xlab("1-Specificity") + ylab("Sensitivity")+
  theme_prism(base_size = 20,base_line_size = 0.75, base_family = "serif",base_fontface ="plain" )+
  scale_color_lancet()+
  theme(axis.line=element_line(colour="black"))+
  theme(axis.text.x = element_text(size = 18,angle = 0,vjust = 0,hjust = 0.5))+
  theme(axis.text.y = element_text(size =18))+
  theme(legend.position=c(0.63,0.25))+
  annotate("text",  x=0.68,y=0.15,label="",size=5)+
  ggtitle("")
p1
ne<-"ROC-traning";w<-8;h<-7
ggsave(filename=paste0(path,ne,".png"),width=w,height=h,dpi=DPI)
ggsave(filename=paste0(path,ne,".tiff"),width=w,height=h,dpi=DPI,compression = 'lzw')





#-------------------------------------------------------------------------------
dtt=subset(dt,dt$setg=='Validation')


rt=reportROC(gold = dtt$status,predictor = dtt$LR,important = "se")
rt1=cbind(data.frame(Model='LR'),rt)
rt1

rt=reportROC(gold = dtt$status,predictor = dtt$SVM,important = "se")
rt2=cbind(data.frame(Model='SVM'),rt)
rt2

rt=reportROC(gold = dtt$status,predictor = dtt$randomForest,important = "se")
rt3=cbind(data.frame(Model='RF'),rt)
rt3

rt=reportROC(gold = dtt$status,predictor = dtt$XGBOOST,important = "se")
rt4=cbind(data.frame(Model='XGB'),rt)
rt4

rst=rbind(rt1,rt2,rt3,rt4)
rst

# Extract Cutoff thresholds from reportROC results
threshold_LR <- as.numeric(rt1$Cutoff)
threshold_SVM <- as.numeric(rt2$Cutoff)
threshold_RF <- as.numeric(rt3$Cutoff)
threshold_XGB <- as.numeric(rt4$Cutoff)

# Calculate confusion matrix using extracted thresholds
conf_matrix_LR <- confusionMatrix(as.factor(ifelse(dtt$LR >= threshold_LR, 1, 0)), as.factor(dtt$status))
conf_matrix_SVM <- confusionMatrix(as.factor(ifelse(dtt$SVM >= threshold_SVM, 1, 0)), as.factor(dtt$status))
conf_matrix_RF <- confusionMatrix(as.factor(ifelse(dtt$randomForest >= threshold_RF, 1, 0)), as.factor(dtt$status))
conf_matrix_XGB <- confusionMatrix(as.factor(ifelse(dtt$XGBOOST >= threshold_XGB, 1, 0)), as.factor(dtt$status))

# Calculate F1 scores
f1_LR <- calculate_f1(conf_matrix_LR)
f1_SVM <- calculate_f1(conf_matrix_SVM)
f1_RF <- calculate_f1(conf_matrix_RF)
f1_XGB <- calculate_f1(conf_matrix_XGB)

# Print F1 scores
cat("F1 Scores:\n")
cat("LR: ", f1_LR, "\n")
cat("SVM: ", f1_SVM, "\n")
cat("RF: ", f1_RF, "\n")
cat("XGB: ", f1_XGB, "\n")


# Plot and save confusion matrices directly using ggplot2
plot_confusion_matrix(conf_matrix_LR, "LR", path)
plot_confusion_matrix(conf_matrix_SVM, "SVM", path)
plot_confusion_matrix(conf_matrix_RF, "RF", path)
plot_confusion_matrix(conf_matrix_XGB, "XGB", path)

# Combine ROC results and export to Excel
rst <- rbind(
  cbind(data.frame(Model = 'LR'), rt1),
  cbind(data.frame(Model = 'SVM'), rt2),
  cbind(data.frame(Model = 'RF'), rt3),
  cbind(data.frame(Model = 'XGB'), rt4)
)

export(rst,'Testing.xlsx')


#------------------------------------------------------------------------------roc
roc1<-roc(dtt$status,dtt$LR,ci=T,smooth=F)
roc1

roc2<-roc(dtt$status,dtt$SVM,ci=T,smooth=F)
roc2

roc3<-roc(dtt$status,dtt$randomForest,ci=T,smooth=F)
roc3

roc4<-roc(dtt$status,dtt$XGBOOST,ci=T,smooth=F)
roc4

#--------------------------------------------------------------------------ROC
p1<-ggroc(list( "LR-AUC: 0.860"=roc1,
                "SVM-AUC: 0.850"=roc2,
                "RF-AUC: 0.879"=roc3,
                "XGB-AUC: 0.882"=roc4
),legacy.axes = TRUE,linetype=1,size=1)+
  scale_y_continuous(expand = c(0,0),limits = c(0,1),breaks = seq(0,1,0.2))+
  scale_x_continuous(expand = c(0,0),limits = c(0,1.02),breaks = seq(0,1,0.2))+
  theme_bw()+ geom_segment(aes(x = 0, xend = 1,
                               y = 0, yend = 1), color="black",   linetype=2)+xlab("1-Specificity") + ylab("Sensitivity")+
  theme_prism(base_size = 20,base_line_size = 0.75, base_family = "serif",base_fontface ="plain" )+
  scale_color_lancet()+
  theme(axis.line=element_line(colour="black"))+
  theme(axis.text.x = element_text(size = 18,angle = 0,vjust = 0,hjust = 0.5))+#横坐标文字设置
  theme(axis.text.y = element_text(size =18))+
  theme(legend.position=c(0.63,0.25))+
  annotate("text",  x=0.68,y=0.15,label="",size=5)+
  ggtitle("")
p1
ne<-"ROC-testing";w<-8;h<-7
ggsave(filename=paste0(path,ne,".png"),width=w,height=h,dpi=DPI)
ggsave(filename=paste0(path,ne,".tiff"),width=w,height=h,dpi=DPI,compression = 'lzw')



