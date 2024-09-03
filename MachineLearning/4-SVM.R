rm(list=ls())
PATH<-'G:/poject/tez'
setwd(PATH)
path<-paste0(PATH,'/Result/');DPI=600
library(rio)
library(dplyr)
library(prodlim)
library(survminer)
library(survival)
library(ggplot2)
library(ggsci)
library(scales)
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

load('dff.RData')

var0=var
var0

df=dff
#--------------------------------------------------------------------------
df$status=df$dementia
table(df$status)



#--------------------------------------------
df$Status=factor(df$status)

FML<-as.formula(paste0("status~",paste0(var0,collapse = "+")))
FML2<-as.formula(paste0("Status~",paste0(var0,collapse = "+")))

table(df$setg)
#--------------------------------------------------------------------
dfx=subset(df,df$setg=='Training')
dft=subset(df,df$setg=='Validation')


nn=10
xtrain=dfx[,var0]
ytrain=as.factor(dfx[,'Status'])



library(caret)
library(iml)
library(tidymodels)
library(mlr)

levels(dfx$Status)


library(e1071)
#-------------------------------------------------------------------------SVM
dfxx=dfx
for (i in c(var0)) {dfxx[,i]=as.numeric(dfxx[,i])}
dftt=dft
for (i in c(var0)) {dftt[,i]=as.numeric(dftt[,i])}

dff=df
for (i in c(var0)) {dff[,i]=as.numeric(dff[,i])}

xtrain=dfxx[,var0]
ytrain=as.factor(dfxx[,'Status'])

Fit2=svm(FML2,data=dfxx)

#------------------------------

param_grid <- expand.grid(
    sigma = c(0.1, 1, 10),
    C = c(0.1, 1, 10))

ctrl <- trainControl(method = "cv", number = nn, verboseIter = FALSE)

set.seed(123)

tuned_model <- caret ::train(
    x = xtrain,
    y = ytrain,
    method = "svmRadial",
    tuneGrid = param_grid,
    trControl = ctrl)

xx=data.frame(print(tuned_model))
xx
set.seed(123)

Fit2=svm(FML2,data=dfxx,cost=xx[2,3],probability = T,kernel='radial')
dff$SVM <- attributes(predict(Fit2,newdata=dff,probability=T))$probabilities[,1]
dff$SVM

Fit22=svm(FML,data=dfxx,cost=xx[2,3],probability = T,kernel='radial')
predictor <- Predictor$new(Fit22,data = dfxx,type = 'prob')
imp <- FeatureImp$new(predictor,loss = "rmse") 


plot(imp)



importance <- plot(imp)
importance<-data.frame(importance$data)


dtp<-importance
for (i in c(2:4)) {dtp[,i]=(dtp[,i]-1)*100}

names(dtp)[3]<-'importance'
names(dtp)[1]<-'variable'
dtp$variable=as.character(dtp$variable)

dtp<-dplyr::arrange(dtp,-importance)

dtp$variable<-as_factor(dtp$variable)
dtp$variable<-fct_rev(dtp$variable)
dtp$weight0=sprintf('%0.3f',dtp$importance)
ybreaks <- pretty(c(0, max(dtp$importance,na.rm = T)*1.01),n=5,high.u.bias = 2)
ybreaks

ggplot(dtp[1:25,],aes(x=variable,y=importance))+
    geom_col(aes(fill=variable),show.legend = F)+
    theme_prism(base_size = 16,base_line_size = 0.75,base_fontface ="plain",base_family = "serif" )+
    scale_y_continuous(expand = c(0,0),limits = c(0,max(ybreaks)*1.01),breaks = ybreaks)+
    theme(axis.line=element_line(colour="black"))+
    theme(axis.text.x = element_text(size = 16,angle = 0,vjust = 0,hjust = 0.5))+#
    theme(axis.text.y = element_text(size =16))+
    labs(x= paste0('Variable'),y = paste0('importance'))+
    geom_text(aes(y=importance+0.00*max(ybreaks),label=weight0),vjust=0.5,hjust=0,position = position_dodge(0.9),
              colour="black",size=5,angle=0,family="serif")+
    coord_flip()
ne<-"Feature-SVMModel";w=12;h=16
ggsave(filename=paste0(path,ne,".png"),width=w,height=h,dpi=DPI)
ggsave(filename=paste0(path,ne,".tiff"),width=w,height=h,dpi=DPI,compression = 'lzw')
export(dtp,paste0(ne,'.xlsx'))



varn=as.character( dtp$variable)
varn

dtx= dff %>% dplyr::select(NACCID,PACKET,NACCDAYS,NACCIDEM,
                           NACCETPR,dementia,set,setg,SVM)

save(dtx,xx,varn,dff,imp,dfxx,dftt,file = 'dtx-SVM.RData')





actual_length = min(40, length(dtp$variable))
varn = as.character(dtp$variable[1:actual_length])
varn


#------------------------------------------------------------------------计算不同变量下的AUC
x=4
var=varn[1:x]
var
FML2<-as.formula(paste0("Status~",paste0(var,collapse = "+")))
FML2
data=dftt
Fitt <- svm(FML2,data=dfxx,cost=xx[2,3],probability = T,kernel='radial')

data$SVM= attributes(predict(Fitt,newdata=data,probability=T))$probabilities[,1]
data$SVM


ROC<-reportROC(gold = data$status,predictor = data$SVM,important = "se")
ROC$P=ifelse(ROC$P=='0.000','<0.001',ROC$P)
result<-data.frame(
    N=x,
    AUC=ROC[,'AUC'],
    lower=ROC[,'AUC.low'],
    high=ROC[,'AUC.up'],
    pvalue=ROC[,'P'])
result


varn
get_auc=function(x){
    varn
    var=varn[1:6]
    FML2<-as.formula(paste0("Status~",paste0(var,collapse = "+")))
    FML2
    Fitt <- svm(FML2,data=dfxx,cost=xx[2,3],probability = T,kernel='radial')
    
    data$SVM=attributes(predict(Fitt,newdata=data,probability=T))$probabilities[,1]
    ROC<-reportROC(gold = data$status,predictor = data$SVM,important = "se")
    ROC$P=ifelse(ROC$P=='0.000','<0.001',ROC$P)
    result<-data.frame(
        N=x,
        AUC=ROC[,'AUC'],
        lower=ROC[,'AUC.low'],
        high=ROC[,'AUC.up'],
        pvalue=ROC[,'P'])
    return(result)
}

#--------------------------------------------------------
data=dfxx
#-------------
result<-lapply(c(1:6),get_auc)
#---------------
rst<-ldply(result,data.frame)
rst

save(rst,file = 'n-auc-SVM-train.RData')
export(rst,'n-auc-SVM-train.xlsx')



#--------------------------------------------------------
data=dftt
dftt
# get_auc(5)
#-------------
result<-lapply(c(1:6),get_auc)
#---------------
rst<-ldply(result,data.frame)
rst

save(rst,file = 'n-auc-SVM-test.RData')
export(rst,'n-auc-SVM-test.xlsx')

var
