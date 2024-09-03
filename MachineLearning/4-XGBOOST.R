rm(list=ls())
PATH<-'G:/poject/New3'
setwd(PATH)
path<-paste0(PATH,'/Result/');DPI=600
library(rio)
library(dplyr)
library(prodlim)
library(plyr)
library(survminer)
library(shapviz)
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

load('dff2.RData')

var0=var
var0

df=dff
#--------------------------------------------------------------------------结局
df$status=df$dementia
table(df$status)




df$Status=factor(df$status)

FML<-as.formula(paste0("status~",paste0(var0,collapse = "+")))
FML2<-as.formula(paste0("Status~",paste0(var0,collapse = "+")))

table(df$setg)

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

dfxx=dfx
for (i in c(var0)) {dfxx[,i]=as.numeric(dfxx[,i])}
dftt=dft
for (i in c(var0)) {dftt[,i]=as.numeric(dftt[,i])}

dff=df
for (i in c(var0)) {dff[,i]=as.numeric(dff[,i])}

xtrain=dfxx[,var0]
ytrain=as.factor(dfxx[,'Status'])



library(xgboost)

dtrain <- xgb.DMatrix(data = as.matrix(dfxx[var0]), label = dfxx$status)

params <- list(objective = "binary:logistic", eval_metric = "logloss", eta = 0.1, max_depth = 3)

nrounds <- 100

xgb_model <- xgboost(params = params, data = dtrain, nrounds = nrounds)



ctrl <- trainControl(
    method = "cv",  
    number = nn,     
    verboseIter = FALSE)


param_grid <- expand.grid(
    nrounds = c(100, 200),
    max_depth = c(3, 6), 
    eta = c(0.1), 
    gamma = c(0, 0.1), 
    colsample_bytree = c(0.8), 
    min_child_weight = c(1, 3),
    subsample = c(0.8)) 

set.seed(123)

xgb_model <- caret ::train(
    x = dfxx[,var0],
    y = dfxx$Status,
    method = "xgbTree",
    trControl = ctrl,
    tuneGrid = param_grid)


print(xgb_model$bestTune)
xx=data.frame(print(xgb_model$bestTune))
xx


params <- list(objective = "binary:logistic", eval_metric = "auc", 
               eta = xx[1,3], max_depth = xx[1,2], gamma = xx[1,4],
               colsample_bytree = xx[1,5],
               min_child_weight = xx[1,6],
               subsample = xx[1,7])


set.seed(123)
Fit4=xgboost(data = as.matrix(dfxx[, c(var0)]),
             label = dfxx$status,
             nrounds = 200,
             objective = "binary:logistic", eval_metric = "auc", 
             eta = xx[1,3], max_depth = xx[1,2], gamma = xx[1,4],
             colsample_bytree = xx[1,5],
             min_child_weight = xx[1,6],
             subsample = xx[1,7] )

#Fit4 <- xgb.train(params = params, data = dtrain, nrounds = 200)
dff$XGBOOST=predict(object = Fit4,newdata = xgb.DMatrix(as.matrix(dff[,var0])), type = "response")
dff$XGBOOST



imp<-xgb.importance(feature_names=colnames(var),model=Fit4)
dtp=imp
names(dtp)[1]<-'variable'

dtp$variable<-as_factor(dtp$variable) 

#------------------------------------------------------
dtp$weight0=sprintf('%0.2f',dtp$Gain) 
#--------------------------------------------------------------
dtp$variable<-fct_rev(dtp$variable) 
ybreaks <- pretty(c(0, max(dtp$Gain,na.rm = T)*1.1),n=5,high.u.bias = 2)
ybreaks
ggplot(data=dtp[1:10,],mapping = aes(x=variable,y=Gain))+
    geom_col(aes(fill=variable),show.legend = F)+
    theme_prism(base_size = 16,base_line_size = 0.75,base_fontface ="plain",base_family = "serif" )+
    scale_y_continuous(expand = c(0,0),limits = c(min(ybreaks),max(ybreaks)*1.01),breaks = ybreaks)+
    theme(axis.line=element_line(colour="black"))+
    theme(axis.text.x = element_text(size =16,angle = 0,vjust = 0,hjust = 0.5))+
    theme(axis.text.y = element_text(size =16))+
    labs(x= 'Variable',y = 'importance')+
    geom_text(aes(y=Gain,label=weight0),hjust=-0.1,position = position_dodge(0.9),
              colour="black",size=5,angle=0,family="serif") +
    coord_flip() 
ne<-"Featureimportant-xgboostmodel";w=12;h=16
graph2pdf(file=paste0(path,ne,".pdf"),width=w,height=h)
graph2ppt(file=paste0(path,ne,".ppt"),width=w,height=h)
ggsave(filename=paste0(path,ne,".png"),width=w,height=h,dpi=DPI)
ggsave(filename=paste0(path,ne,".tiff"),width=w,height=h,dpi=DPI,compression = 'lzw')

export(dtp,paste0(ne,'.xlsx'))



shap_xgboost <- shapviz(Fit4, as.matrix(dfxx[, var0]))
sv_waterfall(shap_xgboost,row_id=45)
sv_importance(shap_xgboost,max_display=25)+ theme_bw()
sv_importance(shap_xgboost,kind='beeswarm')+theme_bw()
sv_force(shap_xgboost, row_id = 40)
sv_dependence(shap_xgboost, 'NACCAGE', alpha = 0.5, size = 1.5)+ theme_bw()
sv_dependence2D(shap_xgboost, 'NACCAGE','NACCCOGF')+ theme_bw()


varn=as.character(dtp$variable)
varn

dtx= dff %>% dplyr::select(NACCID,PACKET,NACCDAYS,NACCIDEM,
                           NACCETPR,dementia,set,setg,XGBOOST)

save(dtx,xx,varn,dff,imp,dfxx,dftt,file = 'dtx-XGBOOST.RData')





varn=as.character( dtp$variable[1:40])
varn


#------------------------------------------------------------------------计算不同变量下的AUC
x=4
var=varn[1:x]
var

data=dftt
Fitt <- xgboost(data = as.matrix(dfxx[, c(var)]),
                label = dfxx$status,
                nrounds = 200,
                objective = "binary:logistic", eval_metric = "auc", 
                eta = xx[1,3], max_depth = xx[1,2], gamma = xx[1,4],
                colsample_bytree = xx[1,5],
                min_child_weight = xx[1,6],
                subsample = xx[1,7] )

data$XGBOOST= predict(object = Fitt,newdata = xgb.DMatrix(as.matrix(data[,var])), type = "response")
data$XGBOOST


ROC<-reportROC(gold = data$status,predictor = data$XGBOOST,important = "se")
ROC$P=ifelse(ROC$P=='0.000','<0.001',ROC$P)
result<-data.frame(
    N=x,
    AUC=ROC[,'AUC'],
    lower=ROC[,'AUC.low'],
    high=ROC[,'AUC.up'],
    pvalue=ROC[,'P'])
result



get_auc=function(x){
    var=varn[1:x]
    Fitt <- xgboost(data = as.matrix(dfxx[, c(var)]),
                    label = dfxx$status,
                    nrounds = 200,
                    objective = "binary:logistic", eval_metric = "auc", 
                    eta = xx[1,3], max_depth = xx[1,2], gamma = xx[1,4],
                    colsample_bytree = xx[1,5],
                    min_child_weight = xx[1,6],
                    subsample = xx[1,7] )
    
    data$XGBOOST=predict(object = Fitt,newdata = xgb.DMatrix(as.matrix(data[,var])), type = "response")
    
    ROC<-reportROC(gold = data$status,predictor = data$XGBOOST,important = "se")
    ROC$P=ifelse(ROC$P=='0.000','<0.001',ROC$P)
    result<-data.frame(
        N=x,
        AUC=ROC[,'AUC'],
        lower=ROC[,'AUC.low'],
        high=ROC[,'AUC.up'],
        pvalue=ROC[,'P'])
    return(result)
}

data=dfxx

result<-lapply(c(1:40),get_auc)

rst<-ldply(result,data.frame)
rst

save(rst,file = 'n-auc-XGBOOST-train.RData')
export(rst,'n-auc-XGBOOST-train.xlsx')




data=dftt
get_auc(5)

result<-lapply(c(1:40),get_auc)

rst<-ldply(result,data.frame)
rst

save(rst,file = 'n-auc-XGBOOST-test.RData')
export(rst,'n-auc-XGBOOST-test.xlsx')





