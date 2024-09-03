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
library(ggsci)
library(scales)
library(tidyverse)
library(reportROC)
library(Hmisc)
library(tableone)
library(naniar)
library(lubridate)
library(simputation)
library(survminer)
library(survival)
library(ggsci)
library(ggpubr)
library(ggprism)
library(ggplot2)
library(rms)
library(survey)
library(gtsummary)
# library(nhanesR)
library(cowplot)
library(xtable)
library(flextable)
library(officer)

setwd(paste0(PATH,'/Result'))

load('dff2.RData')

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


library(randomForest)
library(caret)
levels(dfx$Status)

#-------------------------------------------------------------------------
ran_model <- randomForest(FML2,data = dfx, ntree = 100,importance=TRUE
                          ,proximity = TRUE
                          ,strata = 'y')
print(ran_model)

plot(ran_model)
plot(seq(ran_model$ntree),ran_model$err.rate[,1],type = 'l',xlab = 'decisiontree',ylab = 'OOB')

ctrl <- trainControl(method = "cv", number = nn) 

grid <- expand.grid(mtry = c(2, 4, 6,8,10)) 
set.seed(123)

ntree<-which.min(ran_model$err.rate[,1])
ntree


set.seed(123)
numtry<-tuneRF(xtrain,ytrain
               ,stepFactor = 0.5 
               ,plot = TRUE
               ,ntreeTry = ntree
               ,trace = T
               ,imporve=0.05)

numtry<-data.frame(numtry)
mtry<-numtry[which.min(numtry$OOBError),'mtry']
mtry


rf_model <- train(x = xtrain, y = ytrain,
                  method = "rf",
                  trControl = ctrl,
                  tuneGrid = grid)


print(rf_model)


grid <- expand.grid(mtry = c(10))  


modellist <- list()

for (ntree in c(100,200, 300)) {
    set.seed(123)
    fit <- train(x = xtrain, y = ytrain, method="rf", 
                 metric="Accuracy", tuneGrid=grid, 
                 trControl=ctrl, ntree=ntree)
    key <- toString(ntree)
    modellist[[key]] <- fit
}

# compare results
results <- resamples(modellist)

summary(results)

Fit3 <- randomForest(FML2,data = dfx,
                     ntree = ntree   
                     ,importance=TRUE
                     ,mtry = mtry
                     ,proximity = TRUE
                     ,strata = 'y')
df$randomForest=predict(Fit3,newdata=df,type = 'prob')[,2]
df$randomForest


importance <- Fit3$importance
importance<-data.frame(importance)
colnames(importance)

importance$name <- rownames(importance)
dtp<-importance[,c(5,4)]
names(dtp)[2]<-'importance'
names(dtp)[1]<-'variable'
dtp<-dplyr::arrange(dtp,-importance)

dtp$variable<-as_factor(dtp$variable)
dtp$variable<-fct_rev(dtp$variable)
dtp$weight0=sprintf('%0.3f',dtp$importance)
ybreaks <- pretty(c(0, max(dtp$importance,na.rm = T)*1.1),n=5,high.u.bias = 2)
ybreaks

ggplot(dtp[1:10,],aes(x=variable,y=importance))+
    geom_col(aes(fill=variable),show.legend = F)+
    theme_prism(base_size = 16,base_line_size = 0.75,base_fontface ="plain",base_family = "serif" )+
    scale_y_continuous(expand = c(0,0),limits = c(min(ybreaks),max(ybreaks)*1.01),breaks = ybreaks)+
    theme(axis.line=element_line(colour="black"))+
    theme(axis.text.x = element_text(size = 16,angle = 0,vjust = 0,hjust = 0.5))+
    theme(axis.text.y = element_text(size =16))+
    labs(x= paste0('Variable'),y = paste0('importance'))+
    geom_text(aes(y=importance+0.00*max(ybreaks),label=weight0),vjust=0.5,hjust=0,position = position_dodge(0.9),
              colour="black",size=5,angle=0,family="serif")+
    coord_flip()
ne<-"Featureimportant-randomForestmodel";w=12;h=16
#graph2pdf(file=paste0(path,ne,".pdf"),width=w,height=h)
#graph2ppt(file=paste0(path,ne,".ppt"),width=w,height=h)
ggsave(filename=paste0(path,ne,".png"),width=w,height=h,dpi=DPI)
ggsave(filename=paste0(path,ne,".tiff"),width=w,height=h,dpi=DPI,compression = 'lzw')
export(dtp,paste0(ne,'.xlsx'))



varn=as.character( dtp$variable)
varn

dtx= df %>% dplyr::select(NACCID,PACKET,NACCDAYS,NACCIDEM,
                          NACCETPR,dementia,set,setg,randomForest)

save(dtx,ntree,mtry,varn,file = 'dtx-randomForest.RData')






varn=as.character( dtp$variable[1:40])
varn


#------------------------------------------------------------------------
x=4
var=varn[1:x]
var
FML2<-as.formula(paste0("Status~",paste0(var,collapse = "+")))
FML2
data=dft
Fitt <- randomForest(FML2,data = dfx,
                     ntree = ntree   
                     ,importance=TRUE
                     ,mtry = mtry
                     ,proximity = TRUE
                     ,strata = 'y')
data$randomForest=predict(Fitt,newdata=data,type = 'prob')[,2]
data$randomForest

ROC<-reportROC(gold = data$status,predictor = data$randomForest,important = "se")
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
    FML2<-as.formula(paste0("Status~",paste0(var,collapse = "+")))
    FML2
    Fitt <- randomForest(FML2,data = dfx,
                         ntree = ntree   
                         ,importance=TRUE
                         ,mtry = mtry
                         ,proximity = TRUE
                         ,strata = 'y')
    data$randomForest=predict(Fitt,newdata=data,type = 'prob')[,2]
    ROC<-reportROC(gold = data$status,predictor = data$randomForest,important = "se")
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
data=dfx
#-------------
result<-lapply(c(1:40),get_auc)
#---------------
rst<-ldply(result,data.frame)
rst

save(rst,file = 'n-auc-randomForest-train.RData')
export(rst,'n-auc-randomForest-train.xlsx')



#--------------------------------------------------------
data=dft
get_auc(5)
#-------------
result<-lapply(c(1:40),get_auc)
#---------------
rst<-ldply(result,data.frame)
rst

save(rst,file = 'n-auc-randomForest-test.RData')
export(rst,'n-auc-randomForest-test.xlsx')

var
