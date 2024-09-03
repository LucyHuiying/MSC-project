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
# library(nhanesR)
library(cowplot)
library(xtable)
library(flextable)
library(officer)

setwd(paste0(PATH,'/Result'))

load('dfimp.RData')

var0=names(df)[c(7:ncol(df))]
var0


#--------------------------------------------------------------------------
df$status=df$dementia
table(df$status)

#---------------------------------------------
set.seed(123)
sub=sample(nrow(df),0.8*nrow(df))

dfx=df[sub,]
dft=df[-sub,]

export(dfx,'train.xlsx')
export(dft,'test.xlsx')



dfx$set<-1
dft$set<-2
df<-rbind(dfx,dft)
df$setg<-factor(df$set,levels = c(1,2),labels = c('Training','Validation'))


#--------------------------------------------
df$Status=factor(df$status)

FML<-as.formula(paste0("status~",paste0(var0,collapse = "+")))
FML2<-as.formula(paste0("Status~",paste0(var0,collapse = "+")))

table(df$setg)
#--------------------------------------------------------------------
dfx=subset(df,df$setg=='Training')
dft=subset(df,df$setg=='Validation')


mt<-paste0(var0,"=NA",collapse = ",")
mt
#--------------------------------------------------------
FML<-paste0("setg~",paste0(c(var0),collapse = "+"))
FML
descrTable(FML,df,method =1, 
           p.corrected = F,digits = 2,sd.type = 2,show.all = T)

tb<-descrTable(FML,df,method =1, 
               p.corrected = F,digits = 2,sd.type = 2,show.all = T)

export2xls(tb,"Description.xlsx")




options(scipen = 200)
dfx<-subset(df,df$set==1)
dft<-subset(df,df$set==2)
library(plyr)#----------------
library(questionr)
library(forestmodel)
#-------------------------
uni_glm<- function(x){
    FML<-as.formula(paste0("status~",x))
    glm1<-glm(FML,data = dfx,family = binomial(link = "logit"))
    X<-forest_model(glm1,factor_separate_line = T,exclude_infinite_cis = F,limits=c(0,10))
    table<-X$data
    table1<-table[,c(2,4,8:13)]
    table1$OR<-exp(table1$estimate)
    table1$CI5<-exp(table1$conf.low)
    table1$CI95<-exp(table1$conf.high)
    table1
    tb<-table1[,c(1:6,9:11)]
    tb<-as.data.frame(tb)
    for (i in c(3:5,7:9)) {tb[,i]<-sprintf('%0.3f',tb[,i]) }
    tb$p<-sprintf('%0.3f',tb$p.value)
    tb$p<-ifelse(tb$p=='0.000','<0.001',tb$p)
    tb$`OR(95%CI)`<-paste0(tb$OR," (",tb$CI5,', ',tb$CI95,")")
    tb$`OR(95%CI)`<-ifelse(tb$`OR(95%CI)`=="NA (NA, NA)","",tb$`OR(95%CI)`)
    tb$`OR(95%CI)`<-ifelse(tb$`OR(95%CI)`=="1.000 (NA, NA)","reference",tb$`OR(95%CI)`)
    tb$estimate[tb$estimate==0]<-""
    tbm<-tb[,c(1:5,11,10,6)]
    L<-length(tbm$variable)
    cha<-data.frame("character"=rep(x,L))
    result<-cbind(cha,tbm)
    return(result)}
var0
variable<-var0
variable

#-------------
result_glm_uni<-lapply(variable,uni_glm)

#---------------
result_glm_uni<-ldply(result_glm_uni,data.frame)
result_glm_uni
names(result_glm_uni)[c(4,5,6,7)]<-c("β",'SE','z',"OR(95%CI)")
res_uni_log<-result_glm_uni[,c(2:8)]
res_uni_log
for (i in 2:7) {res_uni_log[,i]<-ifelse(is.na(res_uni_log[,i]),'',res_uni_log[,i])}
for (i in c(2:7)) {res_uni_log[,i]<-ifelse(res_uni_log[,i]=='NA','',res_uni_log[,i])}  #----
res_uni_log

res_uni_log<-simputation::impute_proxy(res_uni_log,variable~level)
res_uni_log<-res_uni_log[-2]
res_uni_log
export(res_uni_log,"One-factor-logistic-regression.xlsx")





#----------------------------------------
var<-result_glm_uni$character[result_glm_uni$p.value<=0.1]

var
var<-na.omit(var)
var
var<-unique(var)
var


dff= df %>% dplyr::select(NACCID,PACKET,NACCDAYS,NACCIDEM,
                          NACCETPR,dementia,set,setg,
                          var
                          )


export(dff,'dff.xlsx')
save(dff,var,file = 'dff2.RData')
