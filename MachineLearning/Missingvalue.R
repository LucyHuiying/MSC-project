rm(list=ls())
PATH<-'G:/poject/New3'
setwd(PATH)
path<-paste0(PATH,'/Result/');DPI=600
library(rio)
library(dplyr)
library(compareGroups)
library(survival)
library(export)
library(ggplot2)
library(ggsci)
library(scales)
library(tidyverse)
library(Hmisc)
library(tableone)
library(naniar)
library(lubridate)
library(simputation)
library(ggprism)
library(survminer)
library(survival)
library(ggprism)
library(ggsci)
library(ggpubr)
library(ggplot2)
library(rms)
library(survey)
library(plyr)


load('dfclean.RData')

df=df_final

setwd(paste0(PATH,'/Result'))

gg_miss_var(df[,7:ncol(df)])
x=gg_miss_var(df[,7:ncol(df)])
TB=x$data

vard=TB$variable[TB$pct_miss>=30]
vard

df=df %>% dplyr::select(-c(vard))



descrTable(dementia~NACCAGE+SEX,df,digits = 2,show.all = T,include.miss = T,
           chisq.test.seed = 123,lab.missing = 'missing',sd.type = 2)

#descrTable(dementia~.,df[,c(6:ncol(df))],digits = 2,show.all = T, include.miss = T,chisq.test.seed = 123,lab.missing = 'missing',sd.type = 2)

tb=descrTable(dementia~.,df[,c(6:ncol(df))],digits = 2,show.all = T,include.miss = T,chisq.test.seed = 123,
              lab.missing = 'missing',sd.type = 2)
export2xls(tb,'1-Before missing value processing.xls')







df1=df[,c(7:ncol(df))]
#------------------------------------
library(mice)
library(lattice)
library(MASS)
library(nnet)
library(missRanger)
set.seed(123)
dfimp<-missRanger(df1,pmm.k = 3, num.trees = 100, splitrule = "extratrees",seed = 123)




sum(is.na(dfimp))
#-------------------------
dfx<-complete(dfimp)

df=cbind(df[,c(1:6)],dfx)


save(df,file = 'dfimp.RData')
export(df,'dfimp.xlsx')



tb=descrTable(dementia~.,df[,c(6:ncol(df))],digits = 2,show.all = T,include.miss = T,chisq.test.seed = 123,
              lab.missing = 'missing',sd.type = 2)
export2xls(tb,'1-Missing value processing.xls')



