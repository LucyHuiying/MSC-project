rm(list=ls())
setwd('G:/poject/New3')
#path<-paste0(PATH,'/结果/');DPI=600
library(rio)
library(dplyr)
# install.packages('compareGroups')
library(compareGroups)
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
library(plyr)

df=import('investigator_nacc64.csv')


table(df$PACKET)
dt0=subset(df,df$PACKET=='I'|df$PACKET=='IT')

table(dt0$NACCVNUM) 
dt02=subset(df,df$NACCVNUM==1) 



dt0=subset(dt0,dt0$VISITYR<=2021)

dt0$year2021=ifelse(dt0$VISITYR==2021&dt0$VISITMO<=12,1,0)

table(dt0$year2021)

dt0=subset(dt0,dt0$year2021==0)

dt0=subset(dt0,dt0$VISITYR>=2011)


table(dt0$NACCAVST) 
dt=subset(dt0,dt0$NACCAVST>=2)




table(dt$NACCIDEM) 
dt=subset(dt,dt$NACCIDEM!=8)
table(dt$NACCIDEM)



dff=subset(df,df$NACCVNUM==2)
dff$delet=ifelse(dff$NACCVNUM==2&dff$NACCFDYS>=24*30,1,0)

table(dff$delet)



dff=dff %>% dplyr::select(NACCID,NACCVNUM,delet)
table(dff$NACCVNUM)

dff=dff %>% dplyr::select(-NACCVNUM)
table(dff$delet)


dt=left_join(dt,dff,by='NACCID')
table(dt$delet)
dt=subset(dt,dt$delet!=1)
table(dt$NACCIDEM)




dfx=subset(df,df$NACCVNUM>=2&df$NACCIDEM==1)
dff=dfx %>% dplyr::select(NACCID,NACCFDYS,NACCIDEM)


dff=dff[!duplicated(dff$NACCID),]
dff= dff %>% arrange(NACCID,NACCFDYS)

dff$dem_24=ifelse(dff$NACCFDYS>24*30,1,0)
dff=dff %>% dplyr::select(c(NACCID,dem_24))
table(dff$dem_24)




dt=left_join(dt,dff,by='NACCID')
table(dt$dem_24)
summary(dt$NACCDAYS)


dt$dementia=ifelse(dt$NACCIDEM==1,1,0)
table(dt$dementia)





table(dt$NACCETPR) 



#---------------------------------------------------------------------------结局数据
dts= dt %>% dplyr::select(NACCID,
                          PACKET, 
                          NACCDAYS,
                          NACCIDEM, 
                          NACCETPR, 
                          dementia
)






dtt= dt %>% dplyr::select(NACCID,
                          NACCAGE, 
                          SEX, 
                          EDUC, 
                          NACCLIVS, 
                          INDEPEND, 
                          HISPANIC,
                          MARISTAT 
                          )


summary(dtt$NACCAGE)


table(dtt$SEX)
dtt$SEX=factor(dtt$SEX,levels = c(1,2),labels = c('Male','Female'))



table(dt$HISPANIC)
dtt$HISPANIC=ifelse(dtt$HISPANIC==9,NA,dtt$HISPANIC)
dtt$HISPANIC=factor(dtt$HISPANIC,levels = c(0,1),labels = c('No','Yes'))

summary(dtt$EDUC)
dtt$EDUC=ifelse(dtt$EDUC>36,NA,dtt$EDUC)

table(dtt$NACCLIVS)
dtt$NACCLIVS=ifelse(dtt$NACCLIVS==9,NA,dtt$NACCLIVS)
dtt$NACCLIVS=factor(dtt$NACCLIVS)

table(dtt$INDEPEND)
dtt$INDEPEND=ifelse(dtt$INDEPEND==9,NA,dtt$INDEPEND)
dtt$INDEPEND=factor(dtt$INDEPEND)


table(dtt$MARISTAT)
dtt$MARISTAT=ifelse(dtt$MARISTAT==9,NA,dtt$MARISTAT)
dtt$MARISTAT=factor(dtt$MARISTAT)

dt1=dtt




dtt= dt %>% dplyr::select(NACCID,

                        NACCMOM,
                         NACCDAD,
                         NACCFAM
)


for (i in 2:ncol(dtt)) { dtt[,i]=ifelse(dtt[,i]<0|dtt[,i]>1,NA,dtt[,i])}
for (i in 2:ncol(dtt)) { dtt[,i]=factor(dtt[,i],levels = c(0,1),labels = c('No','Yes'))}
dt2=dtt



dtt= dt %>% dplyr::select(NACCID,
                          NACCAMD, 

                          ANYMEDS,

                          NACCHTNC,
                          NACCACEI,
                          NACCAAAS,
                          NACCBETA,
                          NACCCCBS,
                          NACCDIUR,
                          NACCVASD,
                          NACCANGI,
                          NACCAHTN,
                          NACCLIPL,
                          NACCNSD,
                          NACCAC,
                          NACCADEP,
                          NACCAPSY,
                          NACCAANX,
                          NACCPDMD,
                          NACCEMD,
                          NACCEPMD,
                          NACCDBMD
                          )

descrTable(~.,dtt,method = 3)
summary(dtt$NACCAMD)
dtt$NACCAMD=ifelse(dtt$NACCAMD<0,NA,dtt$NACCAMD)


for (i in 3:ncol(dtt)) { dtt[,i]=ifelse(dtt[,i]<0|dtt[,i]>1,NA,dtt[,i])}
for (i in 3:ncol(dtt)) { dtt[,i]=factor(dtt[,i],levels = c(0,1),labels = c('No','Yes'))}
dt3=dtt




dtt= dt %>% dplyr::select(NACCID,
                          
                          CVHATT,
                          CVAFIB,
                          CVANGIO,
                          CVBYPASS,
                          CVPACE,
                          CVCHF,
                          CVOTHR,
                          CBSTROKE,
                          HXSTROKE,
                          CBTIA,
                          PD,
                          PDOTHR,
                          HXHYPER,
                          SEIZURES,
                          TRAUMBRF,
                          TRAUMEXT,
                          TRAUMCHR,
                          NCOTHR,
                          HYPERTEN,
                          HYPERCHO,
                          DIABETES,
                          B12DEF,
                          THYROID,
                          INCONTU,
                          INCONTF,
                          DEP2YRS,
                          DEPOTHR,
                          FOCLSYM,
                          FOCLSIGN
)

descrTable(~.,dtt,method = 3)


for (i in 2:ncol(dtt)) { dtt[,i]=ifelse(dtt[,i]<0|dtt[,i]>2,NA,dtt[,i])}
for (i in 2:ncol(dtt)) { dtt[,i]=factor(dtt[,i])}
dt4=dtt




dtt= dt %>% dplyr::select(NACCID,
                          
                          PACKSPER, 
                          #SMOKYRS,
                          #QUITSMOK,
                          ALCOHOL,
                          TOBAC30,
                          TOBAC100,

                          ABUSOTHR,
                          PSYCDIS,
                          NACCTBI
)
descrTable(~.,dtt,method = 3)


for (i in 3:ncol(dtt)) { dtt[,i]=ifelse(dtt[,i]<0|dtt[,i]>2,NA,dtt[,i])}
for (i in 3:ncol(dtt)) { dtt[,i]=factor(dtt[,i])}
dt5=dtt




dtt= dt %>% dplyr::select(NACCID,
                          NACCBMI,  
                          BPSYS,
                          BPDIAS,
                          HRATE,
                          VISION,
                          VISCORR,
                          VISWCORR,
                          HEARING,
                          HEARAID,
                          HEARWAID

)

summary(dtt$NACCBMI)
dtt$NACCBMI=ifelse(dtt$NACCBMI<=10|dtt$NACCBMI>100,NA,dtt$NACCBMI)

summary(dtt$BPSYS)
dtt$BPSYS=ifelse(dtt$BPSYS<=70|dtt$NACCBMI>=230,NA,dtt$BPSYS)

summary(dtt$BPDIAS)
dtt$BPDIAS=ifelse(dtt$BPDIAS<=30|dtt$BPDIAS>=140,NA,dtt$BPDIAS)

summary(dtt$HRATE)
dtt$HRATE=ifelse(dtt$HRATE<=33|dtt$HRATE>=160,NA,dtt$HRATE)

for (i in 6:ncol(dtt)) { dtt[,i]=ifelse(dtt[,i]<0|dtt[,i]>2,NA,dtt[,i])}
for (i in 6:ncol(dtt)) { dtt[,i]=factor(dtt[,i])}
dt6=dtt








dtt= dt %>% dplyr::select(NACCID,
                          
                          ABRUPT,
                          STEPWISE,
                          SOMATIC,
                          EMOT
 
)
descrTable(~.,dtt,method = 3)


for (i in 2:ncol(dtt)) { dtt[,i]=ifelse(dtt[,i]<0|dtt[,i]>2,NA,dtt[,i])}
for (i in 2:ncol(dtt)) { dtt[,i]=factor(dtt[,i])}
dt7=dtt








dtt= dt %>% dplyr::select(NACCID,
                          
                          HACHIN,
                          SPEECH,
                          FACEXP,
                          TRESTFAC,
                          TRESTRHD,
                          TRESTLHD,
                          TRESTRFT,
                          TRESTLFT,
                          TRACTRHD,
                          TRACTLHD,
                          RIGDNECK,
                          RIGDUPRT,
                          RIGDUPLF,
                          RIGDLORT,
                          RIGDLOLF,
                          TAPSRT,
                          TAPSLF,
                          HANDMOVR,
                          HANDMOVL,
                          HANDALTR,
                          HANDALTL,
                          LEGRT,
                          LEGLF,
                          ARISING,
                          POSTURE,
                          GAIT,
                          POSSTAB,
                          BRADYKIN,
                          PDNORMAL #URPDS
)

for (i in 2:ncol(dtt)) { dtt[,i]=ifelse(dtt[,i]<0,NA,dtt[,i])}
dt8=dtt






dtt= dt %>% dplyr::select(NACCID,

                          MEMORY,
                          ORIENT,
                          JUDGMENT,
                          COMMUN,
                          HOMEHOBB,
                          PERSCARE,
                          CDRSUM,
                          CDRGLOB
                         
)

#for (i in 2:ncol(dtt)) { dtt[,i]=ifelse(dtt[,i]<0,NA,dtt[,i])}
dt9=dtt


dtt= dt %>% dplyr::select(NACCID,
                          
                          DELSEV,
                          HALLSEV,
                          AGITSEV,
                          DEPDSEV,
                          ANXSEV,
                          ELATSEV,
                          APASEV,
                          DISNSEV,
                          IRRSEV,
                          MOTSEV,
                          NITESEV,
                          APPSEV
)
descrTable(~.,dtt,method = 3)

gg_miss_var(dtt)
for (i in 2:ncol(dtt)) { dtt[,i]=ifelse(dtt[,i]<0|dtt[,i]>=8,NA,dtt[,i])}

#dt10=dtt





dtt= dt %>% dplyr::select(NACCID,
                          DECAGE,
                          NACCGDS,
                          NACCCOGF,
                          TRAVEL,
                          BILLS,
                          TAXES,
                          SHOPPING,
                          GAMES,
                          STOVE,
                          MEALPREP,
                          EVENTS,
                          PAYATTN,
                          REMDATES,
                          NACCMOTF,
                          MOMODE,
                          COGMODE,
                          #NACCBEHF, 
                          BEMODE,
                          
                          NOGDS,
                          SATIS,
                          DROPACT,
                          EMPTY,
                          BORED,
                          SPIRITS,
                          AFRAID,
                          HAPPY,
                          HELPLESS,
                          STAYHOME,
                          MEMPROB,
                          WONDRFUL,
                          WRTHLESS,
                          ENERGY,
                          HOPELESS,
                          BETTER,
                          
                          
                          NACCNREX,
                          FOCLDEF,
                          GAITDIS,
                          EYEMOVE,
                          DECSUB,
                          DECIN,
                          DECCLIN,

                          COGMEM,
                          COGJUDG,
                          COGLANG,
                          COGVIS,
                          COGATTN,
                          COGOTHR,
                          BEAPATHY,
                          BEDEP,
                          BEVHALL,
                          BEAHALL,
                          BEDEL,
                          BEDISIN,
                          BEIRRIT,
                          BEAGIT,
                          BEPERCH,
                          BEOTHR,

                          MOGAIT,
                          MOFALLS,
                          MOTREM,
                          MOSLOW
                          
                          
)

descrTable(~.,dtt,method = 3)
summary(dtt$NACCGDS)
dtt$NACCGDS=ifelse(dtt$NACCGDS<0|dtt$NACCGDS>=15,NA,dtt$NACCGDS)

summary(dtt$DECAGE)
dtt$DECAGE=ifelse(dtt$DECAGE<15|dtt$DECAGE>=110,NA,dtt$DECAGE)

names(dtt)[19]
for (i in 5:19) { dtt[,i]=ifelse(dtt[,i]<0|dtt[,i]>=8,NA,dtt[,i])}
for (i in 5:19) { dtt[,i]=factor(dtt[,i])}

for (i in 20:ncol(dtt)) { dtt[,i]=ifelse(dtt[,i]<0|dtt[,i]>2,NA,dtt[,i])}
for (i in 20:ncol(dtt)) { dtt[,i]=factor(dtt[,i])}


dt10=dtt







dtt= dt %>% dplyr::select(NACCID,
                          
                          ANIMALS,
                          VEG,
                          TRAILA,
                          TRAILB,
                          NACCMMSE,
                          COURSE,
                          FRSTCHG,
                          MMSELOC,
                          MMSELAN,
                          MMSEORDA,
                          MMSEORLO,
                          NPSYCLOC,
                          NPSYLAN,
                          LOGIMEM,
                          DIGIF,
                          DIGIFLEN,
                          DIGIB,
                          DIGIBLEN,
                          
                          WAIS,
                          MEMUNITS,
                          MEMTIME,
                          BOSTON,
                          COGSTAT,
                          NACCC1
)

summary(dtt$ANIMALS)
dtt$ANIMALS=ifelse(dtt$ANIMALS<0|dtt$ANIMALS>77,NA,dtt$ANIMALS)
summary(dtt$ANIMALS)
dtt$ANIMALS=ifelse(dtt$ANIMALS<0|dtt$ANIMALS>77,NA,dtt$ANIMALS)

summary(dtt$NACCMMSE)
dtt$NACCMMSE=ifelse(dtt$NACCMMSE<0|dtt$NACCMMSE>30,NA,dtt$TRAILA)


summary(dtt$TRAILA)
dtt$TRAILA=ifelse(dtt$TRAILA<0|dtt$TRAILA>150,NA,dtt$TRAILA)

summary(dtt$TRAILB)
dtt$TRAILB=ifelse(dtt$TRAILB<0|dtt$TRAILB>300,NA,dtt$TRAILB)


descrTable(~.,dtt,method = 3)

for (i in 7:ncol(dtt)) { dtt[,i]=ifelse(dtt[,i]<0|dtt[,i]>5,NA,dtt[,i])}
for (i in 7:ncol(dtt)) { dtt[,i]=factor(dtt[,i])}

dt11=dtt


df_final=left_join(dts,dt1,by='NACCID')

df_final=left_join(df_final,dt2,by='NACCID')

df_final=left_join(df_final,dt3,by='NACCID')

df_final=left_join(df_final,dt4,by='NACCID')

df_final=left_join(df_final,dt5,by='NACCID')

df_final=left_join(df_final,dt6,by='NACCID')

df_final=left_join(df_final,dt7,by='NACCID')

df_final=left_join(df_final,dt8,by='NACCID')

df_final=left_join(df_final,dt9,by='NACCID')


df_final=left_join(df_final,dt10,by='NACCID')


df_final=left_join(df_final,dt11,by='NACCID')



save(df_final,file = 'dfclean.RData')
export(df_final,'dfclean.xlsx')



table(df_final$dementia)





