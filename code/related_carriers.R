#set working directory
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)
db1b_string = 'DB1B_MARKETS_%d_Q%d.csv'
related_carriers_dict_string =  "/related_carriers_dict_%d.csv"
related_carriers_string = "related_carriers_%d.csv"
#load libraries
library(sqldf)
library(data.table)
#define function for row selection
`%notin%` <- function(x,y) !(x %in% y) 
ptm <- proc.time()
# set regionals
regionals<-c('MQ','EV','OH','OO','RP','YV','9E','XE','QX','RU')
for (y in seq(2007,2016)){
  print(sprintf("finding related carriers %d",y))
  dls<-list()
  for (q in c(1,2,3,4)) {
      
     
    
      
      #load DB1B markets for year/quarter
      mkn<-read.csv(sprintf(db1b_string, y,q))
      
      #data correction by Tian
      if (y==2010 && q==1){
        mkn[mkn$TICKET_CARRIER=='NW','TICKET_CARRIER']<-'DL'
      }
      #calculate total passengers for each carrier, sort by this total
      carriers<-sqldf('select TICKET_CARRIER, sum(PASSENGERS) T_PASSENGERS from mkn group by TICKET_CARRIER order by T_PASSENGERS desc')
      #View(carriers)
      
      #========================================================
      #find the regional airline match (with at least 10K passengers)
      #========================================================
      #get one stop routes
      mkn2<-mkn[mkn$MARKET_COUPONS==2,]
      #get total passengers in system
      s = sum(carriers$T_PASSENGERS)
      carriers$perc = carriers$T_PASSENGERS/s
      #major carriers carry at least 5% of passengers
      majors <- subset(carriers, perc > .05)$TICKET_CARRIER
      #for each major, get passengers flown by partners, sum to previous quarter totals if they exist
      for (major in majors) {
        dl<-mkn2[mkn2$TICKET_CARRIER==major,]
        
        dl$PARTNER<-substr(dl$OP_CARRIER_GROUP,4,5)
        dl<-sqldf('select PARTNER, sum(PASSENGERS) T_PASSENGERS from dl group by PARTNER order by T_PASSENGERS desc')
        
        if ((length(dls[[major]]) == 0L) | is.null(dls[[major]]))
          {dls[[major]]=dl}
        else {
         
          new_dl = merge(dls[[major]],dl,by.x="PARTNER",by.y="PARTNER",all=TRUE)
          new_dl[is.null(new_dl)]=0
          new_dl[is.na(new_dl)]=0
          new_dl$T_PASSENGERS =  new_dl$T_PASSENGERS.x + new_dl$T_PASSENGERS.y
          dls[[major]]=new_dl[,c('PARTNER','T_PASSENGERS')]
        }
      }
      #remove larger tables from memory
      ##rm(mkn)
      ##rm(mkn2)
  }  
        
      # filter dataset for related carriers
      mr<-list()
      for (major in majors) {
       
        dl<-dls[[major]]
        major_pass = carriers[carriers$TICKET_CARRIER==major,'T_PASSENGERS'][[1]]
        partners <-subset(dl, (T_PASSENGERS>= 100000 | T_PASSENGERS > .10*major_pass) & (((PARTNER %notin% majors) & (PARTNER %in% regionals)) | (major=='WN' & PARTNER=='FL')))$PARTNER
        mr[[major]] <- partners
        
      }
      
      
      
      # write related carriers in dictonary format
      fn<-sprintf(related_carriers_dict_string,y)
      for (major in majors) {
        write(paste(c(major, " : ", paste(mr[[major]], sep=" ", collapse=" ")), sep ='', collapse = ''), append=TRUE, file=fn)
      }
      
      
      related_carriers = c()
      # build a bidirectional table of related carriers from list above
      for (major in majors) {
        for (partner in mr[[major]]) {
        related_carriers = rbind(related_carriers,c(major, partner))
        related_carriers = rbind(related_carriers,c(partner, major))
        }
      }
      # remove bidirectional partners if mapped to by more than one mainline
      for (partner in regionals) {
        if (partner %in% related_carriers[, 2] && table(related_carriers[,2])[[partner]]>=2) {
          related_carriers =  related_carriers[!(related_carriers[,1]==partner),]
        }
      }
      #format and save table for the year
      related_carriers=setnames(data.frame(related_carriers), old = c(1,2), new = c('PRIMARY_CARRIER','SECONDARY_CARRIER'))
      write.table(related_carriers,sprintf(related_carriers_string,y),sep=",")
}