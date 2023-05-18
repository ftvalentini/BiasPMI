
library(cbn)

data(cbn_gender_name_stats)

write.csv(cbn_gender_name_stats, "data/external/cbn_gender_name_stats.csv"
          , row.names=FALSE)
