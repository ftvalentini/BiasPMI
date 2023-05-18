

library(dplyr)
library(weights)

file = "results/figures_data.csv"

df = read.csv(file)

corpora = df$corpus %>% unique()
experiments = df$experiment %>% unique()

bias_and_se = c(
  "wefat_score_sgnsXNULL",
  "wefat_score_sgnsXse_sgns",
  "wefat_score_ftXNULL",
  "wefat_score_ftXse_ft",
  "wefat_score_glovewcXNULL",
  "wefat_score_glovewcXse_glovewc",
  "dpmiXNULL",
  "dpmiXlor_se"
)

cat("p-values of weighted and unweighted correlations:\n\n")

pvalues = list()
for (c_ in corpora) {
  for (e_ in experiments) {
    df_tmp = df %>% filter(corpus == c_ & experiment == e_) 
    nombre = paste0(c_,"_", e_)
    results_ = c()
    for (bs_ in bias_and_se) {
      bias_and_se_ = strsplit(bs_, "X")[[1]]
      bias_ = bias_and_se_[1]
      se_ = bias_and_se_[2]
      if (se_ == "NULL") {
        weights_ = NULL
        } else {
        weights_ = df_tmp[[se_]]
      }
      nn = paste0(bias_,"_",se_)
      res_ = wtd.cor(df_tmp[["score"]], y=df_tmp[[bias_]], weight=weights_)
      pvalue_ = res_[4]
      results_[nn] = pvalue_
      cat(nombre, nn, "/ pvalue = ", pvalue_, "\n")
    }
    pvalues[[nombre]] = results_
  }
}

df_pvalues = bind_rows(pvalues, .id = "experiment") %>%
  filter(experiment != "wiki2021_mturk-affluence") %>% 
  mutate_if(is.numeric, round, 4)

