

library(tidyverse)
library(data.table)
library(boot)
library(jsonlite)


datetime_now = function() format(Sys.time(), "%Y-%m-%d %H:%M:%OS")


bias_score = function(data, indices) {
    d = data[indices, ]
    n = nrow(d)
    means = d[, mean(value), by = .(group)]
    mean_a = means[group == "a", ][[2]]
    mean_b = means[group == "b", ][[2]]
    # use n as denom for sd instead of n-1
    std = sd(d[["value"]]) * sqrt((n - 1) / n)
    score = (mean_a - mean_b) / std
    return(score)
}


bootstrap_statistics = function(df) {
    boot_out = boot(df, R=2000, statistic=bias_score, strata=df$group,
        parallel="multicore", ncpus=3)
    se = sd(boot_out$t)
    ci = boot.ci(boot_out, conf=0.95, type="perc")$percent[4:5]
    res = data.frame(se, lower=ci[1], upper=ci[2])
    return(res)
}

cat(datetime_now(), "Reading input data frame\n")
df = data.table::fread("results/figures_data.csv")

cat(datetime_now(), "String vectors to numeric\n")
similarity_cols = names(df) %>% str_subset("sims_")
df[, similarity_cols] = df %>%
    select(all_of(similarity_cols)) %>%
    mutate_all(function(x) map(x, jsonlite::fromJSON))

cat(datetime_now(), "To long\n")
df_long = df %>%
    select(word, experiment, corpus, all_of(similarity_cols)) %>%
    # slice(1:4) %>%
    unnest(cols = all_of(similarity_cols)) %>%
    pivot_longer(all_of(similarity_cols), names_to = "tmp") %>%
    mutate(
        we = tmp %>% str_extract("(glovewc|sgns)$"),
        group = tmp %>% str_sub(6, 6) %>% as.factor()
    ) %>%
    as.data.table()


cat(datetime_now(), "Running bootstraps\n")

grpn = df_long %>% distinct(we, corpus, experiment, word) %>% nrow()
options(digits.secs=3)

set.seed(10010)
df_res = df_long[
    , {
        cat(datetime_now(), " -- ", .GRP, "/", grpn, "\n", sep="");
        bootstrap_statistics(.SD)
    }
    , by = .(we, corpus, experiment, word)]

cat(datetime_now(), "Transforming dataframe\n")
df_res_wide = df_res %>%
    pivot_wider(names_from = we, values_from = c(se, lower, upper))
df_out = df %>% left_join(df_res_wide, by = c("word", "experiment", "corpus"))

cat(datetime_now(), "Writing output data frame\n")
write_csv(df_out, "results/figures_data.csv")
