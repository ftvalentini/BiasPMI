
import logging

import pandas as pd
import numpy as np

from utils.figures import (
    join_bias_dfs, str_to_floats, add_pvalue, correct_pvalues
)


# (corpus, bias, file)
INPUT_FILES_INFO = {
    "files_pmi" : [
        ("wiki2014", "gender", "results/bias_pmi-wiki2014-FEMALE-MALE.csv"),
        ("subs", "gender", "results/bias_pmi-subs-FEMALE-MALE.csv"),
    ],
    "files_sgns" : [
        ("wiki2014", "gender", "results/bias_sgns-wiki2014-FEMALE-MALE.csv"),
        ("subs", "gender", "results/bias_sgns-subs-FEMALE-MALE.csv"),
    ],
    "files_glovewc" : [
        ("wiki2014", "gender", "results/bias_glovewc-wiki2014-FEMALE-MALE.csv"),
        ("subs", "gender", "results/bias_glovewc-subs-FEMALE-MALE.csv"),
    ]
}

WORDS_LISTS_FILES = {
    "names": "words_lists/NAMES.txt",
    "occupations_cbn": "words_lists/OCCUPATIONS_CBN.txt",
    "glasgow": "words_lists/GLASGOW.txt",
}

EXTERNAL_DATA_FILES = {
    "names": "data/external/cbn_gender_name_stats.csv", # Caliskan et al, 2017
    "occupations_cbn": "data/external/professionsBLS2015.tsv", # Caliskan et al, 2017
    "glasgow": "data/external/GlasgowNorms.csv", # Lewis and Lupyan, 2020
}

OUTPUT_FILE = "results/figures_data.csv"


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


def main():
    
    data_preparer = DataPreparer(
        input_files_info=INPUT_FILES_INFO,
        words_lists_files=WORDS_LISTS_FILES,
        external_data_files=EXTERNAL_DATA_FILES
    )
    
    logging.info("Reading bias files...")
    _ = data_preparer.read_bias_files()
    
    logging.info("Reading words lists...")
    _ = data_preparer.read_words_lists()
    
    logging.info("Reading external data...")
    _ = data_preparer.read_external_data()
    
    logging.info("Transforming external data...")
    _ = data_preparer.transform_external_data()
    
    logging.info("Joining data sources...")
    df_final = data_preparer.prepare_data()

    logging.info("Running permutations tests and bootstrap...")
    df_final = data_preparer.run_permutations(df_final)

    logging.info("Saving csv...")
    df_final.to_csv(OUTPUT_FILE, index=False)
        
    logging.info("DONE!")


class DataPreparer:

    def __init__(
        self, input_files_info: dict, words_lists_files: dict, 
        external_data_files: dict
    ) -> None:
        self.input_files_info = input_files_info
        self.words_lists_files = words_lists_files
        self.external_data_files = external_data_files

    def read_bias_files(self) -> None:
        files_pmi_info = self.input_files_info["files_pmi"]
        files_sgns_info = self.input_files_info["files_sgns"]
        files_glovewc_info = self.input_files_info["files_glovewc"]
        self.df_pmi = pd.concat(
            [pd.read_csv(f) for _, _, f in files_pmi_info], 
            keys=[(corpus, bias) for corpus, bias, _ in files_pmi_info],
            names=["corpus", "bias"]).reset_index(level=[0,1])
        self.df_sgns = pd.concat(
            [pd.read_csv(f) for _, _, f in files_sgns_info], 
            keys=[(corpus, bias) for corpus, bias, _ in files_sgns_info],
            names=["corpus", "bias"]).reset_index(level=[0,1])
        self.df_glovewc = pd.concat(
            [pd.read_csv(f) for _, _, f in files_glovewc_info], 
            keys=[(corpus, bias) for corpus, bias, _ in files_glovewc_info],
            names=["corpus", "bias"]).reset_index(level=[0,1])

    def read_words_lists(self) -> None:
        self.words_lists = {}
        for name, file in self.words_lists_files.items():
            self.words_lists[name] = [line.strip() for line in open(file,'r')]

    def read_external_data(self) -> None:
        self.df_names = pd.read_csv(
            self.external_data_files["names"], 
            usecols=["name", "proportion_female"])
        self.df_occupations_cbn = pd.read_csv(
            self.external_data_files["occupations_cbn"], sep='\t',
            usecols=["label1", "label2", "label3", "label4", "label5", "Women"])
        self.df_glasgow = pd.read_csv(
            self.external_data_files["glasgow"], header=[0, 1])

    def transform_external_data(self) -> None:
        
        ### Names DataFrame
        names = self.words_lists['names']
        self.df_names.drop_duplicates(inplace=True)
        self.df_names['name'] = self.df_names['name'].str.lower()
        self.df_names['proportion_female'] = self.df_names['proportion_female'] * 100
        # keep names from list
        self.df_names.query("name in @names", inplace=True)
        self.df_names.columns = ["word", "score"]
        self.df_names["experiment"] = "names-gender"
        print(
            f"List has {len(names)} names -- Data has {self.df_names.shape[0]} names")

        ### Occupations DataFrame CBN
        # We coalesce Will Lowe's labels
        occupations_cbn = self.words_lists['occupations_cbn']
        self.df_occupations_cbn['word'] = self.df_occupations_cbn['label1']\
            .combine_first(self.df_occupations_cbn['label2'])\
            .combine_first(self.df_occupations_cbn['label3'])\
            .combine_first(self.df_occupations_cbn['label4'])\
            .combine_first(self.df_occupations_cbn['label5'])
        self.df_occupations_cbn.drop(
            columns=['label1', 'label2', 'label3', 'label4', 'label5'], inplace=True)
        self.df_occupations_cbn.rename(
            columns={'Women': 'score'}, inplace=True)
        self.df_occupations_cbn.query(
            "word in @occupations_cbn", inplace=True)  # keep names from list
        self.df_occupations_cbn.drop_duplicates(
            'word', inplace=True)  # keep first if dup
        self.df_occupations_cbn["experiment"] = "occupations-gender"
        print(
            f"List has {len(occupations_cbn)} CBN occs. -- Data has {self.df_occupations_cbn.shape[0]} CBN occs.")

        ### Glasgow Norms DataFrame LL
        # see https://github.com/mllewis/IATLANG/blob/3b2b51d7e26c0554cb7c1cfce68390834089086a/writeup/journal/supporting_information/main_figures/F1/get_F1_data.R#L9
        # and https://github.com/mllewis/IATLANG/blob/3b2b51d7e26c0554cb7c1cfce68390834089086a/writeup/journal/sections/study1_writeup.Rmd#L83
        glasgow = self.words_lists['glasgow']
        self.df_glasgow.columns = self.df_glasgow.columns.to_flat_index().str.join("_")
        self.df_glasgow = self.df_glasgow[['Words_Unnamed: 0_level_1', 'GEND_M']].copy()
        self.df_glasgow.columns = ["word", "GEND_M"]
        self.df_glasgow["word"] = self.df_glasgow["word"].str.split(" ", expand=True)[
            0].str.lower()
        self.df_glasgow = self.df_glasgow.groupby(
            "word", as_index=False).agg(maleness_norm=("GEND_M", "mean"))
        self.df_glasgow["score"] = 8 - \
            self.df_glasgow["maleness_norm"]  # femaleness
        self.df_glasgow.drop_duplicates(inplace=True)
        # keep words from list
        self.df_glasgow.query("word in @glasgow", inplace=True)
        self.df_glasgow = self.df_glasgow[["word", "score"]]
        self.df_glasgow["experiment"] = "glasgow-gender"
        print(
            f"List has {len(glasgow)} names -- Data has {self.df_glasgow.shape[0]} names")

    def prepare_data(self) -> pd.DataFrame:

        # All external data in one DataFrame:
        df_experiments = pd.concat(
            [self.df_names, self.df_occupations_cbn, self.df_glasgow, ]) 
        # All biases in one DataFrame:
        df_bias = join_bias_dfs(self.df_pmi, self.df_glovewc, self.df_sgns)
        # All data in one DataFrame:
        df_final = pd.merge(df_experiments, df_bias, how="inner", on="word")
        # remove unused combinations
        query_ = """(experiment == 'names-gender' & bias == 'gender') | \
            (experiment == 'occupations-gender' & bias == 'gender') | \
            (experiment == 'glasgow-gender' & bias == 'gender')"""
        df_final = df_final.query(query_).reset_index(drop=True).copy()
        return df_final

    def run_permutations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add permutations pvalues and bootstrap SE and CI to df.
        Correct pvalues with Benjamini-Hochberg FDR correction.
        """
        # str to list of floats
        sims_cols = [c for c in df.columns if c.startswith("sims_")]
        for c in sims_cols:
            df[c] = df[c].apply(str_to_floats)
        # run tests
        df = add_pvalue(df, n_resamples_permut=np.inf)
        # pvalues FDR correction
        df = correct_pvalues(df)
        return df


if __name__ == "__main__":
    main()
