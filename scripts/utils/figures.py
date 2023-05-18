
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr, bootstrap, permutation_test
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.multitest import multipletests
from statsmodels.regression.linear_model import WLS
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm


def str_to_floats(s):
    """Convert strings separated by '|' to list
    """
    res = [float(i) for i in s.split("|")]
    return res


def join_bias_dfs(df_bias_pmi, df_bias_glovewc, df_bias_sgns):
    """Join bias DataFrames
    """
    df_bias = pd.merge(
        df_bias_pmi, df_bias_glovewc, how="inner", on=["corpus", "bias", "word", "idx", "word"])
    df_bias = pd.merge(
        df_bias, df_bias_sgns, how="inner", on=["corpus", "bias", "word", "idx", "word", "freq"], 
        suffixes=['_glovewc', '_sgns'])
    return df_bias


def join_with_target_dfs(
    df_bias, df_names, df_occupations_cbn, df_occupations_gsjz, df_glasgow,
    print_data_availability=False):
    """
    Join bias DFrame with external DFrames
    """
    # Names DataFrame
    df_names_bias = pd.merge(
        df_names, df_bias, how="inner", left_on=["name"], right_on=["word"])
    # Occupations CBN DataFrame 
    df_occupations_cbn_bias = pd.merge(
        df_occupations_cbn, df_bias, how="inner", left_on=["Occupation"], right_on=["word"])
    # Occupations Garg DataFrame 
    df_occupations_gsjz_bias = pd.merge(
        df_occupations_gsjz, df_bias, how="inner", left_on=["Occupation"], right_on=["word"])
    # Glasgow DataFrame 
    df_glasgow_bias = pd.merge(
        df_glasgow, df_bias, how="inner", left_on=["word"], right_on=["word"])
    if print_data_availability:
        print(f"External data has {df_names.shape[0]} names -- Vocab has {df_names_bias.shape[0]} names")
        print(f"External data has {df_occupations_cbn.shape[0]} CBN occs. -- Vocab has {df_occupations_cbn_bias.shape[0]} occs.")
        print(f"External data has {df_occupations_gsjz.shape[0]} GSJZ occs. -- Vocab has {df_occupations_gsjz_bias.shape[0]} occs.")
        print(f"External data has {df_glasgow.shape[0]} Glasgow words -- Vocab has {df_glasgow_bias.shape[0]} Glasgow words")
    # concat final DataFrame
    df_out = pd.concat(
        {"names": df_names_bias, "occupations_cbn": df_occupations_cbn_bias, 
        "occupations_gsjz": df_occupations_gsjz_bias, "glasgow": df_glasgow_bias}, 
        names=["experiment"]).reset_index(level=0)
    return df_out


def pearson_cor(data, x_var, y_var, weight_se_var=None):
    """Pearson correlation weighted or unweighted 
    """
    if weight_se_var:
        weights = 1 / (data[weight_se_var] ** 2)
        arr = data[[x_var, y_var]]
        corr = DescrStatsW(arr, weights=weights).corrcoef[0,1]
    else:
        corr, pv = pearsonr(data[x_var], data[y_var])
    return corr


def scatter_color_plt(
    x_var, y_var, data, xlabel, ylabel, color_var=None,
    weight_se_var=None, add_lm=True, error_cols=None,
    print_pearson=True, title=None, point_size=40, edgecolor='black',):
    """WEFAT scatter plots

    NOTE: statsmodels expects 1 / variance --> we pass SE as
    weight_se_var and run 1/v**2 inside
    see https://github.com/statsmodels/statsmodels/blob/a5a6c44674d76c49944ca76d8ff8cd747e91be3a/statsmodels/regression/linear_model.py#L712
    """
    fig, ax = plt.subplots(figsize=(6,4))
    ax.axhline(y=0, linestyle='--', linewidth=0.5, color='black')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    if error_cols:
        lower_err = data[y_var] - data[error_cols[0]] 
        upper_err = data[error_cols[1]] - data[y_var] 
        y_err = [lower_err, upper_err]
        scatter = ax.scatter(
            x=data[x_var], y=data[y_var],
            s=point_size, marker='o', edgecolor=edgecolor, linewidth=0.5)
        errs = ax.errorbar(
            x=data[x_var], y=data[y_var], c=color_var, yerr=y_err,
            linewidth=1, ls='none')
    else:
        scatter = ax.scatter(
            x_var, y_var, c=color_var, data=data,
            cmap='RdYlBu', s=point_size, marker='o', edgecolor=edgecolor, linewidth=0.5)
    weights = 1.0
    if weight_se_var:
        weights = 1 / (data[weight_se_var] ** 2)
    if add_lm:
        y = data[y_var]
        X = data[x_var]
        X = sm.add_constant(X)
        wls_model = WLS(y, X, weights=weights)
        results = wls_model.fit()
        b0, b1 = results.params
        rsq = results.rsquared
        t_test_b1 = results.t_test([0,1])
        pvalue_b1 = t_test_b1.pvalue
        x = np.unique(data[x_var])
        y_pred = b0 + b1 * x
        ax.plot(x, y_pred, linestyle='--', linewidth=1, color='black')
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    if title:
        plt.title(title, fontsize=13)
    if color_var:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.ax.set_title(color_var)
    if print_pearson:
        corr = pearson_cor(data, x_var, y_var, weight_se_var=weight_se_var)
        out = f"\n\nPearson corr. {corr:.2f}"
        print(out)
    return fig, ax


def bias_score(
    a: list, b: list, suma: float=None, std: float=None, n_a: int=None
) -> float:
    """WEFAT bias score

    WARNING supplying suma and std and n_a is valid only for equal length groups!!!
    """
    if std is not None and suma is not None and n_a is not None:
        diff = (2 * np.sum(a) - suma) / n_a
        return diff / std
    else:
        std = np.std(a + b)
        diff = np.mean(a) - np.mean(b)
        return diff / std


def permutations_pvalue(a: list, b: list, n_resamples=np.inf) -> float:
    """Permutation pvalue for WEFAT scores
    """
    data = (a, b)
    suma = np.sum(a + b)
    std = np.std(a + b)
    n_a = len(a)
    bias_score_ = lambda x,y: bias_score(x, y, suma=suma, std=std, n_a=n_a)
    res = permutation_test(
        data, bias_score_, n_resamples=n_resamples, alternative='two-sided', 
        batch=1000)
    # batch != None para que no rompa memoria
    # NOTE np.inf hace aprox combn(16,8)=12870 corridas 
    # pval corrected:
    # (see https://www.sciencedirect.com/science/article/pii/S0002929707604911?via%3Dihub)
    null_values = res.null_distribution
    statistic = res.statistic
    num = (abs(null_values) > abs(statistic)).sum() + 1
    denom = len(null_values) + 1
    pvalue = num / denom
    return pvalue


def add_pvalue(df, **kwargs):
    """Add WEFAT bootstrap SE and permutation pvalues to bias DFrame
    """
    n_resamples_permut = kwargs.get('n_resamples_permut', np.inf)
    we_names = ["sgns", "glovewc"]
    for n in we_names:
        permutations_results = []
        for i, (a, b) in tqdm(df[['sims_a_'+n, 'sims_b_'+n]].iterrows(), desc=n, total=len(df)):
            pvalue = permutations_pvalue(a, b, n_resamples=n_resamples_permut)
            permutations_results += [pvalue]
        df[n + '_pvalue'] = permutations_results
    return df


def correct_pvalues(df):
    """Correct pvalues with Benjamini-Hochberg
    """
    pval_cols = [c for c in df.columns if 'pvalue' in c]
    for col in pval_cols:
        df[col + 'cor'] = multipletests(df[col], method='fdr_bh')[1]
    return df


def scatter_plt(
    x_var, y_var, data, xlabel=None, ylabel=None, title=None, point_size=40, 
    edgecolor='black',
):
    """Scatter plot for pvalues
    """
    fig, ax = plt.subplots(figsize=(8,4))
    ax.axhline(y=0, linestyle='--', linewidth=0.5, color='black')
    ax.axvline(x=0, linestyle='--', linewidth=0.5, color='black')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    scatter = ax.scatter(
        x_var, y_var, data=data,
        s=point_size, marker='o', edgecolor=edgecolor, linewidth=0.5)
    plt.ylim(-0.05, 1.05)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if title:
        plt.title(title, fontsize=12)
    return fig, ax
