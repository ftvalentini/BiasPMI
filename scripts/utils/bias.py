import numpy as np
import pandas as pd

from scipy import stats
import statsmodels.api as sm


def words2indices(words_target, words_attr_a, words_attr_b, str2idx):
    """
    Converts words to indices and checks words out of vocab
    Param:
        - 3 lists of words (target + 2 attributes)
    """
    # handling attr words out of vocab
    words_attr = words_attr_a + words_attr_b
    attr_outofvocab = [w for w in words_attr if w not in str2idx]
    attr_invocab = set(words_attr) - set(attr_outofvocab)
    assert len(attr_invocab) > 0, "ALL ATTRIBUTE WORDS ARE OUT OF VOCAB"
    if attr_outofvocab:
        print(f'\n{", ".join(attr_outofvocab)} \nATTRIBUTE WORDS NOT IN VOCAB')
    # target words out of vocab
    target_outofvocab = [w for w in words_target if w not in str2idx]
    if target_outofvocab:
        print(f'\n{", ".join(target_outofvocab)} \nTARGET WORDS NOT IN VOCAB')
    # words indices
    idx_t = [str2idx[w] for w in words_target if w not in target_outofvocab]
    idx_a = [str2idx[w] for w in words_attr_a if w not in attr_outofvocab]
    idx_b = [str2idx[w] for w in words_attr_b if w not in attr_outofvocab]
    return idx_t, idx_a, idx_b


def pmi(counts_wc, counts_w, count_c, count_tot):
    """
    PMI for given word counts of words W and C. It works vectorized accross W
    if needed.
    Param:
        - counts_wc: co-ocurrence array between C and W
        - counts_w: co-ocurrence array for W words
        - count_c: co-ocurrence count C
        - count_tot: total co-occurrence count

    See Levy et al (2015) https://aclanthology.org/Q15-1016.pdf section 2.1
    """
    numerador = counts_wc * count_tot
    denominador = counts_w * count_c
    res = np.log(numerador / denominador)
    return res


def log_oddsratio(
    count_a1, count_a2, count_b1, count_b2, ci_level=None, two_tailed=True,
):
    """
    Return log Odds Ratio between variable A (levels 1,2) and B (levels 1,2)
    Args:
        ci_level: return (log_oddsratio, std, pvalue, lower, upper)

    NOTE: see https://github.com/statsmodels/statsmodels/blob/b08825acb1ae57ecd40a675fe69574b2da3ef340/statsmodels/stats/contingency_tables.py#L729
    """
    table = np.array([[count_a1, count_b1], [count_a2, count_b2]])
    table = table
    t22 = sm.stats.Table2x2(table, shift_zeros=False)
    # NOTE shift_zeros adds 0.5 to all cels only if any cell equals 0
    lor = t22.log_oddsratio
    if ci_level:
        se = t22.log_oddsratio_se # standard deviation
        lower, upper = t22.log_oddsratio_confint(alpha=1-ci_level, method="normal")
        zscore = (lor - 0) / se
        if two_tailed:
            pvalue = t22.log_oddsratio_pvalue() # twotailed pvalue
        else:
            pvalue = 1 - stats.norm.cdf(zscore) # one tailed pvalue
        return lor, se, pvalue, lower, upper
    return lor


def dpmi_byword(C, words_target, words_attr_a, words_attr_b, str2idx, smoothing=0.):
    """
    Return DataFrame with DPMI A/B for each word in words_target and the
    relevant coocurrence counts
    Params:
        - C: co-occurrence sparse matrix
        - words_target, words_attr_a, words_attr_b: lists of words
        - str2idx: vocab dict
        - smoothing: add this value to ALL co-occurrences
    """
    print("Getting words indices...")
    idx_t, idx_a, idx_b = words2indices(
        words_target, words_attr_a, words_attr_b, str2idx)
    
    print("Computing counts...")
    vocab_size = len(str2idx)
    C_size = vocab_size ** 2
    total_count = C.sum() + smoothing * C_size # total
    count_a = C[idx_a,:].sum() + smoothing * len(idx_a) * vocab_size # total attr A
    count_b = C[idx_b,:].sum() + smoothing * len(idx_b) * vocab_size # total attr B
    counts_target = C.sum(axis=0)[:,idx_t] + smoothing * vocab_size # totales of each target
    counts_target_a = C[idx_a,:][:,idx_t].sum(axis=0) + smoothing * len(idx_a) # of each target with attr A
    counts_target_b = C[idx_b,:][:,idx_t].sum(axis=0) + smoothing * len(idx_b) # of each target with attr B

    print("Computing PMIs/OddsRatios...")
    pmi_a = pmi(counts_target_a, counts_target, count_a, total_count)
    pmi_b = pmi(counts_target_b, counts_target, count_b, total_count)
    
    str2idx_target = {w: str2idx[w] for w in words_target}
    df = pd.DataFrame(str2idx_target.items(), columns=['word','idx'])
    df['pmi_a'] = pmi_a.T
    df['pmi_b'] = pmi_b.T
    counts_nottarget_a = count_a - counts_target_a # de A sin cada target
    counts_nottarget_b = count_b - counts_target_b # de B sin cada target
    df['cooc_nottarget_a'] = counts_nottarget_a.T
    df['cooc_nottarget_b'] = counts_nottarget_b.T
    df['cooc_target_a'] = counts_target_a.T
    df['cooc_target_b'] = counts_target_b.T
    
    ci_level = .95
    def log_oddsratio_(a, b, c, d,):
        res = pd.Series(log_oddsratio(a, b, c, d, ci_level=ci_level, two_tailed=True))
        return res
    df_odds = df.apply(
        lambda d: log_oddsratio_(
            d['cooc_target_a'], d['cooc_nottarget_a'], d['cooc_target_b'],
            d['cooc_nottarget_b']), axis=1)
    df_odds.columns = ['lor','lor_se','lor_pvalue','lor_lower','lor_upper']

    # final DF
    df_final = pd.concat([df, df_odds], axis=1)
    df_final['dpmi'] = df['pmi_a'] - df['pmi_b'] # DPMI approx. w/LOR
    z = -stats.norm.ppf((1-ci_level) / 2)
    df_final['dpmi_lower'] = df_final['dpmi'] - z * df_final['lor_se']
    df_final['dpmi_upper'] = df_final['dpmi'] + z * df_final['lor_se']
    df_final = df_final.drop([
        'cooc_nottarget_a', 'cooc_nottarget_b', 'lor_lower', 'lor_upper', ],
        axis=1)
    return df_final


def cosine_similarities(M, idx_target, idx_attr, use_norm=True):
    """
    Cosine similarity between target words and attribute words. Returns array of
    shape (len(idx_target), len(idx_attr)) with similarity in each cell.
    Param:
        - M: d x V+1 matrix where column indices are words idx from str2idx
        - idx_context: indices of context word
        - idx_target: indices of target words
        - use_norm: divides by norm (as usual cosine); if False: dot product
    Notes:
        - It works OK if len(idx_*) == 1
    """
    M_t = M[:,idx_target] # matriz de target words
    M_a = M[:,idx_attr] # matriz de attr words
    productos = M_t.T @ M_a # rows: target words // cols: dot with each attr
    res = productos
    if use_norm:
        normas_t = np.linalg.norm(M_t, axis=0)
        normas_a = np.linalg.norm(M_a, axis=0)
        denominadores = np.outer(normas_t, normas_a)
        res = res / denominadores
    return res


def wefat_scores(
    M, idx_target, idx_attr_a, idx_attr_b, use_norm=True, return_components=False
):
    """
    WEFAT between target words and attributes A and B. Returns array of shape
    len(idx_target) with score for each context word.
    Param:
        - M: d x V+1 matrix where column indices are words idx from str2idx
        - idx_target: indices of target word
        - idx_attr_*: indices of attribute words
        - use_norm: uses cosine -- if False uses dot product
        - return_components: return all similarities with A and B
    Notes:
        - It works OK if len(idx_*) == 1
    """
    similarities_a = cosine_similarities(
        M, idx_target, idx_attr_a, use_norm=use_norm)
    similarities_b = cosine_similarities(
        M, idx_target, idx_attr_b, use_norm=use_norm)
    similarities_all = np.concatenate([similarities_a, similarities_b], axis=1)
    mean_similarities_a = np.mean(similarities_a, axis=1) # avg accross target A
    mean_similarities_b = np.mean(similarities_b, axis=1) # avg accross target B
    std_similarities_all = np.std(similarities_all, axis=1)
    res = (mean_similarities_a - mean_similarities_b) / std_similarities_all
    if return_components:
        return res, similarities_a, similarities_b
    return res


def bias_we_byword(
    M, words_target, words_attr_a, words_attr_b, str2idx, str2count,
):
    """
    Return DataFrame with WEFAT score A/B for each word in words_target, and the
    freq of each word
    Params:
        - M: word vectors matrix d x V+1
        - words_target, words_attr_a, words_attr_b: lists of words
        - str2idx, str2count: vocab dicts
    """
    # words indices
    idx_t, idx_a, idx_b = words2indices(
        words_target, words_attr_a, words_attr_b, str2idx)
    # bias metrics
    bias_scores, similarities_a, similarities_b = wefat_scores(
        M, idx_t, idx_a, idx_b, use_norm=True, return_components=True)
    # similarites as string joined by '|'
    def to_joined_string(x):
        lista = x.round(8).astype(str)
        res = "|".join(lista)
        return res
    similarities_a = pd.DataFrame(similarities_a).apply(to_joined_string, axis=1)
    similarities_b = pd.DataFrame(similarities_b).apply(to_joined_string, axis=1)
    # results DataFrame (todos los resultados sorted by idx)
    str2idx_target = {w: str2idx[w] for w in words_target}
    str2count_target = {w: str2count[w] for w in str2idx_target}
    df = pd.DataFrame(str2idx_target.items(), columns=['word','idx'])
    df['freq'] = str2count_target.values()
    df['wefat_score'] = bias_scores
    df['sims_a'] = similarities_a
    df['sims_b'] = similarities_b
    return df
