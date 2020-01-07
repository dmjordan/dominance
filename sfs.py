from __future__ import annotations
import itertools
from typing import TextIO, Union

import attr

import numpy as np
import pandas as pd
from scipy import special, stats


func_types_exac = {"LOF" : ['stop_gained', 'splice_acceptor_variant', 'splice_donor_variant'],
                "nonsynon" : ['stop_gained', 'splice_acceptor_variant', 'splice_donor_variant', 'missense_variant'],
                "missense" : ['missense_variant'],
                "synon" : ['synonymous_variant'],
                "benign" : ['benign'],
                "possiblydamaging" : ['possiblydamaging'],
                "probablydamaging" : ["probablydamaging"],
                "damaging" : ["probablydamaging", "possiblydamaging"],
                "LOF_damaging" : ["probablydamaging", "possiblydamaging", 'stop_gained', 'splice_acceptor_variant', 'splice_donor_variant'],
                "LOF_probably" : ["probablydamaging", 'stop_gained', 'splice_acceptor_variant', 'splice_donor_variant']
                   }

func_types_pph = { "LOF" : ["nonsense", "splice"],
                   "nonsynon" : ["nonsense", "splice", "missense"],
                   "missense" : ["missense"],
                   "synon" : ["coding-synon"],
                   "benign" : ["benign"],
                   "possiblydamaging" : ["possibly_damaging"],
                   "probablydamaging" : ["probably_damaging"],
                   "damaging" : ["probably_damaging", "possibly_damaging"],
                   "LOF_damaging" : ["nonsense", "splice", "probably_damaging", "possibly_damaging"],
                   "LOF_probably" : ["nonsense", "splice", "probably_damaging"] }


@attr.s(auto_attribs=True, frozen=True)
class SFSUnroller:
    sample_size: int
    dtype: np.dtype = attr.ib(default=np.uint32, converter=np.dtype) #type: ignore

    @classmethod
    def unroll(cls, sfs_series: pd.Series, sample_size: int, dtype: np.dtype = np.uint32) -> np.ndarray:
        return cls(sample_size, dtype)(sfs_series)

    def __call__(self, sfs_series: pd.Series) -> np.ndarray:
        sfs = np.zeros(self.sample_size, self.dtype)
        sfs.put(sfs_series.index.get_level_values("DAC") - 1, sfs_series)
        return sfs


def load_exac_sfs(exac_filename: str, cpg_filename: str, sample_size: int = 68858) -> pd.Series:
    nt_dtype = pd.api.types.CategoricalDtype(["A", "C", "T", "G"])

    cpg_df = pd.read_csv(cpg_filename, delim_whitespace=True, header=None,
                         names=["chr", "pos", "ref", "alt", "cpg"],
                         dtype={'chr': np.unicode, "ref": nt_dtype, "alt": nt_dtype})

    exac_df = pd.read_csv(exac_filename, delimiter="\t",
                          usecols=["chr", "pos", "ref", "alt", "pantro2", "gene_canonical", "type_canonical", "pphpred", "AC_NFE", "AN_NFE"],
                          dtype={'chr': np.unicode, "ref": nt_dtype, "alt": nt_dtype, "pantro2": nt_dtype}) \
        .dropna(subset=["ref", "alt", "pantro2", "gene_canonical", "type_canonical"]) \
        .astype({"AC_NFE": int}).query("AC_NFE > 0 and AC_NFE < AN_NFE and AN_NFE > 0.8 * @sample_size") \
        .merge(cpg_df, how="left", on=["chr", "pos", "ref", "alt"]).fillna({"cpg": 0}).query("cpg == 0")
    exac_df["DAC"] = exac_df.AC_NFE.where(exac_df.alt != exac_df.pantro2, exac_df.eval("AN_NFE - AC_NFE"))
    exac_df = exac_df.rename({"gene_canonical" : "gene"}, axis='columns')
    exac_consequences = exac_df.type_canonical.str.get_dummies("&").join(exac_df.pphpred.str.get_dummies()).astype(bool)

    unroll_sfs = SFSUnroller(sample_size)
    func_sfsa = []
    for output_func, input_types in func_types_exac.items():
        sfs_sparse = exac_df[exac_consequences.eval(" or ".join(input_types))].groupby("gene").DAC.value_counts()
        sfs_unrolled = sfs_sparse.groupby(level="gene").apply(unroll_sfs)
        sfs_df = pd.DataFrame.from_dict({'func' : output_func, 'alleles' : sfs_unrolled}).set_index("func", append=True)
        func_sfsa.append(sfs_df)

    return pd.concat(func_sfsa).sort_index().squeeze()


def load_exac_log_lengths(func_types_filename: str, pph_filename: str, mu: float = 1e-8) -> pd.Series:
    func_types_table = pd.read_csv(func_types_filename, index_col=None).rename({"type" : "func"}, axis="columns")
    pph_table = pd.read_csv(pph_filename, index_col=None).rename({"prediction" : "func"}, axis="columns")
    combined_table = func_types_table.append(pph_table)
    combined_table = combined_table[combined_table.CpG == 0].drop(["count", "CpG"], axis="columns")
    rates_by_input_func = combined_table.pivot("gene", "func").droplevel(0, axis="columns")
    rates_df = pd.DataFrame.from_dict({output_func : rates_by_input_func[input_funcs].sum(axis="columns")
                                   for output_func, input_funcs in func_types_pph.items()})
    rates_df.rename_axis("func", axis='columns')
    return np.log10(rates_df.stack().squeeze() / mu)


def load_simulated_sfs(filename: str) -> pd.Series:
    return pd.read_csv(filename, sep="\t", header=0, names=["DAC", "alleles"]).set_index("DAC").drop(0)


def load_simulated_sfs_genes(template: str, trials: int = 1000, sample_size: int = 68858) -> pd.Series:
    sim_dict = {}
    for h, s in itertools.chain([(0.5, "NEUTRAL")], itertools.product([0.0, 0.1, 0.3, 0.5], ["-1.0", "-2.0", "-3.0", "-4.0"])):
        for logL in np.arange(2.0, 5.1, 0.1).round(1):
            for seed in range(1, trials+1):
                filename = template.format(h=h, s=s, logL=logL, seed=seed)
                sim_dict[h, s, logL, seed] = load_simulated_sfs(filename)
    return pd.concat(sim_dict, names=["h", "s", "logL", "seed"]).groupby(level=["h","s","logL","seed"]).apply(SFSUnroller(sample_size)).squeeze()


def load_simulated_sfs_reference(template: str, reference_L: float = 1e9, sample_size: int = 68858) -> pd.Series:
    sim_dict = {}
    for h, s in itertools.chain([(0.5, "NEUTRAL")], itertools.product([0.0, 0.1, 0.3, 0.5], ["-1.0", "-2.0", "-3.0", "-4.0"])):
        filename = template.format(h=h, s=s)
        sim_dict[h,s] = load_simulated_sfs(filename)
    return pd.concat(sim_dict, names=["h", "s"]).groupby(level=["h","s"]).apply(SFSUnroller(sample_size)).squeeze() / reference_L


def calculate_prf_loglikelihood(reference_sfs_series: pd.Series, observed_sfs_series: pd.Series, observed_logL: pd.Series) -> pd.DataFrame:
    def single_key_loglikelihood(key):
        scaled_reference_sfs = reference_sfs_series * 10**(observed_logL.get(key, np.nan))
        observed_sfs = observed_sfs_series.loc[key]
        def single_reference_loglikelihood(reference_sfs):
            poisson_logpmfs = observed_sfs * np.log(reference_sfs) - reference_sfs - special.gammaln(observed_sfs + 1)
            return np.nansum(poisson_logpmfs)
        return scaled_reference_sfs.apply(single_reference_loglikelihood)
    keys = observed_sfs_series.index.to_series()
    return keys.apply(single_key_loglikelihood)


def compute_summary_stats(unrolled_sfs: pd.Series, sample_size: int = 68858) -> pd.DataFrame:
    xbar = unrolled_sfs.apply(lambda x: np.arange(1, sample_size + 1) @ x / sample_size)
    logstat = unrolled_sfs.apply(lambda x: np.log1p(x).sum())
    return pd.DataFrame.from_dict({'xbar' : xbar, 'logstat' : logstat})


def initialize_kdes(sumstats: pd.DataFrame, training_set_size: int = 5000) -> pd.Series:
    training_set = sumstats.loc(axis=0)[:,:,:,:training_set_size]
    return training_set.groupby(level=["h","s"]).apply(lambda x: stats.gaussian_kde(x.values.T)).squeeze()


def calculate_kde_loglikelihood(kde_series: pd.Series, observed_sumstats: pd.DataFrame, observed_logL: pd.Series) -> pd.DataFrame:
    observed_sumstats["logL"] = observed_logL
    return observed_sumstats.apply(lambda x: kde_series.apply(lambda kde: np.log(kde(x.values.T).item())), axis=1)


