from __future__ import annotations
import itertools
from functools import reduce
import operator
from typing import TextIO, Union, Optional

import attr

import numpy as np
import pandas as pd
import dask.array as da
import dask
from scipy import special, stats

from tqdm import tqdm
from joblib import Parallel, delayed


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


sim_gene_index = pd.MultiIndex.from_product([[0.0, 0.1, 0.3, 0.5], ["-1.0", "-2.0", "-3.0", "-4.0", "NEUTRAL"], np.arange(2.0, 5.1, 0.1).round(1), range(1, 10000+1)], names=["h", "s", "logL", "seed"])
sim_gene_index = sim_gene_index.delete(sim_gene_index.get_locs([(0.0, 0.1, 0.3), "NEUTRAL"]))

ref_gene_index = pd.MultiIndex.from_product([[0.0, 0.1, 0.3, 0.5], ["-1.0", "-2.0", "-3.0", "-4.0", "NEUTRAL"]], names=["h", "s"])
ref_gene_index = ref_gene_index.delete(ref_gene_index.get_locs([(0.0, 0.1, 0.3), "NEUTRAL"]))

CHUNK_SIZE = 100

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


def load_exac_sfs(exac_filename: str, cpg_filename: str, sample_size: int = 68858) -> pd.DataFrame:
    """
    Read tab-separated files containing information on ExAC variants, filter by coverage and CpG status,
    and return a DataFrame where columns are SFSs (indexed by DAC in NFE), and column keys are (gene, func).
    """
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

    func_sfsa = {}
    for output_func, input_types in func_types_exac.items():
        sfs_df = exac_df[exac_consequences.eval(" or ".join(input_types))].groupby("gene").DAC.value_counts()
        sfs_df = sfs_df.unstack(level="gene", fill_value=0)
        sfs_df = sfs_df.reindex(pd.RangeIndex(1,sample_size), fill_value=0)
        sfs_df = sfs_df.transform(compress_sfs)
        func_sfsa[output_func] = sfs_df

    return pd.concat(func_sfsa, axis=1, names=["func", "gene"]).swaplevel(axis="columns").sort_index(axis="columns")



def load_exac_log_lengths(func_types_filename: str, pph_filename: str, mu: float = 1e-8) -> pd.Series:
    """
    Reads CSVs containing target sizes and returns a Series containing gene lengths, indexed by (gene, func)
    """
    func_types_table = pd.read_csv(func_types_filename, index_col=None).rename({"type" : "func"}, axis="columns")
    pph_table = pd.read_csv(pph_filename, index_col=None).rename({"prediction" : "func"}, axis="columns")
    combined_table = func_types_table.append(pph_table)
    combined_table = combined_table[combined_table.CpG == 0].drop(["count", "CpG"], axis="columns")
    rates_by_input_func = combined_table.pivot("gene", "func").droplevel(0, axis="columns")
    rates_df = pd.DataFrame.from_dict({output_func : rates_by_input_func[input_funcs].sum(axis="columns")
                                   for output_func, input_funcs in func_types_pph.items()})
    rates_df.rename_axis("func", axis='columns')
    return np.log10(rates_df.stack().squeeze() / mu)


def load_simulated_sfs(filename: str, sample_size: int = 68858, compress=True) -> pd.Series:
    """
    Reads the sparse tab-delimited SFS format output by SimDose and returns a single simulated SFS as a Series,
    indexed by DAC.
    """
    data = pd.read_csv(filename, sep="\t", header=0, names=["DAC", "alleles"], index_col="DAC", squeeze=True) # load data
    data = data.reindex(pd.RangeIndex(1, sample_size), fill_value=0) # drop the 0 DAC row and fill in zero counts
    if compress:
        data = compress_sfs(data)
    return data


def load_simulated_sfs_np(filename: str, sample_size: int=68858) -> np.array:
    sfs = pd.read_csv(filename, sep="\t", index_col=0, dtype=int, squeeze=True)
    sfs = sfs.reindex(pd.RangeIndex(1, sample_size), fill_value=0)
    return sfs.values


def load_simulated_sfs_stack(template: str, index: pd.Index, chunks: int = CHUNK_SIZE, sample_size: int = 68858) -> da.array:
    delayed_arrays = []
    for keys in index:
        keys_dict = dict(zip(index.names, keys))
        filename = template.format(**keys_dict)
        delayed_arrays.append(da.from_delayed(dask.delayed(load_simulated_sfs_np)(filename, sample_size), shape=(sample_size-1,), dtype=int))
    return da.stack(delayed_arrays).rechunk((chunks, None))
    

def load_simulated_sfs_indices(template: str, index: pd.Index, sample_size: int = 68858) -> pd.DataFrame:
    sim_dict = {}
    for keys in index:
        keys_dict = dict(zip(index.names, keys))
        filename = template.format(**keys_dict)
        sim_dict[keys] = load_simulated_sfs(filename, sample_size, compress=False)
    return pd.concat(dict(zip(sim_dict.keys(), sim_dict.values())), names=["h", "s", "logL", "seed"], axis=1)


def load_simulated_sfs_chunk(template: str, index: pd.Index, sample_size: int = 68858) -> da.array:
    pandas_chunk = load_simulated_sfs_indices(template, index, sample_size)
    return da.from_array(pandas_chunk.values.T)


def load_simulated_sfs_chunks(template: str, index: pd.Index, sample_size: int=68858) -> da.array:
    delayed_chunks = []
    for i in range(len(index) // CHUNK_SIZE):
        chunked_index = index[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
        delayed_array = da.from_delayed(dask.delayed(load_simulated_sfs_chunk)(template, chunked_index, sample_size), dtype=int, shape=(CHUNK_SIZE, sample_size-1))
        delayed_chunks.append(delayed_array)
    return da.concatenate(delayed_chunks)


def compress_sfs(data: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(data):
        # find the smallest dtype to represent the data
        # we only want to do this for integers
        for dtype in np.uint8, np.uint16, np.uint32, np.uint64:
            if data.lt(np.iinfo(dtype).max).all():
                break
    else:
        dtype = data.dtype.type
    data = data.astype(pd.SparseDtype(dtype, fill_value=dtype(0)))  # compress representation in memory
    return data


def load_simulated_sfs_genes(template: str, trials: int = 1000, sample_size: int = 68858) -> pd.DataFrame:
    """
    Iterates over all combinations of h, s, logL, and seed (up to the specified value) and returns a DataFrame
    whose columns are SFSs. Columns are indexed by h, s, logL, and seed; rows are indexed by DAC.
    """
    sim_dict = {}
    for h, s in itertools.chain([(0.5, "NEUTRAL")], itertools.product([0.0, 0.1, 0.3, 0.5], ["-1.0", "-2.0", "-3.0", "-4.0"])):
        for logL in np.arange(2.0, 5.1, 0.1).round(1):
            for seed in range(1, trials+1):
                filename = template.format(h=h, s=s, logL=logL, seed=seed)
                sim_dict[h, s, logL, seed] = delayed(load_simulated_sfs)(filename, sample_size)
    return pd.concat(dict(zip(sim_dict.keys(), Parallel()(tqdm(sim_dict.values())))), names=["h", "s", "logL", "seed"], axis=1)


def load_simulated_sfs_reference(template: str, reference_L: float = 1e9, sample_size: int = 68858) -> pd.DataFrame:
    """
    Iterates over all combinations of h and s and returns a DataFrame whose columns are SFSs, scaled to L = 1.
    Columns are indexed by h and s; rows are indexed by DAC.
    """
    sim_dict = {}
    for h, s in itertools.chain([(0.5, "NEUTRAL")], itertools.product([0.0, 0.1, 0.3, 0.5], ["-1.0", "-2.0", "-3.0", "-4.0"])):
        filename = template.format(h=h, s=s)
        sim_dict[h,s] = load_simulated_sfs(filename, sample_size) / reference_L
    return pd.concat(sim_dict, names=["h", "s"], axis=1)


def calculate_prf_loglikelihood(reference_sfs_df: pd.DataFrame, observed_sfs_df: pd.DataFrame, observed_logL: pd.Series) -> pd.DataFrame:
    """
    Takes a reference_sfs DataFrame keyed by h and s, an observed_sfs DataFrame keyed by any set of keys,
    and an observed_logL Series keyed by the same key as observed_sfs.
    Returns a DataFrame whose rows are the input keys and columns are (h,s).
    """
    common_observed_keys = observed_sfs_df.keys() & observed_logL.keys() # get the list of keys common to observed SFS and logL
    # convert input data to dask arrays
    observed_logL_da = da.from_array(observed_logL[common_observed_keys].values, chunks=100) # n_keys
    observed_sfs_da = da.from_array(observed_sfs_df[common_observed_keys].values, chunks=(None, 100)) # sample_size x n_keys
    ref_sfs_da = da.from_array(reference_sfs_df.values) # sample_size x 17
    
    # scale reference sfs by observed length
    scaled_ref_da = ref_sfs_da.T[...,np.newaxis] * 10**observed_logL_da # 17 x sample_size x n_keys
    # calculate per-bin log likelihood
    poisson_loglikelihood = observed_sfs_da * da.log(scaled_ref_da) - scaled_ref_da - special.gammaln(scaled_ref_da+1) # 17 x sample_size x n_keys
    # aggregate log likelihood 
    combined_loglikelihood = da.nansum(poisson_loglikelihood, axis=1).T # n_keys x 17
    
    return pd.DataFrame(combined_loglikelihood.compute(), columns = reference_sfs_df.keys(), index = common_observed_keys)



def calculate_prf_loglikelihood_da(ref_sfs_da: da.array, observed_sfs_da: da.array, observed_logL_da: da.array) -> da.array: 
    scaled_ref_da = ref_sfs_da[...,np.newaxis] * 10**observed_logL_da # 17 x sample_size x n_keys
    # calculate per-bin log likelihood
    poisson_loglikelihood = observed_sfs_da.T * da.log(scaled_ref_da) - scaled_ref_da - special.gammaln(scaled_ref_da+1) # 17 x sample_size x n_keys
    # aggregate log likelihood 
    combined_loglikelihood = da.nansum(poisson_loglikelihood, axis=1).T # n_keys x 17
    
    return combined_loglikelihood


def compute_summary_stats_da(sfs: da.array, logL: da.array) -> da.array:
    sample_size = sfs.shape[1] + 1
    x = da.linspace(1/sample_size, 1, sample_size-1, chunks=CHUNK_SIZE)
    xbar = sfs @ x
    logstat = da.log1p(sfs).sum(axis=1)
    return da.stack([xbar, logstat, logL])

def compute_summary_stats(sfs_df: pd.DataFrame, sample_size: int = 68858) -> pd.DataFrame:
    xbar = sfs_df.agg(lambda x: x @ x.index / sample_size)
    logstat = sfs_df.transform(np.log1p).sum()
    return pd.DataFrame.from_dict({'xbar' : xbar, 'logstat' : logstat})


def initialize_kdes(sumstats: pd.DataFrame, training_set_size: int = 5000) -> pd.Series:
    training_set = sumstats.loc(axis=0)[:,:,:,:training_set_size]
    return training_set.groupby(level=["h","s"]).apply(lambda x: stats.gaussian_kde(x.values.T)).squeeze()


def calculate_kde_loglikelihood(kde_series: pd.Series, observed_sumstats: pd.DataFrame, observed_logL: pd.Series) -> pd.DataFrame:
    observed_sumstats["logL"] = observed_logL
    return observed_sumstats.apply(lambda x: kde_series.apply(lambda kde: np.log(kde(x.values.T).item())), axis=1)


def build_kdes_dask(sumstats: da.array, index: pd.Index, train_size: int):
    kdes = {}
    for index_keys in ref_gene_index:
        train_loc = index.get_locs(index_keys + (slice(None), slice(1,train_size)))
        train_data = sumstats[:,da.from_array(train_loc, chunks=CHUNK_SIZE)]
        kdes[index_keys] = dask.delayed(stats.gaussian_kde)(train_data)
    return kdes

def calculate_kdes_loglikelihood_dask(kdes, observed_sumstats: da.array, observed_index: pd.Index):
    loglikelihoods = {}
    for key, kde in kdes.items():
        loglikelihoods[key] = dask.delayed(kde)(observed_sumstats)
    loglikelihoods = dask.compute(loglikelihoods)
    output_df = pd.DataFrame.from_dict(loglikelihoods)
    output_df = output_df.apply(operator.methodcaller("explode"))
    output_df = output_df.set_index(observed_index)
    output_df.columns = pd.MultiIndex.from_tuples(output_df.columns, names=["h","s"])
    return output_df
    
