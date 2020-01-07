from __future__ import annotations
import os
import click
from functools import partial

import attr
import numpy as np
import pandas as pd
from attr.validators import in_

import sfs

from typing import Union, Callable, TextIO

simulation_templates = { 'tennessen' : "SFS_output_v2.6.1/tennessen/sampleSFS_size_68858/SFS_tennessen_european_samplesize_68858_S_{s}_h_{h}_mu_-8.0_L_{logL:0.1f}_seed_{seed}.tsv",
                         'supertennessen' : "SFS_output_v2.6.1/supertennessen/sampleSFS_size_68858/SFS_supertennessen_european_samplesize_68858_S_{s}_h_{h}_mu_-8.0_L_{logL:0.1f}_growth_0.03_seed_{seed}.tsv",
                         'subtennessen' : "SFS_output_v2.6.1/supertennessen/sampleSFS_size_68858/SFS_supertennessen_european_samplesize_68858_S_{s}_h_{h}_mu_-8.0_L_{logL:0.1f}_growth_0.005_seed_{seed}.tsv" }

ref_templates = { 'tennessen' : "SFS_tennessen_european_samplesize_68858_S_{s}_h_{h}_mu_-8.0_L_6.0_seed_sum_1_1000.tsv",
                  'supertennessen' : "SFS_supertennessen_european_samplesize_68858_S_{s}_h_{h}_mu_-8.0_L_6.0_growth_0.03_seed_sum_1_1000.tsv",
                  'subtennessen' : "SFS_supertennessen_european_samplesize_68858_S_{s}_h_{h}_mu_-8.0_L_6.0_growth_0.005_seed_sum_1_1000.tsv" }

exac_template = "ExAC_63K_variants.chr{chr}.tsv.gz"
cpg_template = "CpG/chr{chr}.cpg.tsv"
rates_template = "chr{chr}.rates.csv"
pph_template = "chr{chr}.pph.rates.csv"


@attr.s(auto_attribs=True, kw_only=True)
class FileSFSLoader:
    likelihood: str = attr.ib(default="PRF", validator=in_(["PRF", "KDE"]))
    sim_template: str = simulation_templates["tennessen"]
    ref_template: str = ref_templates["tennessen"]
    ref_L: float = 1e9
    mu: float = 1e-8
    num_trials: int = 10000
    training_size: int = 5000
    exac_template: str = exac_template
    cpg_template: str = cpg_template
    rates_template: str = rates_template
    pph_template: str = pph_template

    def _sfs_or_sumstats(self, the_sfs: pd.Series) -> pd.Series:
        if self.likelihood == "KDE":
            return sfs.compute_summary_stats(the_sfs)
        else:
            return the_sfs

    def load_simulated_genes(self, force_reload: bool = False) -> pd.Series:
        if force_reload or not hasattr(self, "_simulated_genes"):
            gene_sfs = sfs.load_simulated_sfs_genes(self.sim_template, self.num_trials)
            self._simulated_genes = self._sfs_or_sumstats(gene_sfs)
        return self._simulated_genes

    def load_simulated_test_genes(self) -> pd.Series:
        return self.load_simulated_genes()\
                   .reset_index("seed").query(f"seed >= {self.training_size}").set_index("seed", append=True)

    def load_simulated_reference(self) -> pd.Series:
        return sfs.load_simulated_sfs_reference(self.ref_template) / self.ref_L

    def load_empirical_gene_log_lengths(self, chrom: Union[int, str]) -> pd.Series:
        return np.log10(sfs.load_exac_log_lengths(self.rates_template.format(chr=chrom), self.pph_template.format(chr=chrom)) / self.mu)

    def load_empirical_genes(self, chrom: Union[int, str]) -> pd.Series:
        gene_sfs = sfs.load_exac_sfs(self.exac_template.format(chr=chrom), self.cpg_template.format(chr=chrom))
        return self._sfs_or_sumstats(gene_sfs)


    @classmethod
    def from_demography_and_base_dirs(cls, ref_demography: str = "tennessen",
                                      sim_demography: str = "tennessen",
                                      sims_dir: str = "", ref_dir: str = "",
                                      exac_dir: str = "", cpg_dir: str = "", rates_dir: str = "", **kwargs):
        return cls(sim_template=os.path.join(sims_dir, simulation_templates[sim_demography]),
                   ref_template=os.path.join(ref_dir, ref_templates[ref_demography]),
                   exac_template=os.path.join(exac_dir, exac_template),
                   cpg_template=os.path.join(cpg_dir, cpg_template),
                   rates_template=os.path.join(rates_dir, rates_template),
                   pph_template=os.path.join(rates_dir, pph_template),
                   **kwargs)


def get_loglikelihood_function(loader: FileSFSLoader) -> Callable[[pd.Series, pd.Series], pd.DataFrame]:
    if loader.likelihood == "KDE":
        simulated_sumstats = loader.load_simulated_genes()
        kdes = sfs.initialize_kdes(simulated_sumstats, loader.training_size)
        return partial(sfs.calculate_kde_loglikelihood, kdes)
    elif loader.likelihood == "PRF":
        ref_sfs = loader.load_simulated_reference()
        return partial(sfs.calculate_prf_loglikelihood, ref_sfs)
    else:
        raise ValueError(f"Unrecognized likelihood function {loader.likelihood}")


@click.command()
@click.option("--likelihood", type=click.Choice(["PRF", "KDE"], case_sensitive=False), default="PRF")
@click.option("--ref-demography", type=click.Choice(["tennessen", "supertennessen", "subtennessen"], case_sensitive=False), default="tennessen")
@click.option("--sim-demography", type=click.Choice(["tennessen", "supertennessen", "subtennessen"], case_sensitive=False), default="tennessen")
@click.option("--sim-dir", type=click.Path(exists=True, dir_okay=True, file_okay=False), default="")
@click.option("--ref-dir", type=click.Path(exists=True, dir_okay=True, file_okay=False), default="")
@click.option("--exac-dir", type=click.Path(exists=True, dir_okay=True, file_okay=False), default="")
@click.option("--cpg-dir", type=click.Path(exists=True, dir_okay=True, file_okay=False), default="")
@click.option("--rates-dir", type=click.Path(exists=True, dir_okay=True, file_okay=False), default="")
@click.option("--num-simulations", type=int, default=10000)
@click.option("--training-size", type=int, default=5000)
@click.option("--simulations/--empirical", default=True)
@click.option("--chromosome", default=None)
@click.option("--fout", type=click.File('w'), default="-")
def run_inference(likelihood: str, ref_demography: str, sim_demography: str, sim_dir: str, ref_dir: str, exac_dir: str,
                  cpg_dir: str, rates_dir: str,
                  num_simulations: int, training_size: int, simulations: bool, chromosome: Union[str, int], fout: TextIO):
    breakpoint()
    loader = FileSFSLoader.from_demography_and_base_dirs(ref_demography, sim_demography, sim_dir, ref_dir, exac_dir, cpg_dir, rates_dir,
                                                         num_trials=num_simulations, training_size=training_size, likelihood=likelihood)
    loglikelihood_function = get_loglikelihood_function(loader)

    if simulations:
        observed_data = loader.load_simulated_test_genes()
        observed_logL = observed_data.index.to_frame().logL
    else:
        if chromosome is None:
            observed_data = pd.concat([loader.load_empirical_genes(chrom) for chrom in range(1,23)])
            observed_logL = pd.concat([loader.load_empirical_gene_log_lengths(chrom) for chrom in range(1,23)])
        else:
            observed_data = loader.load_empirical_genes(chromosome)
            observed_logL = loader.load_empirical_gene_log_lengths(chromosome)

    loglikelihoods = loglikelihood_function(observed_data, observed_logL)
    ml = loglikelihoods.argmax(axis="columns")
    lrt_score = 2 * (loglikelihoods.max(axis="columns") - loglikelihoods[0.5].max(axis="columns"))
    pd.DataFrame({'ml' : ml, 'lrt' : lrt_score }).to_csv(fout, sep="\t")

if __name__ == "__main__":
    run_inference()
