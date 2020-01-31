import os, time
import click
import sfs, inference
import pandas as pd

import dask
from dask.distributed import Client, LocalCluster
import dask.array as da
#from dask_jobqueue import LSFCluster

@click.command()
@click.option("--cluster-type", type=click.Choice(["processes", "threads", "distributed", "distributed_threads"]), default="distributed")
@click.option("--n-trials", type=int, default=10)
def main(cluster_type, n_trials):
    start_time = time.perf_counter()
    click.echo(f"starting at {time.asctime()}")
    if cluster_type == "processes":
        click.echo("starting dask with local process scheduler")
        dask.config.set(scheduler="processes")
    elif cluster_type == "threads":
        click.echo("starting dask with local thread scheduler")
        dask.config.set(scheduler="threads")
    elif cluster_type == "distributed":
        click.echo("starting dask with distributed process scheduler")
        client = Client()
    elif cluster_type == "distributed_threads":
        click.echo("starting dask with distributed thread scheduler")
        cluster = LocalCluster(processes=False)
        client = Client(cluster)
    
    sim_index = sfs.sim_gene_index[sfs.sim_gene_index.get_locs(pd.IndexSlice[:,:,:,1:n_trials])]
    sim_sfs = sfs.load_simulated_sfs_stack(os.path.join("../genedose/minerva_sim_grid/russian_doll_grid", 
                                                        inference.simulation_templates["tennessen"]),
                                                        sim_index)
    sumstats = sfs.compute_summary_stats_da(sim_sfs, da.from_array(sim_index.get_level_values("logL")))
    n_train = n_trials // 2
    kdes = sfs.build_kdes_dask(sumstats, sim_index, n_train)
    observed_locs = sim_index.get_locs(pd.IndexSlice[:,:,:,n_train+1:])
    sfs.calculate_kdes_loglikelihood_dask(kdes, sumstats[:,observed_locs], sim_index[observed_locs])
    end_time = time.perf_counter()
    click.echo(f"finished at {time.asctime()}")
    click.echo(f"total computation time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
