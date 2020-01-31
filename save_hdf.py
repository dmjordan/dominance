import os
import click
import sfs, inference
import pandas as pd
import h5py
from contextlib import closing
from tqdm import tqdm


@click.command()
@click.option("--root-dir", "-I", type=click.Path(file_okay=False, dir_okay=True, exists=True), default=".")
@click.option("--demography", "-D", type=click.Choice(["tennessen", "supertennessen", "subtennessen"]), default="tennessen")
@click.option("--num-seeds", "-n", type=int)
@click.argument("outfile", type=h5py.File)
@click.option("--sample-size", "-N", type=int, default=68858)
@click.option("--dataset-name", "-d", type=str, default="sim_genes")
def load_save_sim_genes(root_dir, demography, num_seeds, outfile, sample_size, dataset_name):
    sim_template = os.path.join(root_dir, inference.simulation_templates[demography])
    sim_index = sfs.sim_gene_index[sfs.sim_gene_index.get_locs(pd.IndexSlice[:,:,:,1:num_seeds])]
    with closing(outfile):
        sim_dataset = outfile.require_dataset(dataset_name, (len(sim_index), sample_size-1), 'uint32', compression="gzip", shuffle=True, scaleoffset=0)
        for i, row in enumerate(tqdm(sim_index)):
            row_dict = dict(zip(sim_index.names, row))
            row_sfs = sfs.load_simulated_sfs(sim_template.format(**row_dict), sample_size, compress=False)
            sim_dataset[i] = row_sfs.values
         

if __name__ == "__main__":
    load_save_sim_genes()
