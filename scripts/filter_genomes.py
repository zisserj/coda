import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Creates an informative bin template based on coverage A and B as produced by bedtools.')
parser.add_argument('covA', type=str, help='Coverage bedgraph of control parent A.')
parser.add_argument('covB', type=str, help='Coverage bedgraph of control parent B.')
parser.add_argument('out', type=str, help='Filename to output the resulting template to.')
args = parser.parse_args()


print(f'Creating bin template using "{args.covA}" and "{args.covB}".')

genome_a = pd.read_csv(args.covA, sep='\t', names=['chrm', 'start', 'end', 'val'])
genome_b = pd.read_csv(args.covB, sep='\t', names=['chrm', 'start', 'end', 'val'])

genome_names = genome_a.chrm.str.replace(r'\d', '', regex=True).unique()
# Make sure the names are in the correct order
if genome_a[genome_a['chrm'].str.startswith(genome_names[0])].val.mean() < genome_a[~genome_a['chrm'].str.startswith(genome_names[0])].val.mean():
    genome_names = genome_names[::-1]

genome_reads = genome_a.merge(genome_b, on=['chrm','start','end'], suffixes=(genome_names[0], genome_names[1]))

length = genome_reads.iloc[0, 2] - genome_reads.iloc[0, 1]
step = genome_reads.iloc[1, 1] - genome_reads.iloc[0, 1]

for s in genome_names:
    source_bins = genome_reads.index[genome_reads.chrm.str.startswith(s)]
    informative_map = genome_reads.loc[source_bins, 'val' + s] != 0
    unique_map = genome_reads.loc[source_bins, 'val' + s] == genome_reads.iloc[source_bins, 3:].sum(axis=1)
    genome_reads.loc[source_bins, 'keep'] = informative_map & unique_map

keep = genome_reads.keep.sum() * 100 / genome_reads.keep.count()
print('{0:.2f}% bins kept past filtering.'.format(keep))

genome_template = genome_reads[genome_reads.keep].iloc[:, :3]

try:
    genome_template.to_csv(args.out, sep='\t', index=False, header=False)
    print(f'Done. Template written to "{args.out}".')
except Exception as ex:
    print(f'Failed to write results to "{args.out}". Please check the directory exists and has suitable permissions.\n{ex}')
