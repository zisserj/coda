import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Splits the genome into equal sized bins to be used with bedtools.')
parser.add_argument('reference', type=str, help='Reference genome sizes file.')
parser.add_argument('binSize', type=int, help='Size of each bin.')
parser.add_argument('binStep', type=int, default=0, nargs='?',
    help='Step size between bins. Defaults to bin size.')
args = parser.parse_args()
if args.binStep == 0:
    args.binStep = args.binSize

genome_sizes = pd.read_csv(args.reference, sep='\t', index_col=0, names=['chrm', 'size'])
chrm_bins_df = []

print(f'Creating bins for "{args.reference}" with binSize={args.binSize} and binStep={args.binStep}.')

# iterate over chromosomes
for chrm, row in genome_sizes.iterrows():
    start = pd.np.arange(1, row['size'], args.binStep)
    end = start + args.binSize
    name = pd.np.full(len(start), chrm)
    # shape positions as dataframe
    data = {'chr': name, 'start': start, 'end': end}
    df = pd.DataFrame(data)
    chrm_bins_df.append(df)

bin_table = pd.concat(chrm_bins_df)

args.reference = args.reference.split('.')[0]
if args.binStep == args.binSize:
    name_out = f'{args.reference}.bins-{args.binSize}.txt'
else:
    name_out = f'{args.reference}.bins-{args.binSize}-{args.binStep}.txt'
try:
    bin_table.to_csv(name_out, sep='\t', index=False)
    print(f'Done. Bin table written to "{name_out}".')
except Exception as ex:
    print(f'Failed to write results to "{name_out}". Please check the directory exists and has suitable permissions.\n{ex}')