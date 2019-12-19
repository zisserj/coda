
import hmm_analysis
import sys

sample_name = sys.argv[1]

sample = hmm_analysis.CrossoverDetector()
sample.preprocess(sample_name, interval = 20)

sample.genome_hmm_detection(seed=4, smooth_window_size=20)

sample.plot_hmm(comparison = True, to_fname=sample_name.replace('.bedgraph', '.png'))
res = sample.try_match(cutoff=0.02)
res.to_csv(sample_name + ".crossovers", sep='\t', index=False)
