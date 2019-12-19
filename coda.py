import numpy as np
import pandas as pd
# import statsmodels.api as sm  # Used for now deprecated CrossoverDetector.smooth_noise2()
pd.options.mode.chained_assignment = None

from hmmlearn import hmm

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt
from matplotlib import cm   # Colormap func, use int in range(0, 240)
import matplotlib.ticker as ticker
plt.style.use('seaborn')
plt.rcParams["patch.force_edgecolor"] = True
plt.rcParams["figure.figsize"] = (10,5)

import pickle

class CrossoverDetector():

    def __init__(self, log_level=1):
        self._flags = {flag: False for flag in ['is_loaded', 'is_fit', 'refined', 'syri', 'cent_data']}
        self.log_level = log_level
        self._log(0, 'Detector initiated.')
    
    def __repr__(self):
        var_list = vars(self).copy()
        var_list['HMM'] = f'GaussianHMM(n_components = {self.HMM.n_components}, trained_samples({len(self.HMM.sample_names)})'
        var_list = [f'{str(name)} = {str(val)}' if type(val) not in [list, pd.DataFrame] else f'{str(name)}({len(val)})' for name, val in var_list.items()]
        rep = 'CrossoverDetector({})'.format(',\n'.join(sorted(var_list)))
        return rep

    def _log(self, level: int, message: str):
        level_names = {0:'DEBUG', 1:'INFO', 2:'WARNING', 3:'OFF'}
        if level < self.log_level:
            return
        else:
            ts = datetime.datetime.now().isoformat(sep=' ', timespec='milliseconds')
            msg = f'[{ts}, {level_names[level]}] - {message}'
            print(msg)

    def preprocess(self, bedgraph: str, method='arbitrary', interval=20, binsize=6000):
        self._log(0, 'Aggregating sample bedgraph={}, method={}, interval={}, binsize={}.'.format(\
                        bedgraph, method, interval, binsize))
        # Load coverage file
        sample = pd.read_csv(bedgraph, sep='\t', header=None, names=['chrm', 'start', 'end', 'val'])
        sample = sample[sample.val <= np.percentile(sample.val, 99)].reset_index(drop=True) # drop outliers
        self._raw_sample = sample

        # Groups sample according to supplied method
        sample_avg = sample.copy()
        if method == 'arbitrary':
            sample_avg['new_idx'] = sample.index // interval
            sample_avg = sample_avg.groupby(['chrm', 'new_idx']).agg({'start':'min',
                'end':'max', 'val':'mean'}).reset_index(0).reset_index(drop=True)
        elif method == 'robust':
            for chrm in np.unique(sample['chrm']):
                chrm_df = sample_avg[sample_avg['chrm'] == chrm].index
                bins = np.arange(1, sample_avg.loc[chrm_df, 'start'].max(), binsize)
                bin_num = np.digitize(sample_avg.loc[chrm_df, 'start'], bins)
                sample_avg.loc[chrm_df, 'bin_num'] = bin_num
            sample_avg = sample_avg.groupby(['chrm', 'bin_num']).agg({'start':'min',
                'end':'max', 'val':'mean'}).reset_index(0).reset_index(drop=True)
        sample_avg.val = sample_avg.val / sample_avg.val.max() # normalise to [0-1]
        self._sample = sample_avg

        # Transform data to a list of observations
        self.sequences = [self._sample.val[self._sample.chrm == chrm].values for chrm in pd.unique(self._sample.chrm)]
        
        # Save kwargs
        self._sample_name = bedgraph.split("/")[-1]
        self._group_bin_itv = interval
        self._avg_cov = self._raw_sample.val.mean().round(3)
        self._flags['is_loaded'] = True

        self._log(1, f"Successfully loaded '{self._sample_name}'. Mean coverage is {self._avg_cov}.")
        # print(self._sample.val.describe())

    def read_context(self, fname: str):
        self._log(0, f'Reading centromere boundries from file "{fname}".')
        df = pd.read_csv(fname, sep='\t', header=None, names=['chrm','start','end','context'])
        df.chrm = df.chrm.str.replace(r'\D', '', regex=True).astype(int)
        df = df[df.context == 'cent']
        df.set_index('chrm', inplace=True)
        self._context_table = df
        self._flags['cent_data'] = True

    def hmm_from_pickle(self, fname: str):
        self._log(0, f'Loading pickled HMM from file "{fname}".')
        try:
            with open(fname, 'rb+') as f:
                hm_model = pickle.load(f)
                self._flags['is_fit'] = True
                self.HMM = hm_model
        except:
            self._log(2, f'Failed to load HMM pickle from file "{fname}". Initializing new model.')
            self._hmm_init()

    def _hmm_init(self, n_components=3, **kwargs):
        self._log(0, f'Initializing new HM model. n_components = {n_components}, Passing args = {str(kwargs)}')
        if n_components < 2:
            self._log(2, f'Can not initiate model with fewer than 2 states. Quitting.')
            return
        hm_model = hmm.GaussianHMM(n_components=n_components, n_iter=1000, covariance_type='full', init_params='sc', **kwargs)            
        # HMM initial behaviour adjustments
        # expected_mean = obs_sequence[obs_sequence != 0].mean()
        # hm_model.means_ = np.array([[0], [expected_mean], [expected_mean*2]])
        if self._flags['is_loaded']:
            co_odds = 10 / len(self._sample) # Approx 10 COs 
            means = np.arange(0, 1.01, 1/(n_components-1)) * self._sample.val.mean() * 2
        else:
            self._log(2, 'Sample not loaded. Setting default parameters.')
            co_odds = 0.0005
            means = np.arange(0, 1.01, 1/(n_components-1)) * 0.5
        p_stay = 1 - co_odds
        p_shift = co_odds / 2
        transmat_prior = np.full((n_components, n_components), p_shift)
        np.fill_diagonal(transmat_prior, p_stay)
        hm_model.means_ = means.reshape(-1, 1)
        hm_model.transmat_ = transmat_prior
        hm_model.sample_names = []
        self.HMM = hm_model


    def fit_hmm(self, output_pickle=False):
        self._log(0, f'Fitting HMM. Output file = "{output_pickle}".')
        # Drop the noisy centromeric regions from the input data
        if self._flags['cent_data']:
            self._log(0, f'Dropping centromeric regions from input.')
            temp = self._sample.copy()
            temp['chr_n'] = temp['chrm'].str.replace(r'\D','', regex=True).astype(int)
            temp = temp.merge(self._context_table, left_on=['chr_n'], right_on=['chrm'])
            centromere = temp[(temp.start_x > temp.start_y) & (temp.end_x < temp.end_y)].index
            fit_sequences = [self._sample.drop(centromere).val[self._sample.chrm == chrm].values for chrm in pd.unique(self._sample.chrm)]
        else:
            self._log(2, "Context not available. This might affect the quality of the analysis.")
            fit_sequences = self.sequences
        
        # Re-shape list of observations to nparray for HMM input
        obs_reshape = [np.reshape(obs, (len(obs), 1)) for obs in fit_sequences]
        obs_lengths = [len(seq) for seq in fit_sequences]
        X = np.concatenate(obs_reshape)

        self.HMM.fit(X, obs_lengths)
        self.HMM.sample_names.append(self._sample_name)
        self._flags['is_fit'] = True

        if output_pickle:
            self._log(1, f'Writing updated HMM to "{output_pickle}".')
            try:
                with open(output_pickle, 'wb+') as f:
                    pickle.dump(self.HMM, f)
            except Exception as ex:
                self._log(2, f'Failed to write to "{output_pickle}". Please check the directory exists and has suitable permissions.\n{ex}')



    def predict_hmm(self):
        self._log(0, f'Predicting most likely state based on HMM.')
        if not self._flags['is_fit']:
            self._log(2, "Model has not been fitted. Run CrossoverDetector.fit_hmm first.")
            return
        seq_reshape = [np.reshape(seq, (len(seq), 1)) for seq in self.sequences]
        preds = [self.HMM.predict(seq) for seq in seq_reshape]
        scores = [self.HMM.score(seq) for seq in seq_reshape]   # Not certain if the score is helpful
        
        # Sort prediction labels in ascending order according to mean  
        ## Can be discarded since mean order was normalized in model init
        flat_preds = [i for obs in preds for i in obs]  # Flattens nested lists [[0,1],[2,1]] -> [0,1,2,1]
        idx = np.argsort(self.HMM.means_[:,0])
        lut = np.zeros_like(idx)
        lut[idx] = range(self.HMM.n_components)
        updated_preds = [lut[x] for x in flat_preds]    # Map labels
        
        self.scores = scores
        sample_predictions = self._sample.copy()
        sample_predictions['pred'] = updated_preds
        self._sample_predictions = sample_predictions
    
    def extract_shift_pos(self, column='pred') -> pd.DataFrame:
        self._log(0, f'Exctracting transition points, column = "{column}".')
        sample_diff = self._sample_predictions.copy()
        sample_diff['diff'] = 1
        sample_diff['diff'].iloc[:-1] = sample_diff[column][:-1].values - sample_diff[column][1:].values
        
        sample_diff['same_chr'] = False
        sample_diff['same_chr'].iloc[:-1] = sample_diff['chrm'][:-1].values == sample_diff['chrm'][1:].values
        
        # Extract state between shifts
        found_positions = self._sample_predictions[(sample_diff['diff'] != 0) & (sample_diff['same_chr'] != False)]
        found_positions.loc[:, 'change'] = sample_diff.loc[found_positions.index + 1, column].values
        return found_positions.dropna()

    def smooth_noise(self, sus: pd.DataFrame, window_size=5):    # Fill gaps
        self._sample_predictions['smooth'] = -1
        prev_idx = False
        for idx, row in sus.iterrows():
            if prev_idx and self._sample_predictions.loc[prev_idx, 'chrm'] == row['chrm'] and idx - prev_idx < window_size:
                self._sample_predictions.loc[prev_idx:idx, 'smooth'] = self._sample_predictions.loc[prev_idx, 'pred']
                prev_idx = False
            else:
                prev_idx = idx
        self._sample_predictions['updated'] = np.where(self._sample_predictions['smooth'] != -1, self._sample_predictions['smooth'], self._sample_predictions['pred'])

    def smooth_noise2(self, smooth_frac = 0.06):    # Local regression
        if 'statsmodels.api' not in sys.modules:
            self._log(2, 'statsmodels.api, which is necessary for this smoothing model, has not been imported. Quitting.')
            return
        self._sample_predictions['smooth'] = -1
        for chrm in self._sample_predictions.chrm.unique():
            df_chrm = self._sample_predictions[self._sample_predictions.chrm == chrm]
            lowess = sm.nonparametric.lowess(df_chrm.pred, df_chrm.end, frac=smooth_frac, it=0, is_sorted=True)
            self._sample_predictions.loc[df_chrm.index, 'smooth'] = lowess[:, 1]
        self._sample_predictions['updated'] = self._sample_predictions['smooth'].round()
    
    def smooth_noise3(self, window_size=20):    # Rolling mean
        self._sample_predictions['smooth'] = -1
        for chrm in self._sample_predictions.chrm.unique():
            df_chrm = self._sample_predictions[self._sample_predictions.chrm == chrm]
            local_mean = df_chrm.pred.rolling(window_size, center=True).mean()
            self._sample_predictions.loc[df_chrm.index, 'smooth'] = local_mean
        self._sample_predictions['updated'] = self._sample_predictions['smooth'].round()

    def realign_shifts(self, sus: pd.DataFrame, window_size=20):  # Local max
        alt_sus = []
        for idx, row in sus.iterrows():
            if not (idx < window_size or idx + window_size >= self._sample_predictions.index.max()):    # Skip chrm boundaries
                # Slice `window_size` to each direction
                local_region = self._sample_predictions.loc[idx-window_size:idx+window_size].copy()
                local_region['diff'] = (local_region.val - self._sample_predictions.loc[idx-window_size+1:idx+window_size+1, 'val'].values)
                local_region['diff'] = local_region['diff'] * local_region.loc[idx, 'diff'] # Multiply to discard shifts in the negative direction
                biggest_diff = local_region['diff'].idxmax()
                row_update = self._sample_predictions.loc[biggest_diff]
                row_update['updated'] = row['pred']
                row_update['change'] = row['change']
                alt_sus.append(row_update)
        alt_sus = pd.DataFrame(alt_sus).drop_duplicates()
        return alt_sus

    def plot_hmm(self, comparison=True, pred_column='pred', to_fname=False):
        
        COLORS = ["magenta", "turquoise", "lightgreen", "cyan"]
        names = pd.unique(self._sample.chrm)
        seq_count = len(names)
        max_x = self._sample.end.max()
        y_lim = self._sample.val.max()
        
        if comparison: # Two columns view
            fig = plt.figure(dpi = 100, figsize = (15, 2*seq_count + 2))
        else:
            fig = plt.figure(dpi = 100, figsize = (15, 4*seq_count + 2))
        
        title = "{} (cov = {})\ngroup size = {}, smoothing interval = {}".format(self._sample_name, self._avg_cov,
            self._group_bin_itv, self._smooth_window_size)
        fig.suptitle(title, fontsize=16)
        
        for n, chrm in enumerate(names):
            cur_seq = self._sample_predictions[self._sample_predictions.chrm == chrm]
            # Plot input values
            if comparison:
                col = n // (seq_count/2)
                ax1 = plt.subplot(seq_count, 2, 1 + ((n % (seq_count/2)) * 4) + col)
            else:
                ax1 = plt.subplot(seq_count * 2, 1, (n + 1)* 2 - 1)
            ax1.set_ylabel('Read count')
        
            ax1.scatter(cur_seq.start, cur_seq.val, color = cm.plasma_r((cur_seq.val * 240).astype(int)), linestyle='--')
            ax1.set_xlim((-max_x/40, max_x + max_x/40))
            ax1.set_ylim((-y_lim/8, y_lim + y_lim/8))
            
            if len(names) > 0:
                ax1.set_title(names[n])
            
            # Filter predictions to each state
            masks = [cur_seq[pred_column] == i for i in np.arange(self.HMM.n_components)]
        
            if comparison:
                ax2 = plt.subplot(seq_count, 2, 3 + ((n % (seq_count/2)) * 4) + col)
            else:
                ax2 = plt.subplot(seq_count * 2, 1, (n + 1) * 2)
        
            for i, mask in enumerate(masks):
                ax2.fill_between(cur_seq.start, i+0.1, y2=0, where=mask, color = COLORS[i])
                # if len(self.scores) > 0:
                #     ax2.set_title("Score = {}".format(self.scores[n]))
                ax2.set_xlim((-max_x/40, max_x + max_x/40))
                ax2.set_ylim((-0.2, len(masks) - 0.8))
            ax2.set_ylabel('Prediction')
            ax2.set_yticks([0,1,2])
            ax2.set_yticklabels([0,1,2])
        
            # Plot centromere
            if self._flags['cent_data']:
                cent = self._context_table.loc[(n%(seq_count//2)) + 1]
                ax1.axvspan(cent.start, cent.end, facecolor='blue', alpha=0.2, linewidth=0)
                ax2.axvspan(cent.start, cent.end, facecolor='blue', alpha=0.2, linewidth=0)
            
            ax1.xaxis.set_major_formatter(ticker.EngFormatter())
            ax2.xaxis.set_major_formatter(ticker.EngFormatter())
        # plt.figlegend()
        plt.tight_layout(rect=[0, 0.02, 1, 0.96]) # Rect to account for figure title

        if not to_fname:
            plt.show()
        else:
            plt.savefig(to_fname)
    
    def try_match(self, cutoff=0.05, telomere_size=25000) -> pd.DataFrame:
        xover_locations = self.suspects.copy()
        # Calculate position percentile in chromosome
        genome_names = xover_locations.chrm.str.replace(r'\d', '', regex=True).unique()
        genome_lengths = self._sample_predictions.groupby('chrm').max().end
        xover_locations['abs_pos'] = xover_locations.end / xover_locations.chrm.map(genome_lengths)
        # Add middleman (if 0->2, also consider 0->1)
        shift = (xover_locations.change - xover_locations.updated).abs()
        xover_locations = pd.concat([xover_locations, xover_locations[shift == 2].replace(2, 1)]).sort_index()  # If homo to homo, add hetereo

        xover_dfs = [xover_locations[xover_locations.chrm.str.startswith(genome)] for genome in genome_names]    # Seperate genomes A and B
        found_crossovers = []
        # Iterate over A, find reciprocal matches
        for idx, xover in xover_dfs[0].iterrows():
            if self._flags['syri']:
                # Potentially use syri somehow
                pass
            else:
                chrm = xover['chrm'].replace(genome_names[0], genome_names[1])
                matches = xover_dfs[1][xover_dfs[1].chrm == chrm].copy()
                # Reads need to add up to 2 (homo A + no B, no A + homo B, or hetereo A + hetereo B)
                matches.iloc[:, 2:4] += xover.iloc[2:4].values
                matches = matches[(matches.updated == 2) & (matches.change == 2)]
                matches['dist'] = (matches.abs_pos - xover.abs_pos).abs()
                matches.sort_values(by='dist', inplace=True)
                if len(matches) > 0:
                    xover[f'pos_{genome_names[1]}'] = matches.iloc[0, 1]
                    xover['dist'] = matches.iloc[0, 5]
                    found_crossovers.append(xover)
                    # Zoom in on biggest value diff?
        
        res = pd.DataFrame(found_crossovers)
        res = res[res.dist < cutoff].sort_values(by=['chrm', f'pos_{genome_names[1]}', 'end'])   # Filter far away hits
        dups = res.duplicated(subset=['chrm', f'pos_{genome_names[1]}'], keep=False) # mark duplicates for B
        res['possible_dup'] = dups
        
        # Calc bin size percentile
        bin_sizes = self._sample_predictions.iloc[:, 1:3].diff(axis=1).copy().sort_values(by='end')
        bin_sizes.loc[:, 'binsize_percentile'] = 100*np.arange(len(bin_sizes))/len(bin_sizes)
        res['binsize'] = bin_sizes.loc[res.index, 'end'].astype(int)
        res['binsize_percentile'] = bin_sizes.loc[res.index, 'binsize_percentile'].round(2)

        # Mark context
        dist_to_end = np.abs(res.end - res.chrm.map(genome_lengths))
        res.loc[(dist_to_end < telomere_size) | (res.end < telomere_size), 'context'] = 'telomere'
        if self._flags['cent_data']:
            chrm_stripped = res.chrm.str.replace(r'\D', '', regex=True).astype(int)
            cent = (res.end >= chrm_stripped.map(self._context_table.start)) & (res.end <= chrm_stripped.map(self._context_table.end))
            res.loc[cent, 'context'] = 'centromere'
        
        res = res.round(4)
        self.crossovers = res
        return res

    def refine_crossovers(self):
        def get_source(idx):
            r_chrm = self._sample_predictions.loc[idx, 'chrm']
            r_start = self._sample_predictions.loc[idx-2, 'start']
            r_end = self._sample_predictions.loc[idx+2, 'end']
            if r_start > self._sample_predictions.loc[idx+2, 'start']:
                r_start = r_end - 25000
            if r_end < self._sample_predictions.loc[idx, 'end']:
                r_end = r_start + 25000
            source_range = self._raw_sample[(self._raw_sample.chrm == r_chrm) &
                (self._raw_sample.start >= r_start) &
                (self._raw_sample.end <= r_end)]
            return source_range
        refined_co = []
        max_val = self._raw_sample.val.max()
        for idx, row in self.crossovers.iterrows():
            raw_range = get_source(idx)
            raw_range['val'] = raw_range['val'] / max_val
            raw_range['pred'] = self.HMM.predict(raw_range.val.values.reshape(-1, 1))
            co_dir = list(row[['updated','change']])
            cur_opt = []
            for pos in raw_range.index:
                pos_preds = list(raw_range.loc[pos:pos+1, 'pred'])
                if pos_preds == co_dir:
                    refined_pos = row.copy()
                    refined_pos['end'] = raw_range.loc[pos, 'end']
                    cur_opt.append(refined_pos)
            if len(cur_opt) > 0:
                refined_co.append(cur_opt[len(cur_opt)//2])
        self._flags['refined'] = True
        return pd.DataFrame(refined_co)

    def lift_syri(self, syri_file, genome_a_name):
        names = ['a_chr', 'a_start', 'a_end', 'a_seq', 'b_seq',
                 'b_chr', 'b_start', 'b_end', 'id', 'parent_id', 'ann', 'cp']
        syri = pd.read_csv(syri_file, sep='\t', header=None,
                names = names, na_values='-', usecols=[0,1,2,5,6,7,8,9,10,11])
        syri['a_chr'] = syri['a_chr'].str.replace('Chr', genome_a_name)

        sus_end = self.crossovers.end
        syri_start = syri.a_start.values
        syri_end = syri.a_end.values
        syri_chr = syri.a_chr.values

        sus_idx, syri_idx = np.where((self.crossovers.chrm[:, None] == syri_chr) & # same chromosome
            (((sus_end[:, None] >= syri_start) & (sus_end[:, None] <= syri_end)))) # or bin end in syri region
        
        res = [self.crossovers.iloc[sus_idx].reset_index(), syri.loc[syri_idx].reset_index(drop=True)]

        return pd.concat(res, axis=1)

    def genome_hmm_fit(self, fname: str, context=False):
        time_start = time.time()
        self._log(1, f'Started genome_hmm_fit() on "{self._sample_name}"')
        if self._flags['cent_data']:
            self._log(1, 'Context table already loaded.')
        elif context:
            self._log(1, 'Loading context table.')
            self.read_context(context)
        self._log(1, f'Loading HMM from pickle "{fname}".')
        self.hmm_from_pickle(fname)
        self._log(1, 'Fitting...')
        self.fit_hmm(output_pickle=fname)
        time_end = time.time()
        self._log(1, f'Done. Time taken: {round((time_end - time_start), 3)}sec. Saved to "{fname}".')


    def genome_hmm_detection(self, context=False, hmm_pickle='', smooth_window_size=10, n_components=3,
                            smooth_model='v1', match_cutoff=0.05, refine=False, **kwargs):
        self._log(1, f'Started genome_hmm_detection() on "{self._sample_name}"')
        if self._flags['cent_data']:
            self._log(1, 'Context table already loaded.')
        elif context:
            self._log(1, 'Loading context table.')
            self.read_context(context)
        if len(hmm_pickle) > 0:
            self._log(1, f'Loading HMM from pickle "{hmm_pickle}".')
            self.hmm_from_pickle(hmm_pickle)
        else: 
            self._log(1, 'Initializing model...')
            self._hmm_init(n_components, **kwargs)
        if self._flags['is_fit'] == False:
            self._log(1, 'Fitting...')
            self.fit_hmm()
        self._log(1, 'Predicting.')
        self.predict_hmm()
        
        # todo - clean up this part
        if smooth_model == 'v1':
            self._log(1, 'Extractring suspect crossover positions.')
            suspects = self.extract_shift_pos(column='pred')
            self._log(1, f'Smoothing via model {smooth_model}.')
            self.smooth_noise(suspects, smooth_window_size)
            self._log(1, 'Re-exctracting crossover positions.')
            suspects = self.extract_shift_pos(column='updated')
        elif smooth_model == 'v2':
            self._log(1, f'Smoothing via model {smooth_model}.')
            self.smooth_noise2(smooth_frac=0.06)
        elif smooth_model == 'v3':
            self._log(1, f'Smoothing via model {smooth_model}.')
            self.smooth_noise3(window_size=smooth_window_size)
            self._log(1, 'Re-exctracting crossover positions.')
            suspects = self.extract_shift_pos(column='updated')
        elif smooth_model == 'v4':
            self._log(1, 'Extractring suspect crossover positions.')
            suspects = self.extract_shift_pos(column='pred')
            self._log(1, f'Found {len(suspects)} suspects.')
            self._log(1, 'Re-aligning peaks.')
            suspects = self.realign_shifts(suspects, smooth_window_size)
        else:
            self._log(1, f'Invalid smoothing function type "{smooth_model}". Exiting.')
            return
        self._smooth_window_size = smooth_window_size


        self._log(1, f'Found {len(suspects)} suspects.')
        self.suspects = suspects[['chrm', 'end', 'updated', 'change']]
        self._log(1, f'Looking for reciprocal crossover positions.')
        crossovers = self.try_match(match_cutoff)
        co_no_dups = crossovers.drop_duplicates(subset=crossovers.columns[[0, 2, 3, 5]])
        uncertain = len(crossovers) - len(co_no_dups)
        self._log(1, f'{len(co_no_dups)} crossovers found, {uncertain} more possible duplicate(s).')
        if refine:
            self._log(1, 'Refining crossover locations.')
            ref = self.refine_crossovers()
            self.crossovers = ref
        self._log(1, 'Done.')
    

if __name__ == '__main__':
    # todo: migrate xover_script02 as module default (hmm_analysis -m) 
    test = CrossoverDetector(log_level=2)
    test.preprocess("collab/../novaseq6_09_2019/demultiplexed_fastq/O/KirilsF2_10/KirilsF2_10_S10_.dedup.cov.bedgraph",
                interval=10)
    test.genome_hmm_detection(smooth_window_size=10, smooth_model='v1', hmm_pickle='res_hmm.pickle')
    print(test.crossovers)
    test.refine_crossovers()
    print(test)

