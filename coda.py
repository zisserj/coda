import argparse

import numpy as np
import pandas as pd
# import statsmodels.api as sm  # Used for now deprecated CrossoverDetector._smooth_noise2()
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
        self.log_level = log_level
        self._log(0, f'CrossoverDetector(log_level={log_level})')
        self._flags = {flag: False for flag in ['is_loaded', 'is_fit', 'refined', 'syri', 'cent_data']}
    
    def __repr__(self):
        var_list = vars(self).copy()
        if 'HMM' in var_list:
            var_list['HMM'] = f'GaussianHMM(n_components = {self.HMM.n_components}, trained_samples({len(self.HMM.sample_names)})'
        var_list = [f'{str(name)} = {str(val)}' if type(val) not in [list, pd.DataFrame] else f'{str(name)}({len(val)})' for name, val in var_list.items()]
        rep = 'CrossoverDetector({})'.format(',\n'.join(sorted(var_list)))
        return rep

    def _log(self, level: int, message: str):
        level_names = {0:'DEBUG', 1:'INFO', 2:'WARN', 3:'ERROR'}
        if level < self.log_level:
            return
        else:
            ts = datetime.datetime.now().isoformat(sep=' ', timespec='milliseconds')
            msg = f'[{ts}, {level_names[level].rjust(5)}] ' + message
            print(msg)
        if level >= 3:
            exit()

    def load_data(self, bedgraph: str, method='arbitrary', group_count=20, bin_size=6000):
        self._log(0, 'load_data(bedgraph="{}", method="{}", group_count={}, bin_size={}.'.format(\
                        bedgraph, method, group_count, bin_size))
        # Load coverage file
        try:
            data = pd.read_csv(bedgraph, sep='\t', header=None, names=['chrm', 'start', 'end', 'val'])
            data = data[data.val <= np.percentile(data.val, 99)].reset_index(drop=True) # drop outliers
        except Exception as ex:
            self._log(3, f'Failed to read bedgraph "{bedgraph}". Quitting.')
            return
        self._raw_data = data

        # Groups data according to supplied method
        data_agg = data.copy()
        if method == 'arbitrary':
            data_agg['new_idx'] = data.index // group_count
            data_agg = data_agg.groupby(['chrm', 'new_idx']).agg({'start':'min',
                'end':'max', 'val':'mean'}).reset_index(0).reset_index(drop=True)
        elif method == 'robust':
            for chrm in np.unique(data['chrm']):
                chrm_df = data_agg[data_agg['chrm'] == chrm].index
                bins = np.arange(1, data_agg.loc[chrm_df, 'start'].max(), bin_size)
                bin_num = np.digitize(data_agg.loc[chrm_df, 'start'], bins)
                data_agg.loc[chrm_df, 'bin_num'] = bin_num
            data_agg = data_agg.groupby(['chrm', 'bin_num']).agg({'start':'min',
                'end':'max', 'val':'mean'}).reset_index(0).reset_index(drop=True)
        else:
            self._log(3, 'Unrecognised aggregation method. Quitting.')
        data_agg.val = data_agg.val / data_agg.val.max() # normalise to [0-1]
        self._data = data_agg

        # Transform data to a list of observations
        self.sequences = [self._data.val[self._data.chrm == chrm].values for chrm in pd.unique(self._data.chrm)]
        
        # Save kwargs
        self._sample_name = bedgraph.split("/")[-1]
        self._group_count = group_count
        self._avg_cov = self._raw_data.val.mean().round(3)
        self._flags['is_loaded'] = True

        self._log(1, f"Successfully loaded '{self._sample_name}'. Mean coverage is {self._avg_cov}.")

    def _read_context(self, fname: str):
        self._log(0, f'_read_context(fname="{fname}")')
        df = pd.read_csv(fname, sep='\t', header=None, names=['chrm','start','end'], usecols=[0,1,2])
        df.chrm = df.chrm.str.replace(r'\D', '', regex=True).astype(int)    # Strip notation
        df.set_index('chrm', inplace=True)
        self._context_table = df
        self._flags['cent_data'] = True
        
    def _hmm_init(self, n_components=3, **kwargs):
        self._log(0, f'_hmm_init(n_components="{n_components}", **kwargs={kwargs})')
        if n_components < 2:
            self._log(2, f'Can not initiate model with fewer than 2 states. Quitting.')
            return
        hm_model = hmm.GaussianHMM(n_components=n_components, n_iter=1000, init_params='sct', **kwargs)            
        # HMM initial behaviour adjustments
        if self._flags['is_loaded']:
            co_odds = 10 / len(self._data) # Approx 10 COs 
            means = np.arange(0, 1.01, 1/(n_components-1)) * self._data.val.mean() * 2
        else:
            self._log(2, 'Sample not loaded. Setting default parameters.')
            co_odds = 0.0005
            means = np.arange(0, 1.01, 1/(n_components-1)) * 0.5
        p_stay = 1 - co_odds
        p_shift = co_odds / 2
        transmat_prior = np.full((n_components, n_components), p_shift)
        np.fill_diagonal(transmat_prior, p_stay)
        hm_model.transmat_ = transmat_prior
        hm_model.means_ = means.reshape(-1, 1)
        hm_model.sample_names = []
        self.HMM = hm_model

    def _load_model(self, fname: str):
        self._log(0, f'_load_model(fname="{fname}")')
        try:
            with open(fname, 'rb+') as f:
                hm_model = pickle.load(f)
                self._flags['is_fit'] = True
                self.HMM = hm_model
        except Exception as ex:
            self._log(2, f'Failed to load HMM pickle from file "{fname}". Initializing new model. {ex}')
            self._hmm_init()

    def _fit_model(self, output_file=False):
        self._log(0, f'_fit_model(output_file="{output_file}")')
        # Drop the noisy centromeric regions from the input data
        if self._flags['cent_data']:
            self._log(0, f'Dropping centromeric regions from input.')
            temp = self._data.copy()
            temp['chr_n'] = temp['chrm'].str.replace(r'\D','', regex=True).astype(int)
            temp = temp.merge(self._context_table, left_on=['chr_n'], right_on=['chrm'])
            centromere = temp[(temp.start_x > temp.start_y) & (temp.end_x < temp.end_y)].index
            fit_sequences = [self._data.drop(centromere).val[self._data.chrm == chrm].values for chrm in pd.unique(self._data.chrm)]
        else:
            self._log(2, "Context not available. This might affect the quality of the analysis.")
            fit_sequences = self.sequences
        
        if not self._flags['is_fit']:
            self._log(2, 'Initializing new model.')
            self._hmm_init()

        # Re-shape list of observations to nparray for HMM input
        obs_reshape = [np.reshape(obs, (len(obs), 1)) for obs in fit_sequences]
        obs_lengths = [len(seq) for seq in fit_sequences]
        X = np.concatenate(obs_reshape)

        self.HMM.fit(X, obs_lengths)
        self.HMM.sample_names.append(self._sample_name)
        self._flags['is_fit'] = True

        if output_file:
            self._log(1, f'Writing updated HMM to "{output_file}".')
            try:
                with open(output_file, 'wb+') as f:
                    pickle.dump(self.HMM, f)
            except Exception as ex:
                self._log(3, f'Failed to write to "{output_file}". Please check the directory exists and has suitable permissions.\n{ex}')

    def _predict_hmm(self):
        self._log(0, f'_predict_hmm()')
        if not self._flags['is_fit']:
            self._log(3, "Model has not been fitted.")
            return
        seq_reshape = [np.reshape(seq, (len(seq), 1)) for seq in self.sequences]
        preds = [self.HMM.predict(seq) for seq in seq_reshape]
        probs = [self.HMM.predict_proba(seq) for seq in seq_reshape]
        scores = [self.HMM.score(seq) for seq in seq_reshape]   # Not certain if the score is helpful
        
        # Sort prediction labels in ascending order according to mean  
        ## Can be discarded since mean order was normalized in model init
        flat_preds = [i for obs in preds for i in obs]  # Flattens nested lists [[0,1],[2,1]] -> [0,1,2,1]
        idx = np.argsort(self.HMM.means_[:,0])
        lut = np.zeros_like(idx)
        lut[idx] = range(self.HMM.n_components)
        updated_preds = [lut[x] for x in flat_preds]    # Map labels
        
        self.scores = scores
        data_predictions = self._data.copy()
        data_predictions['pred'] = updated_preds
        data_predictions['probs'] = [i.round(3) for chrm in probs for i in chrm]
        self._data_predictions = data_predictions

    def _extract_shift_pos(self, column='pred'):
        self._log(0, f'_extract_shift_pos(column="{column}")')
        data_diff = self._data_predictions.copy()
        data_diff['diff'] = 1
        data_diff['diff'].iloc[:-1] = data_diff[column][:-1].values - data_diff[column][1:].values
        
        data_diff['same_chr'] = False
        data_diff['same_chr'].iloc[:-1] = data_diff['chrm'][:-1].values == data_diff['chrm'][1:].values
        
        # Extract state between shifts
        found_positions = self._data_predictions[(data_diff['diff'] != 0) & (data_diff['same_chr'] != False)]
        found_positions.loc[:, 'change'] = data_diff.loc[found_positions.index + 1, column].values
        self._suspects = found_positions.dropna()

    def _smooth_noise(self, window_size=5):    # Fill gaps
        self._log(0, f'_smooth_noise(window_size={window_size})')
        self._data_predictions['smooth'] = -1
        prev_idx = False
        for idx, row in self._suspects.iterrows():
            if prev_idx and self._data_predictions.loc[prev_idx, 'chrm'] == row['chrm'] and idx - prev_idx < window_size:
                self._data_predictions.loc[prev_idx:idx, 'smooth'] = self._data_predictions.loc[prev_idx, 'pred']
                prev_idx = False
            else:
                prev_idx = idx
        self._data_predictions['updated'] = np.where(self._data_predictions['smooth'] != -1, self._data_predictions['smooth'], self._data_predictions['pred'])

    def _smooth_noise2(self, smooth_frac=0.06):    # Local regression
        self._log(0, f'_smooth_noise2(smooth_frace={smooth_frac})')
        if 'statsmodels.api' not in sys.modules:
            self._log(2, 'statsmodels.api, which is necessary for this smoothing model, has not been imported. Quitting.')
            return
        self._data_predictions['smooth'] = -1
        for chrm in self._data_predictions.chrm.unique():
            df_chrm = self._data_predictions[self._data_predictions.chrm == chrm]
            lowess = sm.nonparametric.lowess(df_chrm.pred, df_chrm.end, frac=smooth_frac, it=0, is_sorted=True)
            self._data_predictions.loc[df_chrm.index, 'smooth'] = lowess[:, 1]
        self._data_predictions['updated'] = self._data_predictions['smooth'].round()
    
    def _smooth_noise3(self, window_size=20):    # Rolling mean
        self._log(0, f'_smooth_noise3(window_size={window_size})')
        self._data_predictions['smooth'] = -1
        for chrm in self._data_predictions.chrm.unique():
            df_chrm = self._data_predictions[self._data_predictions.chrm == chrm]
            local_mean = df_chrm.pred.rolling(window_size, center=True).mean()
            self._data_predictions.loc[df_chrm.index, 'smooth'] = local_mean
        self._data_predictions['updated'] = self._data_predictions['smooth'].round()

    def _realign_shifts(self, sus: pd.DataFrame, window_size=20):  # Local max*
        self._log(0, f'_realign_shifts(sus={(len(sus))}, window_size={window_size})')
        alt_sus = []
        for idx, row in sus.iterrows():
            if not (idx < window_size or idx + window_size >= self._data_predictions.index.max()):    # Skip chrm boundaries
                # Slice `window_size` to each direction
                local_region = self._data_predictions.loc[idx-window_size:idx+window_size].copy()
                local_region['diff'] = (local_region.val - self._data_predictions.loc[idx-window_size+1:idx+window_size+1, 'val'].values)
                local_region['diff'] = local_region['diff'] * local_region.loc[idx, 'diff'] # Multiply to discard shifts in the negative direction
                biggest_diff = local_region['diff'].idxmax()
                row_update = self._data_predictions.loc[biggest_diff]
                row_update['updated'] = row['pred']
                row_update['change'] = row['change']
                alt_sus.append(row_update)
        alt_sus = pd.DataFrame(alt_sus).drop_duplicates()
        return alt_sus

    def plot_hmm(self, comparison=True, val_column='pred', fname=False):
        self._log(0, f'plot_hmm(comparison={comparison}, val_column="{val_column}", fname="{fname}")')
        
        COLORS = ["magenta", "turquoise", "lightgreen", "cyan"]
        names = pd.unique(self._data.chrm)
        seq_count = len(names)
        max_x = self._data.end.max()
        y_lim = self._data.val.max()
        
        if comparison: # Two columns view
            fig = plt.figure(dpi = 100, figsize = (15, 2*seq_count + 2))
        else:
            fig = plt.figure(dpi = 100, figsize = (15, 4*seq_count + 2))
        
        title = "{} (cov = {})\ndatapoints per bin  = {}, smoothing interval = {}".format(self._sample_name, self._avg_cov,
            self._group_count, self._smooth_window)
        fig.suptitle(title, fontsize=16)
        
        for n, chrm in enumerate(names):
            cur_seq = self._data_predictions[self._data_predictions.chrm == chrm]
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
            masks = [cur_seq[val_column] == i for i in np.arange(self.HMM.n_components)]
        
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

        if not fname:
            plt.show()
        else:
            plt.savefig(fname)
    
    def try_match(self, cutoff=0.05, telomere_size=25000) -> pd.DataFrame:
        self._log(0, f'try_match(cutoff={cutoff}, telomere_size={telomere_size})')
        xover_locations = self._suspects.copy()
        # Calculate position percentile in chromosome
        genome_names = xover_locations.chrm.str.replace(r'\d', '', regex=True).unique()
        genome_lengths = self._data_predictions.groupby('chrm').max().end
        xover_locations['abs_pos'] = xover_locations.end / xover_locations.chrm.map(genome_lengths)
        # Add middleman (if 0->2, also consider 0->1)
        shift = (xover_locations.change - xover_locations.updated).abs()
        xover_locations = pd.concat([xover_locations, xover_locations[shift == 2].replace(2, 1)]).sort_index()  # If homo to homo, add hetereo

        xover_dfs = [xover_locations[xover_locations.chrm.str.startswith(genome)] for genome in genome_names]    # Seperate genomes A and B
        found_crossovers = []
        # Iterate over A, find reciprocal matches
        for idx, xover in xover_dfs[0].iterrows():
            chrm = xover['chrm'].replace(genome_names[0], genome_names[1])
            matches = xover_dfs[1][xover_dfs[1].chrm == chrm].copy()
            # Reads need to add up to 2 (homo A + no B, no A + homo B, or hetereo A + hetereo B)
            matches.loc[:,['updated', 'change']] += xover.loc[['updated', 'change']].values
            matches = matches[(matches.updated == 2) & (matches.change == 2)]
            matches['dist'] = (matches.abs_pos - xover.abs_pos).abs()
            matches.sort_values(by='dist', inplace=True)
            if len(matches) > 0:
                xover[f'pos_{genome_names[1]}'] = matches.iloc[0].end
                xover['dist'] = matches.iloc[0].dist
                found_crossovers.append(xover)
        if len(found_crossovers) == 0:
            self._log(3, 'No crossovers found. Please make sure the data is valid.')
            return
        res = pd.DataFrame(found_crossovers)
        res = res[res.dist < cutoff].sort_values(by=['chrm', f'pos_{genome_names[1]}', 'end'])   # Filter far away hits
        dups = res.duplicated(subset=['chrm', f'pos_{genome_names[1]}'], keep=False) # mark duplicates for B
        res['possible_dup'] = dups
        
        # Calc bin size percentile
        bin_sizes = self._data_predictions.iloc[:, 1:3].diff(axis=1).copy().sort_values(by='end')
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
        return res.copy()

    def _refine_crossovers(self) -> pd.DataFrame:
        self._log(0, f'_refine_crossovers()')
        def get_source(idx: int):
            r_chrm = self._data_predictions.loc[idx, 'chrm']
            r_start = self._data_predictions.loc[idx-2, 'start']
            r_end = self._data_predictions.loc[idx+2, 'end']
            if r_start > self._data_predictions.loc[idx+2, 'start']:
                r_start = r_end - 25000
            if r_end < self._data_predictions.loc[idx, 'end']:
                r_end = r_start + 25000
            source_range = self._raw_data[(self._raw_data.chrm == r_chrm) &
                (self._raw_data.start >= r_start) &
                (self._raw_data.end <= r_end)]
            return source_range
        refined_co = []
        max_val = self._raw_data.val.max()
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

    def lift_syri(self, syri_file):
        self._log(0, f'lift_syri(syri_file="{syri_file}")')
        names = ['a_chr', 'a_start', 'a_end', 'a_seq', 'b_seq',
                 'b_chr', 'b_start', 'b_end', 'id', 'parent_id', 'ann', 'cp']
        syri = pd.read_csv(syri_file, sep='\t', header=None,
                names = names, na_values='-', usecols=[0,1,2,5,6,7,8,9,10,11])
        genome_a_name = self._data.iloc[0:1,0].str.replace( r'[\d]+','', regex=True)
        syri['a_chr'] = syri['a_chr'].str.replace('Chr', genome_a_name)
        sus_end = self.crossovers.end
        syri_start = syri.a_start.values
        syri_end = syri.a_end.values
        syri_chr = syri.a_chr.values

        sus_idx, syri_idx = np.where((self.crossovers.chrm[:, None] == syri_chr) & # same chromosome
            (((sus_end[:, None] >= syri_start) & (sus_end[:, None] <= syri_end)))) # or bin end in syri region
        
        res = [self.crossovers.iloc[sus_idx].reset_index(), syri.loc[syri_idx].reset_index(drop=True)]
        return pd.concat(res, axis=1)

    def coda_preprocess(self, bedgraph: str, context_fname: str, model_fname = '',
            method='arbitrary', group_count=20, bin_size=6000):
        self._log(0, 'coda_preprocess(bedgraph="{}", method="{}", group_count={}, bin_size={}.'.format(
                        bedgraph, method, group_count, bin_size))
        self._log(1, f'Loading and aggregating data from file "{bedgraph}" according to method "{method}".')
        self.load_data(bedgraph=bedgraph, method=method, group_count=group_count, bin_size=bin_size)
        if context_fname:
            self._log(1, f'Reading centromere boundries from file "{context_fname}".')
            self._read_context(context_fname)
        if model_fname:
            self._log(1, f'Attempting to load trained hidden markov model from file "{model_fname}".')
            self._load_model(model_fname)


    def coda_fit(self, fname: str):
        self._log(0, f'coda_fit(fname="{self._sample_name}")')
        if not self.flags['is_fit']:
            self._log(1, f'Initializing new HM model.')
            self._hmm_init()
        time_start = time.time()
        self._fit_model(output_file=fname)
        time_end = time.time()
        self._log(1, f'Done. Time taken: {round((time_end - time_start), 3)}sec. Saved to "{fname}".')


    def coda_detect(self, smooth_model='v1', smooth_window=10, match_cutoff=0.05, refine=False, **kwargs):
        self._log(0, f'coda_detect(smooth_model={smooth_model}, ' +
            f'smooth_window={smooth_window}, match_cutoff={match_cutoff}, refine={refine}, {kwargs})')
        self._log(1, f'Started genome_hmm_detection() on "{self._sample_name}"')
        if self._flags['is_fit'] == False:
            self._log(1, 'Fitting HMM based on this sample.')
            self._fit_model()
        
        self._log(1, 'Predicting most likely state according to HMM.')
        self._predict_hmm()
        
        self._log(1, 'Extractring initial transition locations.')
        self._extract_shift_pos(column='pred')

        if smooth_model == 'v1':
            self._log(1, f'Found {len(self._suspects)} suspects.')
            self._log(1, f'Smoothing by filling gaps <{smooth_window}.')
            self._smooth_noise(smooth_window)
            self._log(1, 'Re-exctracting crossover positions.')
            suspects = self._extract_shift_pos(column='updated')
        elif smooth_model == 'v2':
            self._log(1, f'Smoothing by applying local ({match_cutoff}) regression.')
            self._smooth_noise2(smooth_frac=match_cutoff)
        elif smooth_model == 'v3':
            self._log(1, f'Smoothing by applying rolling mean. Window size = {smooth_window}.')
            self._smooth_noise3(window_size=smooth_window)
            self._log(1, 'Re-exctracting crossover positions.')
            suspects = self._extract_shift_pos(column='updated')
        elif smooth_model == 'v4':
            self._log(1, f'Found {len(self.suspects)} suspects.')
            self._log(1, f'Re-aligning peaks by maximal difference in local (window = {smooth_window}) area.')
            suspects = self._realign_shifts(suspects, smooth_window)
        else:
            self._log(2, f'Invalid smoothing function type "{smooth_model}". Attempting to continue.')
        self._smooth_window = smooth_window
        self._log(1, f'Found {len(self._suspects)} suspects post smoothing.')

        self._log(1, f'Looking for reciprocal crossover positions.')
        self.try_match(match_cutoff)
        co_filter = self.crossovers[~self.crossovers.possible_dup]
        co_filter = co_filter[co_filter.context.isna()]
        uncertain = len(self.crossovers) - len(co_filter)
        self._log(1, f'{len(co_filter)} crossovers found, {uncertain} more possible positions(s).')
        if refine:
            self._log(1, 'Refining crossover locations.')
            ref = self._refine_crossovers()
            self.crossovers = ref
        self._log(1, 'Done.')
    
    def coda_output(self, prefix, table=True, plot=True, raw_predictions=False, syri=None):
        self._log(0, f'coda_output(prefix="{prefix}", table={table}, plot={plot}, raw_predictions={raw_predictions})')
        self._log(1, f'Outputting results to files, dir = "{prefix}".')

        if not any([table, plot, raw_predictions, syri]):
            self._log(2, f'No output options selected.')
            return
        
        if table:
            self.crossovers.to_csv(prefix + "_crossovers.csv", index=False)
        if plot:
            self.plot_hmm(fname=prefix + '_predictions.png')
        if raw_predictions:
            self._data_predictions.to_csv(prefix + "_raw_predictions.csv")
        if syri:
            self.lift_syri(syri).to_csv(prefix + '_syri.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coda: Train and predict crossover positions using Hidden Markov Models.')

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument('-v', '--verbose', action='store_true')
    verbosity.add_argument('-q', '--quiet', action='store_true')

    parser.add_argument('--method', type=str, choices=['arbitrary', 'robust'], default='arbitrary', help='Group aggregation method.')
    parser.add_argument('--group-count', type=int, default=20, help='[Arbitrary] Number of points to be aggregated per bin (20).')
    parser.add_argument('--bin-size', type=int, default=8000, help='[Robust] Size of bin, in which all points are aggregated (8000).')
    parser.add_argument('--context', type=str, help='Genome centromere (noisy) positions to be ignored.')
    parser.add_argument('--hmm', '--model', type=str, help='Trained model to use.')
    parser.add_argument('bedgraph', type=str, help='Sample coverage file, produced by bedtools.')

    subparsers = parser.add_subparsers(dest='function', required=True, help='Available functionalities')
    
    parser_fit = subparsers.add_parser('fit', help='Use sample to fit a hidden markov model.')
    parser_fit.add_argument('-output-hmm', type=str, required=True, help='File to store the learned model parameters. Can be the same file as --hmm.')

    parser_predict = subparsers.add_parser('predict', help='Detect F2 sample crossover locations.')
    parser_predict.add_argument('--smooth-model', type=str, default='v1', help='Method of dealing with noise. Detailed in docs. (v1)')
    parser_predict.add_argument('--smooth-window', type=int, default=10, help='Maximum window of bins by which a transition is considered "noise". (10)')
    parser_predict.add_argument('--match-cutoff', type=float, default=0.05, help='Percentage of genome to look at for reciprocal crossovers. (0.05)')
    parser_predict.add_argument('--refine', action='store_true', default=False, help='[EXPERIMENTAL] Attempt to improve position accuracy by using the raw data.')
    parser_predict.add_argument('--syri', type=str, default='', help='[EXPERIMENTAL] Exctract crossover positions from Syri file.')
    parser_predict.add_argument('--output-raw', action='store_true', help='Output raw per bin predictions.')
    
    args = parser.parse_args()
    
    # Common processing
    log_level = 1
    if args.quiet:
        log_level += 1
    if args.verbose:
        log_level -= 1
    
    sample = CrossoverDetector(log_level=log_level)
    sample.coda_preprocess(bedgraph=args.bedgraph, context_fname=args.context, model_fname=args.hmm,
            method=args.method, group_count=args.group_count, bin_size=args.bin_size)
    
    # Fitting
    if args.function == 'fit':
        sample.coda_fit(args.output_hmm)
    
    # Predicting
    if args.function == 'predict':
        sample.coda_detect(smooth_model=args.smooth_model,
            smooth_window=args.smooth_window, match_cutoff=args.match_cutoff, refine=args.refine)
        sample.coda_output(args.bedgraph, syri=args.syri, raw_predictions=args.output_raw)

    # syri = sample.lift_syri('/home/labs/alevy/zisserh/datasets/TAIR10/syri_denovo.out', 'col')
    # syri.to_csv(sample_name + ".syri", sep='\t', index=False)

