
# About crossover detection and analysis pipeline - coda


Coda is a tool which predicts and visualizes crossovers in F2 samples. It aims to guide the user though a step-by-step procedure from raw reads (fastq files) to results in the form of plots and tables.

The core hypothesis of this project is that given assemblies of two plants A and B, reads from plant A would map to assembly A over B. As such, Coda relies on alignment to what we termed a "dual genome" - a concatnated reference of the two parents.  

Development of Coda was done in [Weizmann Institute, Avraham Levy's Lab](https://www.weizmann.ac.il/plants/levy/) by Jules Zisser.

## Pipeline prerequisites  
- [BWA](http://bio-bwa.sourceforge.net/bwa.shtml#3)
- [samtools](http://www.htslib.org/doc/samtools.1.html)
- [Picard](https://software.broadinstitute.org/gatk/documentation/tooldocs/4.0.4.0)
- [bedtools](https://bedtools.readthedocs.io/en/latest)
- [Python 3.7+](https://www.python.org) with:
    - pandas
    - matplotlib
    - [hmmlearn](https://hmmlearn.readthedocs.io/en/latest/)

# Coda workflow

![coda_workflow.png](README_files/coda_workflow.png)

## Assembling the reference
1. **Change fasta file headers.** Determines the chromosome notation used downstream. Use distinct, alphabetical names. *Do not* use digits.
```bash
 awk '/^>/{print ">alpha" ++i; next}{print}' < genomeA.fasta
 awk '/^>/{print ">beta" ++i; next}{print}' < genomeB.fasta
 ```
2. **Concatenate reference files.** This is the reference all samples are mapped to.
```bash
cat genomeA.fasta genomeB.fasta > genomeAB.fasta
```
3. **Generate genome sizes.** Used to generate bins. View `genomeAB_sizes.txt` afterwards to verify the genome names and sizes.
```bash
# module load samtools-gnu 
samtools faidx genomeAB.fasta
cut -f1,2 genomeAB.fasta.fai > genomeAB_sizes.txt # notice input is .fai here
```
4. **Index concatenated reference.** Required for BWA alignment.
```bash
# module load bwa-gnu
bwa index genomeAB.fasta
```

## Create bin template
5. **Map source alignments and filter.** Using `BWA mem`, map each pair of reads `genomeA_R1_00N.fastq.gz`, `genomeB_R2_00N.fastq.gz` to `genomeAB.fasta` (`-t 5` is the amount of parallel threads, it does not alter the alignment result). Pipe the output to samtools in order to filter out reads with quality / Fred score < 30 and convert to BAM format (binary / smaller form of SAM).
```bash
# module load bwa-gnu samtools-gnu
for R1 in genomeA_control_L001_00*R1.fastq.gz; 
do
    bwa mem -t 5 references/genomeAB.fasta $R1 $R2 | samtools view -Sbh -q 30 - > genomeA_control_00N.bam
done
```
4. **Merge and sort** all alignment files of the same library.
```bash
samtools merge genomeA_control_001.bam genomeA_control_002.bam  [...] > genomeA_control_merged.bam
samtools sort genomeA_control_merged.bam > genomeA_control_merged.sorted.bam
```
6. **Deduplicate** in order to retain unique reads
```bash
# module load jdk picard
picard MarkDuplicates I=genomeA_control_merged.sorted.bam O=genomeA_control_merged.sorted.dedup.bam M=genomeA_control_merged.dedup_metrics.txt REMOVE_DUPLICATES=true
```
7. **Create bins table** (decide on bin size, step size) using `make_bins.py`.
```bash
python make_bins.py genomeAB_sizes.txt 200
```
8. **Calcuate control coverage** to see where each parent aligns
```bash
bedtools coverage -counts -sorted -a genomeAB_sizes.bins.txt -b genomeA_control_merged.sorted.dedup.bam > genomeA_control.cov-sizes.bedgraph
```
8. **Filter uninformative bins** with `filter_genomes.py`. This results in reads that aren't unique to either parent to be thrown out, resulting in a much cleaner template.
```bash
python filter_genomes.py genomeA_control.cov-sizes.bedgraph genomeB_control.cov-sizes.bedgraph genomeAB_template.cov-sizes.bedgraph
```
9. **Optional: Calculate average bin coverage**.
```bash
cat genomeA_control.cov-sizes.bedgraph | awk '{sum+=$4} END { print "Average = ",sum/NR}'
```


## Prepare individual dataset
Sample level steps can be mass deployed using the script `processing_pipeline.sh` coupled with `dirs_process.sh`.  
The pipeline expects as an input the R1 fastq file of a reads pair.  
All of the commands are the same as processing the control samples, except for the last step: when calculating coverage, use the template `genomeAB_template.cov-sizes.bedgraph` generated from `filter_genome.py` in step 2.2.9 instead of `genomeAB_sizes.bins.txt`.

1. Map sample reads to concatenated reference `genomeAB.fasta`
2. Quality filter and sort
3. Remove duplicates
4. Get coverage over bin template using bedtools

## Analysis with Coda
There are multiple python dependencies required in order for `coda.py` to run. The recommended way to set up a suitable environment is by using [Anaconda3](https://www.anaconda.com/distribution/).
```bash
module load anaconda # for servers that use "module" imports such as wexac
conda env create -f coda_env.yml # creates a virtual environment with specified requirements
conda activate xoe # activate it
python coda.py my_sample.dedup.cov.bedgraph predict # running the default pipeline on file "my_sample"
```
For advanced users who would rather use their own environments, the dependencies are listed below. Later versions of the packages are likely to work, but were not tested.
```
  - python=3.7.*                                   
  - pandas=0.24.*
  - matplotlib=3.1.2
  - hmmlearn=0.2.3
```

# Module documentation

## transform_data() - aggregating read count
Two methods: `arbitrary` or `robust`, which use `group_count` or `bin_size` respectively.
`Arbitrary` is used by default.  
Plotted below is an example of the sample data processed by the different methods.

**Arbitrary - aggregate every `group_count` points.**
* Consistent distribution across samples
* Less bins
* Clear Heterozygote 0 group
* Better results overall

**Robust - all points in each `bin_size` section are grouped**
* Distribution highly affected by coverage
* More bins
* Bins contain very few datapoints
* Better defined aggregation


```python
fig = plt.figure(figsize=(12,4))

for i, method in enumerate(['arbitrary', 'robust']):
    s = coda.CrossoverDetector(log_level=2)
    s.transform_data(kiril_files(10), method=method)
    ax = plt.subplot(1,2,i+1)
    ax.hist(s._data.val, bins=np.arange(0, 1.01, 0.05), color='skyblue')
    ax.set_ylabel('Bin count')
    ax.set_xlabel('Normalised Reads per bin')
    ax.set_title(f'Grouping by {method.capitalize()} Method (total={len(s._data)})')
plt.tight_layout()
```


![transform_data.png](README_files/transform_data.png)


## read_context() - input centromere data
Centromeres tend to have a lot of variations. Adding centromere positions means these inconsistent regions will be taken out before training, as well as marked in the results.  
As of 2019, the context positions in `fname` needs to be listed as follows:

#| end | end
--- | --- | ---
chr1  | 14000000  | 15599999
chr2  | 2900000 | 3949999
chr3  | 13600000  | 14549999
chr4  | 2000000 | 4259999
chr5  | 10930000  | 12659999


## Hidden markov model training
By default the model will be initialized with 3 states - Homozygote A, Heterozygote, and Homozygote B. Each state is defined by mean, variance, and transition odds. During the fitting process these parameters are adjusted in a process called [Expectation-maximization](https://www.statisticshowto.datasciencecentral.com/em-algorithm-expectation-maximization/), or EM for short. It's recommended to train the model on multiple samples in order to produce more reliable results.

Approximate position offset:

    min       166.50
    p25      2364.25
    p50      6860.50
    p75     13613.50
    max    331572.50



## [Experimental] Syri - genome matching

Note: Syri was considered as a possibility to match crossover locations during development, but was eventually discarded.

```bash
module load mummer
nucmer --maxmatch -c 100 -b 500 -l 50 refgenome qrygenome
delta-filter -m -i 90 -l 100 out.delta > out_m_i90_l100.delta
show-coords -THrd out_m_i90_l100.delta > out_m_i90_l100.coords
~/apps/syri/syri/bin/syri -c out_m_i90_l100.coords -r TAIR10_filtered.fasta -q ../Denovo_Ler_Assembly/GCA_001651475.1_Ler_Assembly_genomic_filtered.fasta -d out_m_i90_l100.delta --nc 5
```
Sources:  
[Syri Publication Additional File 2 Col-0-Ler](https://www.biorxiv.org/content/10.1101/546622v3.supplementary-material)  
https://www.biostars.org/p/49820/  
[Assembly match 3 Syri](https://schneebergerlab.github.io/syri/example.html)  

# Python usage example

```python
import coda

# Notebook only settings
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

Coda can be used as either a script or a module, which share the same functions. In this example it will be used as a python module.  
The same result can be achieved by running the script like so:
```bash
python coda.py --group-count 10 --context centromere_pos.bed --hmm res_hmm.pickle KirilsF2_7_S7_.dedup.cov.bedgraph predict --smooth-model v1 --smooth-window 10
```

We begin by importing coda to our project, as well as other packages used for data manipulation and visualisation.


```python
sample_file = 'README_files/example.dedup.cov-200.bedgraph'
sample1 = coda.CrossoverDetector()
sample1.coda_preprocess(sample_file, context_fname='centromere_pos.bed', group_count=10)
                        #'res_hmm.pickle'
sample1.coda_detect(smooth_model='v1', smooth_window=10)
```

    [2020-01-23 14:50:01.754,  INFO] Loading and aggregating data from file "/home/labs/alevy/zisserh/proj/crossover_detection/README_files/example.dedup.cov-200.bedgraph" according to method "arbitrary".
    [2020-01-23 14:50:02.418,  INFO] Successfully loaded 'README_files/example.dedup.cov-200.bedgraph'. Mean coverage is 10.732.
    [2020-01-23 14:50:02.418,  INFO] Reading centromere boundries from file "centromere_pos.bed".
    [2020-01-23 14:50:02.425,  INFO] Started genome_hmm_detection() on "KirilsF2_7_S7_.dedup.cov.bedgraph"
    [2020-01-23 14:50:02.425,  INFO] Fitting HMM based on this sample.
    [2020-01-23 14:50:02.603,  WARN] Initializing new model.
    [2020-01-23 14:50:06.257,  INFO] Predicting most likely state according to HMM.
    [2020-01-23 14:50:06.784,  INFO] Extractring initial transition locations.
    [2020-01-23 14:50:06.806,  INFO] Found 289 suspects.
    [2020-01-23 14:50:06.806,  INFO] Smoothing by filling gaps <10.
    [2020-01-23 14:50:06.961,  INFO] Re-exctracting crossover positions.
    [2020-01-23 14:50:06.983,  INFO] Found 29 suspects post smoothing.
    [2020-01-23 14:50:06.983,  INFO] Looking for reciprocal crossover positions.
    [2020-01-23 14:50:07.277,  INFO] 6 crossovers found, 2 more possible duplicate(s).
    [2020-01-23 14:50:07.278,  INFO] Done.



```python
plt.figure(figsize=(8,4))
sample1._raw_data.val.hist(bins=np.arange(sample1._raw_data.val.max()), label='Read count')
plt.ylabel('Bin count')
plt.xlabel('Reads per bin')
plt.title('Pre Transformation Coverage Per Bin Distribution')
```


![pre_transformation.png](README_files/pre_transformation.png)



```python
plt.figure(figsize=(8,4))
plt.ylabel('Bin count')
plt.xlabel('Reads per bin')
plt.title('HMM Determined State Distribution')

for pred, name in enumerate(['Homozygote 0', 'Heterozygote', 'Homozygote 1']):
    sample1._data_predictions[sample1._data_predictions.pred == pred].val.hist(bins=np.arange(0,1,0.05),
                                                                                   alpha=0.8, label=name)
plt.legend()
```

![hmm_distribution.png](README_files/hmm_distribution.png)



```python
sample1.plot_hmm(val_column='pred')
```


![coda_res.png](README_files/coda_res.png)



```python
sample1.crossovers
```

|       | chrm   |    start |      end |    val |   pred | probs               |   smooth |   updated |   change |   abs_pos |   pos_ler |   dist | possible_dup   |   binsize |   binsize_percentile | context    |
|------:|:-------|---------:|---------:|-------:|-------:|:--------------------|---------:|----------:|---------:|----------:|----------:|-------:|:---------------|----------:|---------------------:|:-----------|
|  2298 | col1   | 15079201 | 15086201 | 0.0175 |      0 | [0.983 0.017 0.   ] |       -1 |         0 |        1 |    0.4958 |  13884201 | 0.0234 | False          |      7000 |                82.91 | centromere |
|  2326 | col1   | 15488401 | 15491401 | 0.035  |      1 | [0.001 0.999 0.   ] |       -1 |         1 |        0 |    0.5091 |  14099001 | 0.0294 | False          |      3000 |                47.87 | centromere |
|  5071 | col1   | 29377401 | 29381001 | 0      |      0 | [1. 0. 0.]          |       -1 |         0 |        1 |    0.9656 |  28350001 | 0.0009 | False          |      3600 |                56.66 | nan        |
|  9632 | col3   |  5068201 |  5077801 | 0      |      0 | [1. 0. 0.]          |       -1 |         0 |        1 |    0.2165 |   5035001 | 0.0064 | False          |      9600 |                91.05 | nan        |
| 12763 | col3   | 18397201 | 18411801 | 0.2187 |      1 | [0.    0.961 0.039] |       -1 |         1 |        2 |    0.7849 |  17583601 | 0.0065 | False          |     14600 |                96.65 | nan        |
| 13317 | col3   | 22814601 | 22819001 | 0.7143 |      2 | [0. 0. 1.]          |       -1 |         2 |        1 |    0.9728 |  22034201 | 0.0027 | False          |      4400 |                66.69 | nan        |
| 16872 | col4   | 18031201 | 18036201 | 0.2041 |      1 | [0. 1. 0.]          |       -1 |         1 |        0 |    0.9705 |  17426801 | 0.0013 | False          |      5000 |                71.04 | nan        |
| 19813 | col5   | 14959801 | 14964601 | 0.2945 |      1 | [0.    0.934 0.066] |       -1 |         1 |        2 |    0.5547 |  14323801 | 0.0116 | False          |      4800 |                68.86 | nan        |