#!/bin/bash

# -------Set-Constants--------------------------

# Concatenated genome reference
# Remember to run bwa index on the index beforehand
REF="collab/references/mergedAB.fasta"

# Informative bin template created by filter_genomes.py
TEMPLATE="collab/references/mergedAB.cov-200.bedgraph"

# -------Alignment and Filtering----------------

# Input: First of paired illumina reads (.fastq)
# Output: Sorted and filtered bedgraph with coverage data

# Only relevant to HPC module supported systems
module load bwa-gnu         # http://bio-bwa.sourceforge.net/bwa.shtml#3
module load samtools        # http://www.htslib.org/doc/samtools.1.html
module load jdk             # Java, required to run Picard
module load picard          # https://software.broadinstitute.org/gatk/documentation/tooldocs/4.0.4.0/picard_sam_markduplicates_MarkDuplicates.php
module load bedtools-gnu    # https://bedtools.readthedocs.io/en/latest/content/tools/coverage.html

# $1 is command line arg fastq, get R2 file by replacing pair symbol
R2="${1/R1/R2}"
# Get sample name by stripping pair symbol and file extension
sample="${1/R1/}"
fname="${sample%%.*}"

# Perform alignment, pipe directly to samtools in order to reduce storage reqs
# Convert to BAM, filter by Q> 30 and unmapped / non-primary / PCR / alignments
# Sort and output to fname
bwa mem -t 5 $ref $1 $R2 | samtools view -Sbh -F 1284 -q 30 - | samtools sort - -o "$fname".q30.sorted.bam

# Remove duplicates
picard MarkDuplicates I="$fname".q30.sorted.bam O="$fname".dedup.q30.sorted.bam M="$fname".dup_metrics.txt REMOVE_DUPLICATES=true

# Extract template extension
extname="${TEMPLATE##*.}"
# Get coverage count according to template
bedtools coverage -counts -sorted -a "$TEMPLATE" -b "$fname".dedup.q30.sorted.bam > "$fname".dedup.q30.sorted."$extname"


# ----------------------------

# Run coda

module load anaconda
source "$CONDA_PREFIX"/bin/activate xoe
python coda.py "$fname".dedup.cov-200-200.bedgraph predict
