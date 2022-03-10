The source code for Hegel, a method to discover causal factors in high dimensional data

To function to generate the synthetic datasets, is located at scripts/synthesize.m. The functijon will produces a compound causal discovery (CCD) dataset in csv format, including features, confounders, output, and indices of causal features, based on the input parameters:
 

m = 100;% number of features

n = 500;% number of samples

l1 = 0;	% number of singular causes

l2 = 1;	% number of pair causes

l3 = 0; % number of triplet causes

p = 1/4;% non-sparsity of signal

z = 3/4;% non-sparsity of necessary confounders

q = 0.05;% rate of noise

sp = 3/4;% distribution mean of 1D Prior Score

f = 28; % number of functions used

output_dir = Path for the generated dataset to be saved at.

----------------------

To generate the synthetic datasets with the same configuration as the ones used in the manuscript, call synthesize_wrapper.m script:

**matlab -nodisplay -nosplash -nodesktop -r "run('scripts/synthesize_wrapper.m');exit;"**


It is a wrapper script that makes call to the synthesize function included in synthesize.m

----------------------
To run the algorithm on a dataset, run the following python command:

**python main.py --input_path "path/to/dataset/" --u 50 --beam_width 10 --weight 2**


Running the above command produces two output files, compounds.txt and columns_names.csv, in the current directory, where copounds.txt contains the list of causal compounds, with each line identifying one compound. The corresponding selectors for IDs in this file can be found in columns_names.csv.



