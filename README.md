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
To run the search phase on the generated datasets, run the following python code:

**python sgd.py**

**python post_process_data_params.py**

The above two commands processess the synthetic dataset, and after post-processing the results, produces a text file beam.txt which contains the survived elements in the beam.


To run the assembly and evaluation phases, run the following command:


**matlab -nodisplay -nosplash -nodesktop -r "run('AE.m');exit;"**

