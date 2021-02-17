import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import argparse
#data_path = "/home/aliarab/scratch/sgd2/sim_data/data_params/"
#result_path = "/home/aliarab/scratch/sgd2/result/data_params/"

#output_root= "/home/aliarab/scratch/sgd/post_process/data_params/"

import importlib
import sgd
importlib.reload(sgd)
from sgd import Conjuction
from sgd import EquitySelector
import glob
import pickle
import numpy as np
import pandas as pd

def getResultsFiles(result_root_path, params_list):
    print(params_list)
    dirs= []
    for subject_folder in glob.glob(os.path.join(result_root_path, "*", "*", "*")):
        tokens = subject_folder.split("/")
        param_name = tokens[-3]
        trial_number= int(tokens[-1])
        
        print(param_name)
        if param_name in params_list and trial_number in [8, 17, 11, 22, 24]:
            dirs.append(subject_folder)
    return dirs

def getDataFiles(data_path_root, result_files):
    dirs = []
    for directory in result_files:
        tokens = directory.split("/")
        target_path = os.path.join(data_path_root, tokens[-3], tokens[-2], tokens[-1])
        dirs.append(target_path)
    return dirs

def getFinalResultPaths(output_root, data_paths):
    dirs = []
    for directory in data_paths:
        tokens = directory.split("/")
        target_path = os.path.join(output_root, tokens[-3], tokens[-2], tokens[-1])
        dirs.append(target_path)
    return dirs



def main(data_path, result_path, output_root, params_list=""):
    results_files_path = getResultsFiles(result_path, params_list)
    print (results_files_path)
    data_files_path = getDataFiles(data_path, results_files_path)
    final_results_path = getFinalResultPaths(output_root, results_files_path)
    from sgd import Conjuction
    print(results_files_path)
    results_files_path.reverse()
    for i in range(len(results_files_path)):
        print(results_files_path[i])
        processCase(data_files_path[i], results_files_path[i], final_results_path[i])


def processCase(data_files_path, results_files_path, final_results_path):
    [X, Y, V, W, true_causes] = sgd.readData_sim(data_files_path, return_true_causes=True)
    [selectors, original_features, sg_to_index] = sgd.createSelectors(X, [])
    
    with open(os.path.join(results_files_path,"res.pkl"), "rb") as f:
        beam = pickle.load(f)
    found_sgs=[]
    print (beam)
    beam_selectors = set()
    for pair in beam:
        found_sgs.append(pair[1])
        for sel in pair[1].selectors:
            beam_selectors.add(sel)

    refined_X = []
    column_names= []
    for sel in beam_selectors:
        column_names.append(str(sel))
        sel_id = sg_to_index[sel] 
        refined_X.append(list(sel.covers(X)))
    refined_X.append(Y["outcome"])
    column_names.append("outcome")
    refined_X = np.array(refined_X)
    print(refined_X.shape)
    saveResults(refined_X.T, column_names, true_causes, final_results_path, beam)

def saveResults(X, column_names, true_causes, result_path, beam):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    df = pd.DataFrame(X, columns=column_names)
    df.to_csv(os.path.join(result_path,"D.csv"), index=None)
    true_cause_path = os.path.join(result_path, "true_causes.txt")
    with open(true_cause_path, "w") as f:   
        for sg in true_causes:
            f.write(str(sg))
            f.write("; ")   
                
    with open(os.path.join(result_path,"beam.txt"), "w") as f:        
        for score, sg in beam:
            f.write(str(sg))
            f.write("; ") 

         

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="data path")
    parser.add_argument("result_path", help="result path")
    parser.add_argument("output_root", help="output path")
    parser.add_argument("--params_list", help="list of parameters, separated by underscore")

    args = parser.parse_args()
    params_list = set(args.params_list.split("_"))
    main(args.data_path, args.result_path, args.output_root, params_list = params_list)
