import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
data_path = "/home/aliarab/scratch/sgd/sim_data/model_params/"
result_path = "/home/aliarab/scratch/sgd/result/model_params/"

output_root= "/home/aliarab/scratch/sgd/post_process/model_params/"
import importlib
import sgd
importlib.reload(sgd)
from sgd import Conjuction
from sgd import EquitySelector
import glob
import pickle
import numpy as np
import pandas as pd

def getResultsFiles(result_root_path):
    dirs= []
    for subject_folder in glob.glob(os.path.join(result_root_path, "*", "*", "*")):
        tokens = subject_folder.split("/")
        param_name = tokens[-3]
        if True or param_name in set(["sp","l2","q", "p", "l1", "n" ,"z"]):
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


def main():
    results_files_path = getResultsFiles(result_path)
    data_files_path = getDataFiles(data_path, results_files_path)
    final_results_path = getFinalResultPaths(output_root, results_files_path)
    from sgd import Conjuction
    print(results_files_path)
    for i in range(len(results_files_path)):
        print(results_files_path[i])
        [X, Y, V, W, true_causes] = sgd.readData_sim(data_files_path[i], return_true_causes=True)
        [selectors, original_features, sg_to_index] = sgd.createSelectors(X, [])
        
        with open(os.path.join(results_files_path[i],"res.pkl"), "rb") as f:
            beam = pickle.load(f)
        found_sgs=[]
        print (beam)
        beam_selectors = set()
        for pair in beam:
            found_sgs.append(pair[1])
            for sel in pair[1].selectors:
                beam_selectors.add(sel)
        #print(found_sgs)
        #found=True
        #for tc in true_causes:
        #    if not(tc in set(found_sgs)):
        #        found=False

        #if found==True:
        #    print ("found")
        #else:
        #    print("failed")


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
        saveResults(refined_X.T, column_names, true_causes, final_results_path[i], beam)

def saveResults(X, column_names, true_causes, result_path, beam):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    df = pd.DataFrame(X, columns=column_names)
    df.to_csv(os.path.join(result_path,"D.csv"), index=None)
    true_cause_path = os.path.join(result_path, "true_causes.txt")
    with open(true_cause_path, "w") as f:   
        for sg in true_causes:
            f.write(str(sg))
            f.write("\n")   
                
    with open(os.path.join(result_path,"beam.txt"), "w") as f:        
        for score, sg in beam:
            f.write(str(sg))
            f.write("\n") 

         

if __name__ == "__main__":
    main()
