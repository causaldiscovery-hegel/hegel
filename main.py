import subprocess


def run_cmd(cmd, working_directory=None):
	if working_directory!= None:
		try:
			output = subprocess.check_output(cmd,shell=True,cwd=working_directory)
			print "output:"+output
		except:
			print "failed:"+cmd
			# pass
	else:
		try:
			output = subprocess.check_output(cmd,shell=True)
			print(output)
		except:
			print "failed:"+cmd
			# pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="input path")
    parser.add_argument("--output_path", help="output path")
    parser.add_argument("--beam_width", help="beam width", default=10, type=int)
    parser.add_argument("--u", help="number of iterations", default=100, type=int)
    parser.add_argument("--weight", help="weight", default=2, type=float)
    parser.add_argument("--params_list", help="list of parameters, separated by underscore")

    args = parser.parse_args()
    sgd.main_sgd(args.input_path, args.output_path, args.u, args.beam_width, args.weight)
    sgd.post_process_result(args.input_path, args.output_path)
    cmd = """matlab -nodisplay -nosplash -nodesktop -r "addpath('path/to/Causal_Explorer_p_files/', '/path/to/Causal_Explorer_root') ;run('AE.m');exit;"""
    run_cmd(cmd)
