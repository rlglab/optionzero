import pandas as pd
import argparse
import subprocess
import os
def execute_linux_cmd(cmd: str):
    ps = subprocess.Popen(cmd, executable='/bin/bash', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = ps.communicate()[0]
    ps.terminate()
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir', dest='in_dir', type=str, help='dir to analysis option')
    args = parser.parse_args()
    if not args.in_dir:
        parser.print_help()
        exit(1)
    in_dir=args.in_dir
    in_option_dir = os.path.join(in_dir, 'option_analysis')

    # option in games
    result={}
    output = execute_linux_cmd(f'cat {in_option_dir}/stats/moves-stats.txt | grep \'%o\'').decode().split('\n')[:-1]
    avg_len=0
    for i, o in enumerate(output):
        ratio=float(o.split('=')[1].replace('%', 'e-2')) if o[-1] != '=' else 0
        if i == 0:
            result[f'% a'] = [round(1 - ratio, 4)]
            result[f'% o'] = [round(ratio, 4)]
            continue
        avg_len += ratio * i
        if i > 1:
            result[f'% {i}'] = [ratio]
    result['avg. l'] = [round(avg_len, 2)]
    output = execute_linux_cmd(f'cat {in_option_dir}/stats/repeated-options.txt | grep \'%repeat\'').decode().split('\n')[0]
    ratio=float(output.split('=')[1].replace('%', 'e-2')) if output[-1] != '=' else 1
    result['% Rpt.'] = [round(ratio, 4)]
    result['% NRpt.'] = [round(1 - ratio, 4)]
    result_df = pd.DataFrame(result)
    result_df.to_csv(os.path.join(in_option_dir, f'option_in_games.csv'), index=False)

    # option in trees
    result={}
    output = execute_linux_cmd(f'cat {in_option_dir}/stats/options-in-tree.txt | grep \'*\'').decode().split('\n')[0].split('\t')
    num = float(output[1])
    num_opt = float(output[4]) if len(output) > 4 else 0
    num_opt_sim = float(output[5]) if len(output) > 5 else 0

    output = execute_linux_cmd(f'cat {in_dir}/$(basename \'{in_dir}\').cfg | grep -oE \'actor_num_simulation=[0-9]*\'')
    actor_num_simulation = int(output.decode().split('actor_num_simulation=')[1])
    result['% in Tree'] = [round(num_opt / num if num > 0 else 0, 4)]
    result['% in Sim.'] = [round(num_opt_sim / (num * actor_num_simulation) if num > 0 else 0, 4)]  
    output = execute_linux_cmd(f'cat {in_option_dir}/stats/options-depth-percentile.txt').decode().replace('\n','').split(' ')[4:]
    result['Avg. tree depth'] = [float(output[0])]
    result['Median tree depth'] = [float(output[2])]
    result['Max tree depth'] = [float(output[3])]
    result_df = pd.DataFrame(result)
    result_df.to_csv(os.path.join(in_option_dir, f'option_in_trees.csv'), index=False)
