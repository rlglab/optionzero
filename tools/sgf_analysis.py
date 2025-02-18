#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import subprocess
import argparse
import matplotlib.pyplot as plt
from matplotlib import colormaps
from cycler import cycler
import matplotlib.ticker as ticker
from time import time
import sys
# plt.rc('axes', prop_cycle=(cycler(color=colormaps['tab20'].colors)))


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs, flush=True)


def execute_linux_cmd(cmd: str):
    ps = subprocess.Popen(cmd, executable='/bin/bash', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = ps.communicate()[0]
    ps.terminate()
    return output


class SgfAnalysis:
    def __init__(self, in_dirs: list, fout_name: str = 'sgf_record', games_per_iter: int = 100, out_format: str = 'png',
                 action_size: int = 18, plot_x_by_iter: bool = False, plot_range=[-np.inf, np.inf], program_quiet: bool = False, compare: bool = False):
        self.in_dirs = in_dirs
        self.fout_name = fout_name
        self.games_per_iter = games_per_iter
        self.out_format = out_format
        self.action_size = action_size
        self.plot_x_by_iter = plot_x_by_iter
        self.plot_range = plot_range
        self.compare = compare
        self.statistics = []
        self.count_statistics = []
        self.ratio_statistics = []
        self.round_to_int = False
        self.is_abs_max_over_1k = False
        self.program_quiet = program_quiet

    def eprint(self, *args, **kwargs):
        if not self.program_quiet:
            print(*args, file=sys.stderr, **kwargs, flush=True)

    def format_y_axis_labels(self, value, pos):
        if self.is_abs_max_over_1k:
            if self.round_to_int:
                return f'{value/1000:.0f}k'
            else:
                return f'{value/1000:.1f}k'
        else:
            return value

    def plot_ratio_curve(self):
        plt.rcParams.update({'font.size': 30})
        _, ax = plt.subplots(figsize=(25, 20))
        title = os.path.basename(os.path.dirname(self.statistics[0][0] + '/'))
        title = title[:100]+'\n'+title[100:]
        ax.set_title(title)
        ax.set_xlabel('iterations')
        ax.set_ylabel('Env Act Action Ratio')
        ax.set_prop_cycle(cycler(color=colormaps['tab20'].colors))
        fig_name = os.path.join(self.statistics[0][0], 'sgf_record', f'ratio.{self.out_format}') if not self.compare else f'ratio.{self.out_format}'
        for in_dir, ratio_records in self.ratio_statistics:
            base_dir = os.path.basename(os.path.dirname(in_dir + '/'))
            label = base_dir[:len(base_dir) // 2] + '\n' + base_dir[len(base_dir) // 2:] if len(base_dir) > 100 else base_dir
            for key, value in ratio_records.items():
                label = str(key) if key < self.action_size else 'OP'
                ax.plot(range(1, len(value) + 1), value, label=label, linewidth=4)
        ax.set_ylim(ymax=1.1)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncols=7)
        plt.tight_layout()
        plt.savefig(f'{fig_name}')
        plt.close('all')

    def plot_count_curve(self):
        plt.rcParams.update({'font.size': 30})
        _, ax = plt.subplots(figsize=(25, 20))
        ax.set_title(f'Avg. Option MCTS Count')
        ax.set_xlabel('iterations')
        ax.set_ylabel('Average Option Count')
        fig_name = os.path.join(self.statistics[0][0], 'sgf_record', f'count.{self.out_format}') if not self.compare else f'count.{self.out_format}'
        for in_dir, count_records in self.count_statistics:
            base_dir = os.path.basename(os.path.dirname(in_dir + '/'))
            label = base_dir[:100] + '\n' + base_dir[100:]
            ax.plot(range(1, len(count_records) + 1), count_records, label=label, linewidth=4)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1)
        plt.tight_layout()
        plt.savefig(f'{fig_name}')
        plt.close('all')

    def plot_curve(self):
        plt.rcParams.update({'font.size': 30})
        _, ax = plt.subplots(figsize=(25, 20))
        ax.set_xlabel('iterations') if self.plot_x_by_iter else ax.set_xlabel('nn steps')
        ax.set_ylabel('Returns')
        ax.set_title(f'Returns of each iteration {self.games_per_iter} games')
        max_step_num = -np.inf
        min_step_num = np.inf
        fig_name = os.path.join(self.statistics[0][0], 'sgf_record', f'{self.fout_name}.{self.out_format}') if not self.compare else f'{self.fout_name}.{self.out_format}'
        for in_dir, result_df in self.statistics:
            output = execute_linux_cmd(f'cat {in_dir}/$(basename \'{in_dir}\').cfg | grep -oE \'learner_training_step=[0-9]*\'')
            learner_training_step = int(output.decode().split('learner_training_step=')[1])
            iter_nums = result_df['iter'].to_numpy()
            step_nums = iter_nums if self.plot_x_by_iter else iter_nums * learner_training_step
            all_means = result_df['Mean'].to_numpy()[np.where((step_nums >= self.plot_range[0]) & (step_nums <= self.plot_range[1]))]
            all_stds = result_df['Std'].to_numpy()[np.where((step_nums >= self.plot_range[0]) & (step_nums <= self.plot_range[1]))]
            step_nums = step_nums[np.where((step_nums >= self.plot_range[0]) & (step_nums <= self.plot_range[1]))]
            base_dir = os.path.basename(os.path.dirname(in_dir + '/'))
            label = base_dir[:100] + '\n' + base_dir[100:]
            line, = ax.plot(step_nums, all_means, label=label, linewidth=4)
            ax.fill_between(step_nums, all_means - all_stds, all_means + all_stds, color=line.get_color(), alpha=0.1)
            max_step_num = max(max_step_num, step_nums[-1]) if len(step_nums) > 0 else max_step_num
            min_step_num = min(min_step_num, step_nums[0]) if len(step_nums) > 0 else min_step_num
        if not np.isinf(max_step_num) and min_step_num != max_step_num:
            ax.set_xlim([min_step_num, max_step_num])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1)
        yticks = ax.get_yticks()
        self.is_abs_max_over_1k = yticks[0] <= -1000 or yticks[-1] >= 1000
        self.round_to_int = yticks[-1] % 1000 == 0 and yticks[-2] % 1000 == 0
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.format_y_axis_labels))
        ax.grid()
        plt.tight_layout()
        plt.savefig(f'{fig_name}')
        plt.close('all')
        self.eprint(fig_name)

    def load_npz(self, file, refresh = True):
        result = {'iter': [],
                  'Min': [],
                  'Max': [],
                  'Median': [],
                  'Mean': [],
                  'Std': [],
                  'Avg. Len.': [],
                  'Total': [], }
        re_records = []
        dlen_records = []
        count_records = []
        ratio_records = {}
        for i in range(self.action_size + 1):
            ratio_records[i] = []
        last_stored_iter = -1
        last_game_id_of_each_iter = [-1]
        if os.path.isfile(file):
            npz_file = np.load(file, allow_pickle=True)
            last_stored_iter = npz_file['last_stored_iter']
            last_game_id_of_each_iter = npz_file['last_game_id_of_each_iter'].tolist()
            # records until last second iter
            re_records = npz_file['re_records'].tolist()
            dlen_records = npz_file['dlen_records'].tolist()
            count_records = npz_file['count_records'].tolist()
            # if 'ratio_records' in npz_file:
            ratio_records = npz_file['ratio_records'].item()
            result = npz_file['result'].item()
            if refresh:
                last_game_id_of_each_iter = last_game_id_of_each_iter[:-1]
                re_records = re_records[:last_game_id_of_each_iter[-1] + 1]
                dlen_records = dlen_records[:last_game_id_of_each_iter[-1] + 1]
                count_records = count_records[:-1]
                for i in range(self.action_size + 1):
                    ratio_records[i] = ratio_records[i][:-1]
                result = {'iter': result['iter'][:-1],
                      'Min': result['Min'][:-1],
                      'Max': result['Max'][:-1],
                      'Median': result['Median'][:-1],
                      'Mean': result['Mean'][:-1],
                      'Std': result['Std'][:-1],
                      'Avg. Len.': result['Avg. Len.'][:-1],
                      'Total': result['Total'][:-1],
                    }
        return result, re_records, dlen_records, count_records, ratio_records, last_stored_iter, last_game_id_of_each_iter

    def analysis(self, refresh=True):
        for in_dir in self.in_dirs:
            if not os.path.isdir(os.path.join(in_dir, 'sgf_record')):
                os.mkdir(os.path.join(in_dir, 'sgf_record'))
            result, re_records, dlen_records, count_records, ratio_records, last_stored_iter, last_game_id_of_each_iter = self.load_npz(os.path.join(in_dir, 'sgf_record', 'sgf_record.npz'), refresh)
            start = time()
            self.eprint(f'directory: {in_dir}')
            if refresh:
                sgf_files = execute_linux_cmd(f'''ls {os.path.join(in_dir,'sgf','*.sgf')} | sort -V''').decode().split('\n')[max(last_stored_iter - 1, 0):-1]
            else:
                sgf_files = execute_linux_cmd(f'''ls {os.path.join(in_dir,'sgf','*.sgf')} | sort -V''').decode().split('\n')[max(last_stored_iter, 0):-1]
            if len(sgf_files) == 0 and last_stored_iter == -1:
                return
            for sgf_file in sgf_files:
                output = execute_linux_cmd(f'cat {sgf_file} | grep \'#$\' | grep -oE \'RE\\[([-0-9.])*\\]|DLEN\\[[0-9-]*\\]\'').decode().split('\n')
                re_records += [float(line.split("RE[")[1].split("]")[0]) for line in output if 'RE[' in line]
                dlen_records += [float(line.split('DLEN[')[1].split('-')[1].split(']')[0]) + 1 for line in output if 'DLEN[' in line]
                count_output = execute_linux_cmd(f'cat {sgf_file} | grep -oE \'\\]C\\[[0-9]+\' | sed \'s/\\]C\\[//g\'').decode().split('\n')[:-1]
                cnt_sum = 0
                for cnt in count_output:
                    cnt_sum += int(cnt)
                total_len = len(count_output) if len(count_output) > 0 else 1
                count_records.append(cnt_sum / total_len)
                total_count = 0
                action_counts = []
                for i in range(self.action_size + 1):
                    action_count = int(execute_linux_cmd(f'cat {sgf_file} | grep -oE \';B\\[{i}\\]\' | wc -l').decode())
                    action_counts.append(action_count)
                    total_count+=action_count
                if total_count > 0:
                    action_counts = [action_count/total_count for action_count in action_counts]
                for i in range(self.action_size + 1):
                    ratio_records[i].append(action_counts[i])
                last_stored_iter = int(os.path.basename(sgf_file).split('.sgf')[0])
                while len(last_game_id_of_each_iter) < last_stored_iter + 1:
                    last_game_id_of_each_iter.append(len(re_records) - 1)
                self.eprint(f'file: {os.path.basename(sgf_file)}, time: {time()-start}')
                assert len(dlen_records) == len(re_records)
            start_iter = result['iter'][-1] + 1 if len(result['iter']) > 0 else 1
            for cur_iter in range(start_iter, len(last_game_id_of_each_iter)):
                last_game_id = last_game_id_of_each_iter[cur_iter]
                if last_game_id == -1:
                    continue
                first_game_id = max(last_game_id + 1 - self.games_per_iter, 0)
                game_lengths = dlen_records[first_game_id:last_game_id + 1]
                returns = re_records[first_game_id:last_game_id + 1]
                result['iter'].append(cur_iter)
                result['Min'].append(np.min(returns).round(2))
                result['Max'].append(np.max(returns).round(2))
                result['Median'].append(np.median(returns).round(2))
                result['Mean'].append(np.mean(returns).round(2))
                result['Std'].append(np.std(returns).round(2))
                result['Avg. Len.'].append(np.array(game_lengths).mean().round(2))
                result['Total'].append(len(returns))
            result_df = pd.DataFrame(result)
            result_df.to_csv(os.path.join(in_dir, 'sgf_record', f'{self.fout_name}.csv'), index=False)
            self.statistics.append([in_dir, result_df])
            self.count_statistics.append([in_dir, count_records])
            self.ratio_statistics.append([in_dir, ratio_records])
            if refresh and last_stored_iter != -1:
                np.savez(os.path.join(in_dir, 'sgf_record', 'sgf_record.npz'), result=result, last_stored_iter=last_stored_iter, re_records=re_records,
                         dlen_records=dlen_records, count_records=count_records, ratio_records=ratio_records, last_game_id_of_each_iter=last_game_id_of_each_iter)
        if not self.compare:
            self.eprint(result_df.to_string(index=False))
            self.plot_ratio_curve()
        self.plot_curve()
        self.plot_count_curve()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir', dest='in_dir', type=str, help='dir to analysis sgf')
    parser.add_argument('-out_dir', dest='fout_name', type=str, default='sgf_record', help='output flie')
    parser.add_argument('-n', dest='games_per_iter', type=int, default=100, help='iteration')
    parser.add_argument('-c', dest='compare', action='store_true', help='compare many version of same game, allow regular expressions. ex: -c \".*gopher.*PER.*|a|b\"')
    parser.add_argument('-xmin', dest='plot_min', type=int, default=-np.inf, help='minimum value of x-axis')
    parser.add_argument('-xmax', dest='plot_max', type=int, default=np.inf, help='maximum value of x-axis')
    parser.add_argument('-out_format', dest='out_format', type=str, default='png', help='out put file format')
    parser.add_argument('-action_size', dest='action_size', type=int, default=18, help='number of primitive actions in the environment (default is 18 for atari games)')
    parser.add_argument('--x_by_iter', dest='plot_x_by_iter', action='store_true', help='plot x-axis by iteration (default is nn-step)')
    parser.add_argument('--save', dest='save', action='store_true', help='save record')
    args = parser.parse_args()
    plot_range = [args.plot_min, args.plot_max]
    if not args.in_dir:
        parser.print_help()
        exit(1)
    in_dirs = execute_linux_cmd(
        f'ls -ld {{{args.in_dir},}} | grep ^d | awk \'{{print $NF}}\'').decode().split('\n')[:-1]
    if len(in_dirs) == 0:
        eprint(f'No folder matches \"{args.in_dir}\"!')
        exit(1)
    if args.compare:
        sgf_analysis = SgfAnalysis(in_dirs, args.fout_name, args.games_per_iter, args.out_format, args.action_size, args.plot_x_by_iter, [args.plot_min, args.plot_max], compare=args.compare)
        sgf_analysis.analysis(refresh=args.save)
    else:
        for in_dir in in_dirs:
            sgf_analysis = SgfAnalysis([in_dir], args.fout_name, args.games_per_iter, args.out_format, args.action_size, args.plot_x_by_iter, [args.plot_min, args.plot_max])
            sgf_analysis.analysis(refresh=args.save)
