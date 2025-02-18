#!/usr/bin/env python3

import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import argparse
import os
import time
from multiprocessing import Process
import matplotlib.image as mpimg
import warnings
import io
import matplotlib.patches as patches
import matplotlib.colors as mcolors
ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}
ACTION_MEANING_short = {
    0: "N",
    1: "F",
    2: "U",
    3: "R",
    4: "L",
    5: "D",
    6: "UR",
    7: "UL",
    8: "DR",
    9: "DL",
    10: "UF",
    11: "RF",
    12: "LF",
    13: "DF",
    14: "URF",
    15: "ULF",
    16: "DRF",
    17: "DLF",
}
start_color = '#F9F900'
end_color = '#FF0000'
start_rgba = mcolors.to_rgba(start_color)
end_rgba = mcolors.to_rgba(end_color)

start_rgb = start_rgba[:3]  # 取出RGB部分
end_rgb = end_rgba[:3]

num_colors = 10

colors = [mcolors.rgb2hex(start_rgb + (i * (np.array(end_rgb) - np.array(start_rgb)) / (num_colors - 1))) for i in range(num_colors)]
use_option=1==1

class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if self.is_empty():
            return None
        return self.items.pop(0)

    def peek(self):
        if self.is_empty():
            return None
        return self.items[-1]

    def size(self):
        return len(self.items)

    def average(self):
        if self.is_empty():
            return None
        self.keep_10()
        total = sum(self.items)
        return total / 10

    def keep_10(self):
        if self.is_empty():
            return None
        while len(self.items) > 10:
            self.items.pop(0)


class AtariEnv:
    def __init__(self, game_name, gym_game_name, video_file_name, seed=None):
        self.video_file_name = video_file_name
        self.game_name = game_name
        self.env = gym.make(gym_game_name, frameskip=4, repeat_action_probability=0.25, full_action_space=True, render_mode="rgb_array")
        self.reset(seed)
        self.stack = Stack()
        self.action_counter = 0

    def reset(self, seed):
        self.seed = seed
        self.done = False
        self.total_reward = 0
        self.env.ale.setInt("random_seed", seed)
        self.env.ale.loadROM(f"/opt/atari57/{self.game_name}.bin")
        self.observation, self.info = self.env.reset()
        self.img = plt.imshow(self.env.render())
        plt.axis('off')
        plt.tight_layout()
        fig = self.env.render()
        self.img.set_data(fig)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        # Read the saved image into a NumPy array
        fig = mpimg.imread(buffer)
        self.frames = [fig]
        self.frames_buffer = []
        self.combined_frames = []
        self.option = []
        self.option_thinking = []
        self.frame_id = -1

    def act(self, action_id, action_ids, store_frame=True):
        self.observation, self.reward, self.done, _, self.info = self.env.step(action_id)
        self.frame_id += 1
        self.option.append(f'{action_id}')
        if store_frame:
            fig = self.env.render()
            self.img.set_data(fig)
            # print(self.option)
            # print(len(self.option))
            if len(self.option) > 1:
                self.stack.push(1)
            else:
                self.stack.push(0)
            mapped_actions = [ACTION_MEANING_short[int(action_id)] for action_id in self.option]
            mapped_actions_2 = [ACTION_MEANING_short[int(action_id)] for action_id in action_ids]
            if use_option:
                text = plt.text(-50, 20, f'{self.frame_id}: ' + '-'.join(mapped_actions), color='red', fontsize=20)
                text2 = plt.text(-50, 40, '-'.join(mapped_actions_2), color='red', fontsize=20) if action_ids is not [] else ''
                color_code = mcolors.to_rgba('#FF00FF')
                rect = patches.Rectangle((0, 200), 160, 10, linewidth=1, edgecolor=colors[max(int(self.stack.average()) * 10 - 1, 0)], facecolor=colors[max(int(self.stack.average() * 10 - 1), 0)])
            self.action_counter += 1
            if use_option:
                plt.gca().add_patch(rect)
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            # buffer.seek(0)
            # buffer_tmp = buffer
            # if len(self.frames) % 1 == 0:
            #     self.frames_buffer.append(mpimg.imread(buffer))
            # save individual frame to png
            # frame_path = os.path.join(f"{self.video_file_name}", f'frame_{len(self.frames)}.png')
            # with open(frame_path, 'wb') as f:
            #     f.write(buffer_tmp.getvalue())
            # Read the saved image into a NumPy array
            # if len(self.frames_buffer) == 6:  # If 6 frames accumulated, save them
            #     # Combine frames horizontally
            #     combined_frame = np.concatenate(self.frames_buffer, axis=1)
            #     frame_path = os.path.join(f"{self.video_file_name}", f'combined_frames_{len(self.combined_frames)}.png')
            #     plt.imsave(frame_path, combined_frame)
            #     self.combined_frames.append(combined_frame)
            #     self.frames_buffer.clear()

            buffer.seek(0)
            fig = mpimg.imread(buffer)
            self.frames.append(fig)
            if use_option:
                text.remove()
                if text2 != '':
                    text2.remove()
                rect.remove()
            self.option.clear()
        self.total_reward += self.reward

    def is_terminal(self):
        return self.done

    def get_eval_score(self):
        return self.total_reward


def save_video(video_file_name, index, record, fps, force):
    # collect frames
    seed = int(record.split("SD[")[1].split("]")[0])
    env_name = record.split("GM[")[1].split("]")[0].replace("atari_", "")
    if not force and os.path.isfile(f'{video_file_name}/{env_name}-{index}.mp4'):
        print(f'*** {video_file_name}/{env_name}-{index}.mp4 exists! Use --force to overwrite it. ***')
        return
    env = AtariEnv(env_name, 'ALE/' + ''.join([w.capitalize() for w in env_name.split('_')]) + '-v5', video_file_name, seed)
    for action in record.split("B[")[1:]:
        action_id = int(action.split("|")[0].split(']')[0])
        if action_id == 18:
            action_ids = action.split('OP1[')[1].split(']')[0].split('-')
            for i, a in enumerate(action_ids):
                a = env.env.get_action_meanings().index(ACTION_MEANING[int(a)])
                if use_option:
                    env.act(a, action_ids, store_frame=(i == len(action_ids) - 1))
                else:
                    env.act(a, action_ids, store_frame=True)
        # mapping action
        else:
            action_ids = action.split('OP1[')[1].split(']')[0].split('-') if 'OP1[' in action else []
            action_id = env.env.get_action_meanings().index(ACTION_MEANING[action_id])
            env.act(action_id, action_ids)

    # check consistency
    if env.get_eval_score() != float(record.split("RE[")[1].split("]")[0]):
        print(f"replay mismatch, score: {env.get_eval_score()}, record_score: {record.split('RE[')[1].split(']')[0]}")

    # save video
    img = plt.imshow(env.frames[0])
    plt.axis('off')
    plt.tight_layout()
    video = FuncAnimation(plt.gcf(), lambda i: img.set_data(env.frames[i]), frames=len(env.frames))
    video.save(f'{video_file_name}/{env_name}-{index}.mp4', writer=matplotlib.animation.FFMpegWriter(fps=fps))


def join_all_processes(all_processes):
    for i in range(len(all_processes)):
        all_processes[i].join()
    all_processes.clear()


def process_datas(video_file_name, source, num_processes, fps, force):
    assert num_processes > 0
    all_processes = []
    for index, record in enumerate(source):
        all_processes.append(Process(target=save_video, args=(video_file_name, index, record, fps, force)))
        all_processes[len(all_processes) - 1].start()
        if len(all_processes) >= num_processes:
            join_all_processes(all_processes)
    if len(all_processes) > 0:
        join_all_processes(all_processes)


def mkdir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_file', dest='fin_name', type=str, help='input flie')
    parser.add_argument('-out_dir', dest='dir_name', type=str, default=time.strftime(
        '%Y-%m-%d %H:%M:%S', time.localtime()), help='output directory (default: current time)')
    parser.add_argument('-c', dest='num_processes', type=int,
                        default=8, help='process number (default: 8)')
    parser.add_argument('-fps', dest='fps', type=int,
                        default=60, help='fps (default: 60)')
    parser.add_argument('--force', action='store_true',
                        dest='force', help='overwrite files')
    args = parser.parse_args()
    mkdir(args.dir_name)
    if args.fin_name:
        if os.path.isfile(args.fin_name):
            with open(args.fin_name, 'r') as fin:
                process_datas(video_file_name=args.dir_name, source=fin.readlines(
                ), num_processes=args.num_processes, fps=args.fps, force=args.force)
        else:
            print(f'\"{args.fin_name}\" does not exist!')
            exit(1)
    else:
        process_datas(video_file_name=args.dir_name, source=sys.stdin,
                      num_processes=args.num_processes, fps=args.fps, force=args.force)
