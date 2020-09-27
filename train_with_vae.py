import argparse
import base64

import IPython
import numpy as np
import torch
from torch import optim

from agent_vae import VAEAgent
from env_wrapper import Env
from utils import DrawLine
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--vis',  default=0, action='store_true', help='use visdom')
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()


def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename, 'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="480" height="480" controls>
        <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())
    return IPython.display.HTML(tag)

latent_size = 128
if __name__ == "__main__":
    agent = VAEAgent(args, latent_size)
    env = Env(args)
    if args.vis:
        draw_reward = DrawLine(env="car", title="PPO", xlabel="Episode", ylabel="Moving averaged episode reward")

    training_records = []
    total_score = np.array([])
    data_for_plot=np.array([])
    state = env.reset()
    for i_ep in range(5000):
        score = 0

        for t in range(1000):
            action, a_logp = agent.select_action(state)
            state_, reward, done, _ = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                # env.render()
                pass
            if agent.store_transition(state, action, a_logp, reward, state_):
                print('updating')
                agent.update()
            score += reward
            state = state_
            if done:
                state = env.reset()
                break
        total_score = np.append(total_score, score)

        if i_ep % args.log_interval == 0 and i_ep != 0:
            if args.vis:
                draw_reward(xdata=i_ep, ydata=np.mean(total_score[-args.log_interval:]))
            tmp=np.mean(total_score[-args.log_interval:])
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, tmp))
            data_for_plot=np.append(data_for_plot,tmp)
            agent.save_param()
            '''
            video_filename = 'sac_minitaur_' + str(i_ep) + '.mp4'
            test_env = Env(video_file=video_filename)
            for _ in range(1):
                done = False
                state = test_env.reset()
                while not done:
                    action, a_logp = agent.select_action(state)
                    state_, reward, done, _  = test_env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            test_env.close_writer()
            '''
    plt.plot(data_for_plot)
    plt.show()
        #if running_score > env.reward_threshold:
         #   print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
          #  break
