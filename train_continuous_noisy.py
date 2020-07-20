import tensorflow as tf
import gym
import time
import os, sys

import deepq
import datetime

flags = tf.app.flags
flags.DEFINE_integer('start', 0, "start from index")
flags.DEFINE_integer('end', 6, "to index")
flags.DEFINE_string('struct', '0-1-2', "structure")
flags.DEFINE_string('env', 'Hopper', "env")

flags = flags.FLAGS

start_i = int(flags.start)
end_i = int(flags.end)
env = flags.env

if env == 'Hopper':
    env_name = 'Hopper-v2'
    total_num_episodes = 4000
elif env == 'Walker':
    env_name = 'Walker2d-v2'
    total_num_episodes = 6000
elif env == 'Cheetah':
    env_name = 'HalfCheetah-v2'
    total_num_episodes = 6000
elif env == 'Human':
    env_name = 'Humanoid-v1'
    total_num_episodes = 15000

struct = []
st = flags.struct
sts = st.split('_')

for s in sts:
    t = []
    s = s.split('-')
    for e in s:
        t.append(int(e))
    struct.append(t)

print(start_i)
print(end_i)
print(env_name)
print(struct)

TARGET_UPDATE_FREQUENCY = 1000
LEARNING_RATE =1e-4

def main(log_prefix, log_index = 0, struct = None):
    num_actions_pad = 17 # numb discrete sub-actions per action dimension

    env = gym.make(env_name)

    time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S') 

    model = deepq.models.mlp_noisy_branching(
        hiddens_common=[512, 256],
        hiddens_actions=[128],
        num_action_branches=env.action_space.shape[0],
        struct = struct,
    )

    act = deepq.learn_continuous_tasks_noisy(
        env,
        q_func=model,
        env_name=env_name,
        dir_path=os.path.abspath(path_logs),
        time_stamp=time_stamp,
        total_num_episodes=total_num_episodes,
        lr=LEARNING_RATE,
        gamma=0.99,
        #gamma=0.7,
        batch_size=64,
        buffer_size=int(1e6),
        prioritized_replay=False,
        num_actions_pad=num_actions_pad,
        grad_norm_clipping=None,
        learning_starts=1000,
        target_network_update_freq=TARGET_UPDATE_FREQUENCY,
        train_freq=1,
        initial_std=0.2,
        final_std=0.2,
        epsilon_greedy=True,
        timesteps_std=1e6,
        eval_freq=50,
        n_eval_episodes=30,
        eval_std=0.0,
        num_cpu=16,
        print_freq=10,
        callback=None,
        log_index = log_index,
        log_prefix = log_prefix,
        model_file = 'trained_models/'+str(log_index)+log_prefix
    )

    act.save('trained_models/'+str(log_index)+log_prefix)

if __name__ == '__main__':


    for ind in range(start_i,end_i):
        with tf.Graph().as_default():
            prefix = "_noisy_"+env_name+st

            print(str(datetime.datetime.now().strftime("%I_%M_%B_%d_%Y")))
            print(str(ind)+prefix)

            main(prefix,ind, struct=struct)
