import os
# os.environ['CUDA_VISIBLE_DEVICES']=''
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from param import *
from utils import *
from spark_env.env import Environment
from average_reward import *
from compute_baselines import *
from compute_gradients import *
from actor_agent import ActorAgent


def invoke_model(actor_agent, obs, exp):
    # parse observation
    job_dags, source_job, num_source_exec, \
    frontier_nodes, executor_limits, \
    exec_commit, moving_executors, action_map = obs

    if len(frontier_nodes) == 0:
        # no action to take
        return None, num_source_exec

    # invoking the learning model
    node_act, job_act, \
        node_act_probs, job_act_probs, \
        node_inputs, job_inputs, \
        node_valid_mask, job_valid_mask, \
        gcn_mats, gcn_masks, summ_mats, \
        running_dags_mat, dag_summ_backward_map, \
        exec_map, job_dags_changed = \
            actor_agent.invoke_model(obs)

    if sum(node_valid_mask[0, :]) == 0:
        # no node is valid to assign
        return None, num_source_exec

    # node_act should be valid
    assert node_valid_mask[0, node_act[0]] == 1

    # parse node action
    node = action_map[node_act[0]]

    # find job index based on node
    job_idx = job_dags.index(node.job_dag)

    # job_act should be valid
    assert job_valid_mask[0, job_act[0, job_idx] + \
        len(actor_agent.executor_levels) * job_idx] == 1

    # find out the executor limit decision
    if node.job_dag is source_job:
        agent_exec_act = actor_agent.executor_levels[
            job_act[0, job_idx]] - \
            exec_map[node.job_dag] + \
            num_source_exec
    else:
        agent_exec_act = actor_agent.executor_levels[
            job_act[0, job_idx]] - exec_map[node.job_dag]

    # parse job limit action
    use_exec = min(
        node.num_tasks - node.next_task_idx - \
        exec_commit.node_commit[node] - \
        moving_executors.count(node),
        agent_exec_act, num_source_exec)

    # for storing the action vector in experience
    node_act_vec = np.zeros(node_act_probs.shape)
    node_act_vec[0, node_act[0]] = 1

    # for storing job index
    job_act_vec = np.zeros(job_act_probs.shape)
    job_act_vec[0, job_idx, job_act[0, job_idx]] = 1

    # store experience
    exp['node_inputs'].append(node_inputs)
    exp['job_inputs'].append(job_inputs)
    exp['summ_mats'].append(summ_mats)
    exp['running_dag_mat'].append(running_dags_mat)
    exp['node_act_vec'].append(node_act_vec)
    exp['job_act_vec'].append(job_act_vec)
    exp['node_valid_mask'].append(node_valid_mask)
    exp['job_valid_mask'].append(job_valid_mask)
    exp['job_state_change'].append(job_dags_changed)

    if job_dags_changed:
        exp['gcn_mats'].append(gcn_mats)
        exp['gcn_masks'].append(gcn_masks)
        exp['dag_summ_back_mat'].append(dag_summ_backward_map)

    return node, use_exec


# def train_agent(agent_id, param_queue, reward_queue, adv_queue, gradient_queue):
#     # model evaluation seed
#     tf.set_random_seed(agent_id)

#     # set up environment
#     env = Environment()

#     # gpu configuration
#     config = tf.ConfigProto(
#         device_count={'GPU': args.worker_num_gpu},
#         gpu_options=tf.GPUOptions(
#             per_process_gpu_memory_fraction=args.worker_gpu_fraction))

#     sess = tf.Session(config=config)

#     # set up actor agent
#     actor_agent = ActorAgent(
#         sess, args.node_input_dim, args.job_input_dim,
#         args.hid_dims, args.output_dim, args.max_depth,
#         range(1, args.exec_cap + 1))

#     # collect experiences
#     while True:
#         # get parameters from master
#         (actor_params, seed, max_time, entropy_weight) = \
#             param_queue.get()
        
#         # synchronize model
#         actor_agent.set_params(actor_params)

#         # reset environment
#         env.seed(seed)
#         env.reset(max_time=max_time)

#         # set up storage for experience
#         exp = {'node_inputs': [], 'job_inputs': [], \
#                'gcn_mats': [], 'gcn_masks': [], \
#                'summ_mats': [], 'running_dag_mat': [], \
#                'dag_summ_back_mat': [], \
#                'node_act_vec': [], 'job_act_vec': [], \
#                'node_valid_mask': [], 'job_valid_mask': [], \
#                'reward': [], 'wall_time': [],
#                'job_state_change': []}

#         try:
#             # The masking functions (node_valid_mask and
#             # job_valid_mask in actor_agent.py) has some
#             # small chance (once in every few thousand
#             # iterations) to leave some non-zero probability
#             # mass for a masked-out action. This will
#             # trigger the check in "node_act and job_act
#             # should be valid" in actor_agent.py
#             # Whenever this is detected, we throw out the
#             # rollout of that iteration and try again.

#             # run experiment
#             obs = env.observe()
#             done = False

#             # initial time
#             exp['wall_time'].append(env.wall_time.curr_time)

#             while not done:
                
#                 node, use_exec = invoke_model(actor_agent, obs, exp)

#                 obs, reward, done = env.step(node, use_exec)

#                 if node is not None:
#                     # valid action, store reward and time
#                     exp['reward'].append(reward)
#                     exp['wall_time'].append(env.wall_time.curr_time)
#                 elif len(exp['reward']) > 0:
#                     # Note: if we skip the reward when node is None
#                     # (i.e., no available actions), the sneaky
#                     # agent will learn to exhaustively pick all
#                     # nodes in one scheduling round, in order to
#                     # avoid the negative reward
#                     exp['reward'][-1] += reward
#                     exp['wall_time'][-1] = env.wall_time.curr_time

#             # report reward signals to master
#             assert len(exp['node_inputs']) == len(exp['reward'])
#             reward_queue.put(
#                 [exp['reward'], exp['wall_time'],
#                 len(env.finished_job_dags),
#                 np.mean([j.completion_time - j.start_time \
#                          for j in env.finished_job_dags]),
#                 env.wall_time.curr_time >= env.max_time])

#             # get advantage term from master
#             batch_adv = adv_queue.get()

#             if batch_adv is None:
#                 # some other agents panic for the try and the
#                 # main thread throw out the rollout, reset and
#                 # try again now
#                 continue

#             # compute gradients
#             actor_gradient, loss = compute_actor_gradients(
#                 actor_agent, exp, batch_adv, entropy_weight)

#             # report gradient to master
#             gradient_queue.put([actor_gradient, loss])

#         except AssertionError:
#             # ask the main to abort this rollout and
#             # try again
#             reward_queue.put(None)
#             # need to still get from adv_queue to 
#             # prevent blocking
#             adv_queue.get()


def main():
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # create result and model folder
    create_folder_if_not_exists(args.result_folder)
    create_folder_if_not_exists(args.model_folder)

    # gpu configuration
    config = tf.ConfigProto(
        device_count={'GPU': args.master_num_gpu},
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=args.master_gpu_fraction))

    sess = tf.Session(config=config)

    # set up actor agent
    actor_agent = ActorAgent(
        sess, args.node_input_dim, args.job_input_dim,
        args.hid_dims, args.output_dim, args.max_depth,
        range(1, args.exec_cap + 1))

    # store average reward for computing differential rewards
    avg_reward_calculator = AveragePerStepReward(
        args.average_reward_storage_size)

    # initialize entropy parameters
    entropy_weight = args.entropy_weight_init

    # initialize episode reset probability
    reset_prob = args.reset_prob

    ep_finish_time = []

    # ---- start training process ----
    for ep in range(1, args.num_ep+1):

        print('training epoch', ep)

        # synchronize the model parameters for each training agent
        actor_params = actor_agent.get_params()

        # generate max time stochastically based on reset prob
        max_time = generate_coin_flips(reset_prob)

        all_rewards, all_diff_times, all_times, all_cum_reward = [], [], [], []

        env = Environment()

        env.reset(max_time=max_time)

        exp = {'node_inputs': [], 'job_inputs': [], \
               'gcn_mats': [], 'gcn_masks': [], \
               'summ_mats': [], 'running_dag_mat': [], \
               'dag_summ_back_mat': [], \
               'node_act_vec': [], 'job_act_vec': [], \
               'node_valid_mask': [], 'job_valid_mask': [], \
               'reward': [], 'wall_time': [],
               'job_state_change': []}

        obs = env.observe()
        done = False

        exp['wall_time'].append(env.wall_time.curr_time)

        idxs = 0
        while not done:
            node, use_exec = invoke_model(actor_agent, obs, exp)

            if node is None or use_exec is None :
                print(str(idxs) + " 本次不执行调度。")
            else:
                idxs = idxs + 1
                print(
                    f"{idxs} - 本次调度 {node.idx} 号node, num_taks: {node.num_tasks}, next_task_id: {node.next_task_idx} ，分配执行器 {use_exec}个。time-{env.wall_time.curr_time} \n\t\t max-time-{max_time} reset_prob-{reset_prob} remain dags-{len(env.job_dags)}")
            # -------------------
            obs, reward, done = env.step(node, use_exec)
            # -------------------
            
            if node is not None:
                # valid action, store reward and time
                exp['reward'].append(reward)
                exp['wall_time'].append(env.wall_time.curr_time)
            elif len(exp['reward']) > 0:
                # Note: if we skip the reward when node is None
                # (i.e., no available actions), the sneaky
                # agent will learn to exhaustively pick all
                # nodes in one scheduling round, in order to
                # avoid the negative reward
                exp['reward'][-1] += reward
                exp['wall_time'][-1] = env.wall_time.curr_time
            
            if sum(exp['job_state_change']) > 1 or done:
                agent_exp_valid = True
                batch_reward, batch_time = exp["reward"], exp["wall_time"]
                if len(batch_reward) > 0 and len(batch_time) > 0:
                    diff_time = np.asarray(batch_time[1:]) - np.asarray(batch_time[:-1])

                    all_rewards.append(batch_reward)
                    all_diff_times.append(diff_time)
                    all_times.append(batch_time[1:])

                    avg_reward_calculator.add_list_filter_zero(batch_reward, diff_time)
                else:
                    agent_exp_valid = False
                
                if agent_exp_valid:
                    avg_per_step_reward = avg_reward_calculator.get_avg_per_step_reward()
                    for i in range(args.num_agents):
                        if args.diff_reward_enabled:
                            rewards = np.array([r - avg_per_step_reward * t for (r, t) in zip(all_rewards[i], all_diff_times[i])])
                        else:
                            rewards = np.array([r for (r, t) in zip(all_rewards[i], all_diff_times[i])])
                        cum_reward = discount(rewards, args.gamma)

                        all_cum_reward.append(cum_reward)

                    baselines = get_piecewise_linear_fit_baseline(all_cum_reward, all_times)
                    for i in range(args.num_agents):
                        batch_adv = all_cum_reward[i] - baselines[i]
                        batch_adv = np.reshape(batch_adv, [len(batch_adv), 1])

                        # compute gradients
                        if batch_adv is not None:
                            actor_gradient = compute_actor_gradients(
                                actor_agent, exp, batch_adv, entropy_weight)
                            if actor_gradient is not None:
                                print("apply gradient...")
                                actor_agent.apply_gradients(actor_gradient, args.lr)
                                actor_gradient.clear()
                                del actor_gradient
                                exp = {
                                    'node_inputs': [],
                                    'gcn_mats': [],
                                    'gcn_masks': [],
                                    'summ_mats': [],
                                    'running_dag_mat': [],
                                    'dag_summ_back_mat': [],
                                    'node_act_vec': [],
                                    'node_valid_mask': [],
                                    'reward': [],
                                    'wall_time': [],
                                    'job_state_change': []
                                }
                                # initial time
                                exp['wall_time'].append(env.wall_time.curr_time)
                                all_rewards, all_diff_times, all_times, all_cum_reward = [], [], [], []

            if done:
                # print(done)
                print(f"任务排布的虚拟时间: {env.wall_time.curr_time}")
                print(
                    f"done-{done} max_time-{env.max_time} job_dags_num-{len(env.job_dags)}")
                if ep % args.model_save_interval == 0:
                        actor_agent.save_model(args.model_folder + 'model_ep_' + str(ep))

        print(f"epoch {ep} end...")
        entropy_weight = decrease_var(entropy_weight,
                args.entropy_weight_min, args.entropy_weight_decay)

        # decrease reset probability
        reset_prob = decrease_var(reset_prob,
                args.reset_prob_min, args.reset_prob_decay)

        ep_finish_time.append(env.wall_time.curr_time)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(ep_finish_time, label="finish time", marker="o")
        plt.xlabel("epoch")
        plt.savefig(os.path.join(args.model_folder, "finish_time.png"))
        plt.close()

    sess.close()


if __name__ == '__main__':
    main()