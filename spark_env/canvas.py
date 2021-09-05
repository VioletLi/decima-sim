from typing import cast
from spark_env.task import Task
from param import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt


def visualize_executor_usage(job_dags, file_path):
    exp_completion_time = int(np.ceil(np.max([
        j.completion_time for j in job_dags])))

    job_durations = \
        [job_dag.completion_time - \
        job_dag.start_time for job_dag in job_dags]

    executor_occupation = np.zeros(exp_completion_time)
    executor_limit = np.ones(exp_completion_time) * args.exec_cap

    num_jobs_in_system = np.zeros(exp_completion_time)

    for job_dag in job_dags:
        for node in job_dag.nodes:
            for task in node.tasks:
                executor_occupation[
                    int(task.start_time) : \
                    int(task.finish_time)] += 1
        num_jobs_in_system[
            int(job_dag.start_time) : \
            int(job_dag.completion_time)] += 1

    executor_usage = \
        np.sum(executor_occupation) / np.sum(executor_limit)

    fig = plt.figure()

    plt.subplot(2, 1, 1)
    # plt.plot(executor_occupation)
    # plt.fill_between(range(len(executor_occupation)), 0,
    #                  executor_occupation)
    plt.plot(moving_average(executor_occupation, 10000))

    plt.ylabel('Number of busy executors')
    plt.title('Executor usage: ' + str(executor_usage) + \
              '\n average completion time: ' + \
              str(np.mean(job_durations)))

    plt.subplot(2, 1, 2)
    plt.plot(num_jobs_in_system)
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Number of jobs in the system')

    fig.savefig(file_path)
    plt.close(fig)


def visualize_dag_time(job_dags, executors, plot_total_time=None, plot_type='stage'):

    dags_makespan = 0
    all_tasks = []
    # 1. compute each DAG's finish time
    # so that we can visualize it later
    dags_finish_time = []
    dags_duration = []
    executor_ids = []

    for dag in job_dags:
        for node in dag.nodes:
            for task in node.tasks:
                executor_ids.append(task.executor.idx)
                all_tasks.append(task)
        dags_finish_time.append(dag.completion_time)
        dags_duration.append(dag.completion_time - dag.start_time)

    # 2. visualize them in a canvas

    exec_idx_set = set(executor_ids)
    num_executor_used = len(exec_idx_set)
    exec_id_to_idx = {}
    for i, idx in enumerate(exec_idx_set):
        exec_id_to_idx[idx] = i
    if plot_total_time is None:
        canvas = np.ones([len(executors), int(max(dags_finish_time))]) * args.canvas_base
    else:
        canvas = np.ones([len(executors), int(plot_total_time)]) * args.canvas_base

    base = 0
    bases = {}  # job_dag -> base

    for job_dag in job_dags:
        bases[job_dag] = base
        base += job_dag.num_nodes

    for task, exec_id in zip(all_tasks, executor_ids):
        task = cast(Task, task)
        start_time = round(task.start_time)
        finish_time = round(task.finish_time)
        # exec_id = task.executor.idx

        if plot_type == 'stage':

            canvas[exec_id_to_idx[exec_id], start_time : finish_time+1] = \
                bases[task.node.job_dag] + task.node.idx

        elif plot_type == 'app':
            canvas[exec_id_to_idx[exec_id], start_time : finish_time+1] = \
                job_dags.index(task.node.job_dag)

    return canvas, dags_finish_time, dags_duration, num_executor_used


def visualize_dag_time_save_pdf(
        job_dags, executors, file_path, plot_total_time=None, plot_type='stage'):
    
    canvas, dag_finish_time, dags_duration = \
        visualize_dag_time(job_dags, executors, plot_total_time, plot_type)

    fig = plt.figure()

    # canvas
    plt.imshow(canvas, interpolation='nearest', aspect='auto')
    plt.colorbar()
    # each dag finish time
    # for finish_time in dag_finish_time:
    #     plt.plot([finish_time, finish_time],
    #              [- 0.5, len(executors) - 0.5], 'r')
    plt.title('average DAG completion time: ' + str(np.mean(dags_duration)))
    fig.savefig(file_path, bbox_inches='tight')
    plt.close(fig)

