# include import
import os
from argparse import ArgumentParser
import pathlib
from abc import ABC, abstractmethod
from threading import Lock, Thread
from dataclasses import dataclass
import socket
import tempfile
import atexit
import logging
import shutil
import time
import json
import pickle
import itertools
from copy import deepcopy
from os.path import exists
from requests import get
from pathlib import Path
from random import randint

# tmrl import
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.training import TrainingAgent
from tmrl.custom.custom_memories import last_true_in_list, replace_hist_before_done
from tmrl.networking import print_with_timestamp, get_connected_socket, select_and_send_or_close_socket, poll_and_recv_or_close_socket
from tmrl.util import partial
from tmrl.actor import ActorModule
from tmrl.util import pandas_dict, dump, partial_to_dict
from tmrl.nn import copy_shared, no_grad
from tmrl.util import cached_property
from tmrl.memory_dataloading import MemoryBatchSampler, check_samples_crc

# torch import
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader

# local import
from common import ENV_CLS, SquashedGaussianMLPActor, MLPQFunction, collate, Buffer
from function import *

import numpy as np
from pandas import DataFrame

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_run_instance(checkpoint_path):
    """
    Default function used to load trainers from checkpoint path
    Args:
        checkpoint_path: the path where instances of run_cls are checkpointed
    Returns:
        An instance of run_cls loaded from checkpoint_path
    """
    return load(checkpoint_path)


def dump_run_instance(run_instance, checkpoint_path):
    """
    Default function used to dump trainers to checkpoint path
    Args:
        run_instance: the instance of run_cls to checkpoint
        checkpoint_path: the path where instances of run_cls are checkpointed
    """
    dump(run_instance, checkpoint_path)

def run_with_wandb(entity, project, run_id, interface, run_cls, checkpoint_path: str = None, dump_run_instance_fn=None, load_run_instance_fn=None):
    """
    Main training loop (remote).
    saves config and stats to https://wandb.com
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance
    wandb_dir = tempfile.mkdtemp()  # prevent wandb from polluting the home directory
    atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)  # clean up after wandb atexit handler finishes
    import wandb
    logging.debug(f" run_cls: {run_cls}")
    config = partial_to_dict(run_cls)
    config['environ'] = log_environment_variables()
    # config['git'] = git_info()  # TODO: check this for bugs
    resume = checkpoint_path and exists(checkpoint_path)
    wandb.init(dir=wandb_dir, entity=entity, project=project, id=run_id, resume=resume, config=config)
    # logging.info(config)
    for stats in iterate_epochs_tm(run_cls, interface, checkpoint_path, dump_run_instance_fn, load_run_instance_fn):
        [wandb.log(json.loads(s.to_json())) for s in stats]

def run(interface, run_cls, checkpoint_path: str = None, dump_run_instance_fn=None, load_run_instance_fn=None):
    """
    Main training loop (remote).
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance
    for stats in iterate_epochs_tm(run_cls, interface, checkpoint_path, dump_run_instance_fn, load_run_instance_fn):
        pass

__docformat__ = "google"

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU, act_buf_len=0):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.numpy()

@dataclass(eq=0)
class SAC_Agent(TrainingAgent):  # Adapted from Spinup
    observation_space: type
    action_space: type
    device: str = None  # device where the model will live (None for auto)
    model_cls: type = MLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2  # fixed (v1) or initial (v2) value of the entropy coefficient
    lr_actor: float = 1e-3  # learning rate
    lr_critic: float = 1e-3  # learning rate
    lr_entropy: float = 1e-3  # entropy autotuning (SAC v2)
    learn_entropy_coef: bool = True  # if True, SAC v2 is used, else, SAC v1 is used
    target_entropy: float = None  # if None, the target entropy for SAC v2 is set automatically

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logging.debug(f" device SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.model.q1.parameters(), self.model.q2.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = Adam(self.q_params, lr=self.lr_critic)

        if self.target_entropy is None:  # automatic entropy coefficient
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):

        o, a, r, o2, d = batch
        print(len(o[0]),len(o[1]),len(o[2]))
        # print(o[:3])
        pi, logp_pi = self.model.actor(o)
        # FIXME? log_prob = log_prob.reshape(-1, 1)

        # loss_alpha:

        loss_alpha = None
        if self.learn_entropy_coef:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()

        # Run one gradient descent step for Q1 and Q2

        # loss_q:

        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.model.actor(o2)

            # Target Q-values
            q1_pi_targ = self.model_target.q1(o2, a2)
            q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha_t * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = (loss_q1 + loss_q2) / 2  # averaged for homogeneity with REDQ

        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for actor.

        # loss_pi:

        # pi, logp_pi = self.model.actor(o)
        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha_t * logp_pi - q_pi).mean()

        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        ret_dict = dict(
            loss_actor=loss_pi.detach(),
            loss_critic=loss_q.detach(),
        )

        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict

@dataclass(eq=0)
class TrainingOffline:
    """
    Training wrapper for off-policy algorithms.

    Args:
        env_cls (type): class of a dummy environment, used only to retrieve observation and action spaces if needed. Alternatively, this can be a tuple of the form (observation_space, action_space).
        memory_cls (type): class of the replay memory
        training_agent_cls (type): class of the training agent
        epochs (int): total number of epochs, we save the agent every epoch
        rounds (int): number of rounds per epoch, we generate statistics every round
        steps (int): number of training steps per round
        update_model_interval (int): number of training steps between model broadcasts
        update_buffer_interval (int): number of training steps between retrieving buffered samples
        max_training_steps_per_env_step (float): training will pause when above this ratio
        sleep_between_buffer_retrieval_attempts (float): algorithm will sleep for this amount of time when waiting for needed incoming samples
        profiling (bool): if True, run_epoch will be profiled and the profiling will be printed at the end of each epoch
        agent_scheduler (callable): if not None, must be of the form f(Agent, epoch), called at the beginning of each epoch
        start_training (int): minimum number of samples in the replay buffer before starting training
        device (str): device on which the model of the TrainingAgent will live (None for automatic)
    """

    total_updates = 0

    def __init__(self, env_cls: type = None , memory_cls: type = None, training_agent_cls: type = None, epochs: int = 10, rounds: int = 50, steps: int = 2000, update_model_interval: int = 100, update_buffer_interval: int = 100, max_training_steps_per_env_step: float = 1.0, sleep_between_buffer_retrieval_attempts: float = 0.1, profiling: bool = False, agent_scheduler: callable = None, start_training: int = 0, device: str = None, total_updates = 0) -> None:
        self.env_cls = env_cls  # = GenericGymEnv  # dummy environment, used only to retrieve observation and action spaces if needed
        self.memory_cls = memory_cls  # = MemoryDataloading  # replay memory
        self.training_agent_cls = training_agent_cls # = TrainingAgent  # training agent
        self.epochs = epochs  # total number of epochs, we save the agent every epoch
        self.rounds = rounds  # number of rounds per epoch, we generate statistics every round
        self.steps = steps  # number of training steps per round
        self.update_model_interval = update_model_interval  # number of training steps between model broadcasts
        self.update_buffer_interval = update_buffer_interval  # number of training steps between retrieving buffered samples
        self.max_training_steps_per_env_step = max_training_steps_per_env_step  # training will pause when above this ratio
        self.sleep_between_buffer_retrieval_attempts = sleep_between_buffer_retrieval_attempts  # algorithm will sleep for this amount of time when waiting for needed incoming samples
        self.profiling = profiling  # if True, run_epoch will be profiled and the profiling will be printed at the end of each epoch
        self.agent_scheduler = agent_scheduler # if not None, must be of the form f(Agent, epoch), called at the beginning of each epoch
        self.start_training = start_training  # minimum number of samples in the replay buffer before starting training
        self.device = device  # device on which the model of the TrainingAgent will live (None for automatic)

        self.total_updates = total_updates

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = 0
        self.memory = self.memory_cls(nb_steps=self.steps, device=device)
        # print(self.memory)
        if type(self.env_cls) == tuple:
            observation_space, action_space = self.env_cls
        else:
            with self.env_cls() as env:
                observation_space, action_space = env.observation_space, env.action_space
        self.agent = self.training_agent_cls(observation_space=observation_space,
                                             action_space=action_space,
                                             device=device)
        # print(self.training_agent_cls)
        self.total_samples = len(self.memory)
        logging.info(f" Initial total_samples:{self.total_samples}")

    def update_buffer(self, interface):
        buffer = interface.retrieve_buffer()
        self.memory.append(buffer)
        if len(buffer.memory)> 0:
            pass
            #print(buffer.memory)
            #print(len(self.memory[0].memory[0][1][0]), len(self.memory[0].memory[0][1][1]), len(self.memory[0].memory[0][1][2]))
        self.total_samples += len(buffer)

    def check_ratio(self, interface):
        ratio = self.total_updates / self.total_samples if self.total_samples > 0.0 and self.total_samples >= self.start_training else -1.0
        if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
            logging.info(f" Waiting for new samples")
            while ratio > self.max_training_steps_per_env_step or ratio == -1.0:
                # wait for new samples
                self.update_buffer(interface)
                ratio = self.total_updates / self.total_samples if self.total_samples > 0.0 and self.total_samples >= self.start_training else -1.0
                if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
                    time.sleep(self.sleep_between_buffer_retrieval_attempts)
            logging.info(f" Resuming training")

    def run_epoch(self, interface):
        self.training_agent_cls = AGENT
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if type(self.env_cls) == tuple:
            observation_space, action_space = self.env_cls
        else:
            with self.env_cls() as env:
                observation_space, action_space = env.observation_space, env.action_space
        self.agent = self.training_agent_cls(observation_space=observation_space,
                                             action_space=action_space,
                                             device=device)
        stats = []
        state = None

        if self.agent_scheduler is not None:
            self.agent_scheduler(self.agent, self.epoch)

        for rnd in range(self.rounds):
            logging.info(f"=== epoch {self.epoch}/{self.epochs} ".ljust(20, '=') + f" round {rnd}/{self.rounds} ".ljust(50, '='))
            logging.debug(f"(Training): current memory size:{len(self.memory)}")

            stats_training = []

            t0 = time.time()
            self.check_ratio(interface)
            t1 = time.time()

            if self.profiling:
                from pyinstrument import Profiler
                pro = Profiler()
                pro.start()

            t2 = time.time()

            t_sample_prev = t2

            for batch in self.memory:  # this samples a fixed number of batches

                if cfg.SYNCHRONIZE_CUDA:
                    torch.cuda.synchronize()

                t_sample = time.time()

                if self.total_updates % self.update_buffer_interval == 0:
                    # retrieve local buffer in replay memory
                    self.update_buffer(interface)

                t_update_buffer = time.time()

                if self.total_updates == 0:
                    logging.info(f"starting training")
                
                stats_training_dict = self.agent.train(batch)

                if cfg.SYNCHRONIZE_CUDA:
                    torch.cuda.synchronize()

                t_train = time.time()

                stats_training_dict["return_test"] = self.memory.stat_test_return
                stats_training_dict["return_train"] = self.memory.stat_train_return
                stats_training_dict["episode_length_test"] = self.memory.stat_test_steps
                stats_training_dict["episode_length_train"] = self.memory.stat_train_steps
                stats_training_dict["sampling_duration"] = t_sample - t_sample_prev
                stats_training_dict["training_step_duration"] = t_train - t_update_buffer
                stats_training += stats_training_dict,
                self.total_updates += 1
                if self.total_updates % self.update_model_interval == 0:
                    # broadcast model weights
                    interface.broadcast_model(self.agent.get_actor())
                self.check_ratio(interface)

                if cfg.SYNCHRONIZE_CUDA:
                    torch.cuda.synchronize()

                t_sample_prev = time.time()

            t3 = time.time()

            round_time = t3 - t0
            idle_time = t1 - t0
            update_buf_time = t2 - t1
            train_time = t3 - t2
            logging.debug(f"round_time:{round_time}, idle_time:{idle_time}, update_buf_time:{update_buf_time}, train_time:{train_time}")
            stats += pandas_dict(memory_len=len(self.memory), round_time=round_time, idle_time=idle_time, **DataFrame(stats_training).mean(skipna=True)),

            logging.info(stats[-1].add_prefix("  ").to_string() + '\n')

            if self.profiling:
                pro.stop()
                logging.info(pro.output_text(unicode=True, color=False, show_all=True))

        self.epoch += 1
        return stats

class TrainerInterface:
    """
    This is the trainer's network interface
    This connects to the server
    This receives samples batches and sends new weights
    """
    def __init__(self, server_ip=None, model_path=cfg.MODEL_PATH_TRAINER):
        self.__buffer_lock = Lock()
        self.__weights_lock = Lock()
        self.__weights = None
        self.__buffer = Buffer()
        self.model_path = model_path
        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'
        self.recv_tiemout = cfg.RECV_TIMEOUT_TRAINER_FROM_SERVER

        print_with_timestamp(f"local IP: {self.local_ip}")
        print_with_timestamp(f"public IP: {self.public_ip}")
        print_with_timestamp(f"server IP: {self.server_ip}")

        Thread(target=self.__run_thread, args=(), kwargs={}, daemon=True).start()

    def __run_thread(self):
        """
        Trainer interface thread
        """
        while True:  # main client loop
            ack_time = time.time()
            recv_time = time.time()
            wait_ack = False
            s = get_connected_socket(cfg.SOCKET_TIMEOUT_CONNECT_TRAINER, self.server_ip, cfg.PORT_TRAINER)
            if s is None:
                print_with_timestamp("get_connected_socket failed in TrainerInterface thread")
                continue
            while True:
                # send weights
                self.__weights_lock.acquire()  # WEIGHTS LOCK...........................................................
                if self.__weights is not None:  # new weights
                    if not wait_ack:
                        obj = self.__weights
                        if select_and_send_or_close_socket(obj, s):
                            ack_time = time.time()
                            wait_ack = True
                        else:
                            self.__weights_lock.release()
                            print_with_timestamp("select_and_send_or_close_socket failed in TrainerInterface")
                            break
                        self.__weights = None
                    else:
                        elapsed = time.time() - ack_time
                        print_with_timestamp(f"CAUTION: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= cfg.ACK_TIMEOUT_TRAINER_TO_SERVER:
                            print_with_timestamp("ACK timed-out, breaking connection")
                            self.__weights_lock.release()
                            wait_ack = False
                            break
                self.__weights_lock.release()  # END WEIGHTS LOCK.......................................................
                # checks for samples batch
                success, obj = poll_and_recv_or_close_socket(s)
                if not success:
                    print_with_timestamp("poll failed in TrainerInterface thread")
                    break
                elif obj is not None and obj != 'ACK':  # received buffer
                    print_with_timestamp(f"trainer interface received obj")
                    recv_time = time.time()
                    self.__buffer_lock.acquire()  # BUFFER LOCK.........................................................
                    self.__buffer += obj
                    self.__buffer_lock.release()  # END BUFFER LOCK.....................................................
                elif obj == 'ACK':
                    wait_ack = False
                    print_with_timestamp(f"transfer acknowledgment received after {time.time() - ack_time}s")
                elif time.time() - recv_time > self.recv_tiemout:
                    print_with_timestamp(f"Timeout in TrainerInterface, not received anything for too long")
                    break
                time.sleep(cfg.LOOP_SLEEP_TIME)  # TODO: adapt
            s.close()

    def broadcast_model(self, model: ActorModule):
        """
        model must be an ActorModule
        broadcasts the model's weights to all connected RolloutWorkers
        """
        t0 = time.time()
        self.__weights_lock.acquire()  # WEIGHTS LOCK...................................................................
        t1 = time.time()
        torch.save(model.state_dict(), self.model_path)
        t2 = time.time()
        with open(self.model_path, 'rb') as f:
            self.__weights = f.read()
        t3 = time.time()
        self.__weights_lock.release()  # END WEIGHTS LOCK...............................................................
        print_with_timestamp(f"broadcast_model: lock acquire: {t1 - t0}s, save dict: {t2 - t1}s, read dict: {t3 - t2}s")

    def retrieve_buffer(self):
        """
        returns a copy of the TrainerInterface's local buffer, and clears it
        """
        self.__buffer_lock.acquire()  # BUFFER LOCK.....................................................................
        buffer_copy = deepcopy(self.__buffer)
        self.__buffer.clear()
        self.__buffer_lock.release()  # END BUFFER LOCK.................................................................
        return buffer_copy

def iterate_epochs_tm(run_cls,
                      interface: TrainerInterface,
                      checkpoint_path: str,
                      dump_run_instance_fn=dump_run_instance,
                      load_run_instance_fn=load_run_instance,
                      epochs_between_checkpoints=1):
    """
    Main training loop (remote)
    The run_cls instance is saved in checkpoint_path at the end of each epoch
    The model weights are sent to the RolloutWorker every model_checkpoint_interval epochs
    Generator yielding episode statistics (list of pd.Series) while running and checkpointing
    """
    checkpoint_path = checkpoint_path or tempfile.mktemp("_remove_on_exit")

    try:
        print(f"DEBUF: checkpoint_path: {checkpoint_path}")
        if not exists(checkpoint_path):
            logging.info(f"=== specification ".ljust(70, "="))
            run_instance = run_cls()
            dump_run_instance_fn(run_instance, checkpoint_path)
            logging.info(f"")
        else:
            logging.info(f" Loading checkpoint...")
            t1 = time.time()
            run_instance = load_run_instance_fn(checkpoint_path)
            logging.info(f" Loaded checkpoint in {time.time() - t1} seconds.")
        while run_instance.epoch < run_instance.epochs:
            # time.sleep(1)  # on network file systems writing files is asynchronous and we need to wait for sync
            yield run_instance.run_epoch(interface=interface)  # yield stats data frame (this makes this function a generator)
            if run_instance.epoch % epochs_between_checkpoints == 0:
                logging.info(f" saving checkpoint...")
                t1 = time.time()
                dump_run_instance_fn(run_instance, checkpoint_path)
                logging.info(f" saved checkpoint in {time.time() - t1} seconds.")
                # we delete and reload the run_instance from disk to ensure the exact same code runs regardless of interruptions
                # del run_instance
                # gc.collect()  # garbage collection
                # run_instance = load_run_instance_fn(checkpoint_path)

    finally:
        if checkpoint_path.endswith("_remove_on_exit") and exists(checkpoint_path):
            os.remove(checkpoint_path)

class MemoryDataloading(ABC):  # FIXME: should be an instance of Dataset but partial doesn't work with Dataset
    """
    Interface for a simple replay buffer.

    This class supports sampling and collating simple batches of prev_obs, new_act, new_obs, rew, done.

    In case you need more advanced replay buffers, you can store whatever you need in the `info` dict and collate
    batches manually in your TrainingAgent.

    .. note::
       When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
       Your `__init__` method needs to take at least all the arguments of the superclass.
    """
    def __init__(self,
                 device,
                 nb_steps,
                 sample_preprocessor: callable = None,
                 memory_size=1000000,
                 batch_size=256,
                 dataset_path="",
                 crc_debug=False,
                 use_dataloader=False,
                 num_workers=0,
                 pin_memory=False):
        """
        Args:
            device (str): output tensors will be collated to this device
            nb_steps (int): number of steps per round
            sample_preprocessor (callable): can be used for data augmentation
            memory_size (int): size of the circular buffer
            batch_size (int): batch size of the output tensors
            dataset_path (str): an offline dataset may be provided here to initialize the memory
            crc_debug (bool): False usually, True when using CRC debugging of the pipeline
            use_dataloader (bool): Not yet supported
            num_workers (int): Not yet supported
            pin_memory: Not yet supported
        """
        self.nb_steps = nb_steps
        self.use_dataloader = use_dataloader
        self.device = device
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.sample_preprocessor = sample_preprocessor
        self.crc_debug = crc_debug

        # These stats are here because they reach the trainer along with the buffer:
        self.stat_test_return = 0.0
        self.stat_train_return = 0.0
        self.stat_test_steps = 0
        self.stat_train_steps = 0

        # init memory
        self.path = Path(dataset_path)
        print(f"DEBUG: MemoryDataloading self.path:{self.path}")
        if os.path.isfile(self.path / 'data.pkl'):
            with open(self.path / 'data.pkl', 'rb') as f:
                self.data = list(pickle.load(f))
                # print(f"DEBUG: len data:{len(self.data)}")
                # print(f"DEBUG: len data[0]:{len(self.data[0])}")
        else:
            print("INFO: no data found, initializing empty replay memory")
            self.data = []

        if len(self) > self.memory_size:
            # TODO: crop to memory_size
            print(f"WARNING: the dataset length ({len(self)}) is longer than memory_size ({self.memory_size})")

        # init dataloader
        self._batch_sampler = MemoryBatchSampler(data_source=self, nb_steps=nb_steps, batch_size=batch_size)
        self._dataloader = DataLoader(dataset=self, batch_sampler=self._batch_sampler, num_workers=num_workers, pin_memory=pin_memory)

    def __iter__(self):
        if not self.use_dataloader:
            for _ in range(self.nb_steps):
                yield self.sample()
        else:
            for batch in self._dataloader:
                yield batch  # TODO: move this to self.device !!!

    @abstractmethod
    def append_buffer(self, buffer):
        """
        Must append a Buffer object to the memory.

        Args:
            buffer (tmrl.networking.Buffer): the buffer of samples to append.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """
        Must return the length of the memory.

        Returns:
            length (int): the maximum `item` argument of `get_transition`

        """
        raise NotImplementedError

    @abstractmethod
    def get_transition(self, item):
        """
        Must return a transition.

        `info` is required in each sample for CRC debugging. The 'crc' key is what is important when using this feature.
        Do NOT apply observation preprocessing here, as it will be applied automatically.

        Args:
            item (int): the index where to sample

        Returns:
            sample (Tuple): (prev_obs, prev_act, rew, obs, done, info)
        """
        raise NotImplementedError

    def append(self, buffer):
        if len(buffer) > 0:
            self.stat_train_return = buffer.stat_train_return
            self.stat_test_return = buffer.stat_test_return
            self.stat_train_steps = buffer.stat_train_steps
            self.stat_test_steps = buffer.stat_test_steps
            self.append_buffer(buffer)

    def __getitem__(self, item):
        prev_obs, new_act, rew, new_obs, done, info = self.get_transition(item)
        if self.crc_debug:
            po, a, o, r, d = info['crc_sample']
            check_samples_crc(po, a, o, r, d, prev_obs, new_act, new_obs, rew, done)
        if self.sample_preprocessor is not None:
            prev_obs, new_act, rew, new_obs, done = self.sample_preprocessor(prev_obs, new_act, rew, new_obs, done)
        done = np.float32(done)  # we don't want bool tensors
        return prev_obs, new_act, rew, new_obs, done

    def sample_indices(self):
        return (randint(0, len(self) - 1) for _ in range(self.batch_size))

    def sample(self, indices=None):
        indices = self.sample_indices() if indices is None else indices
        batch = [self[idx] for idx in indices]
        batch = collate(batch, self.device)
        return batch

class MemoryTM(MemoryDataloading):
    def __init__(self,
                 memory_size=None,
                 batch_size=None,
                 dataset_path="",
                 imgs_obs=4,
                 act_buf_len=1,
                 nb_steps=1,
                 use_dataloader=False,
                 num_workers=0,
                 pin_memory=False,
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu"):
        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)
        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         nb_steps=nb_steps,
                         use_dataloader=use_dataloader,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)

    def append_buffer(self, buffer):
        raise NotImplementedError

    def __len__(self):
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        if res < 0:
            return 0
        else:
            return res

    def get_transition(self, item):
        raise NotImplementedError

class MemoryTMNFCustom(MemoryTM):
    def get_transition(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """
        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        poss = self.load_poss(item)
        poss_last_obs = self.data[3][idx_last]
        poss_new_obs = self.data[3][idx_now]
        
        poss_last_obs = np.array(poss_last_obs)
        poss_new_obs = np.array(poss_new_obs)

        # if a reset transition has influenced the observation, special care must be taken
        last_dones = self.data[5][idx_now - self.min_samples:idx_now]  # self.min_samples values
        last_done_idx = last_true_in_list(last_dones)  # last occurrence of True

        assert last_done_idx is None or last_dones[last_done_idx], f"DEBUG: last_done_idx:{last_done_idx}"

        last_infos = self.data[7][idx_now - self.min_samples:idx_now]
        last_ignored_dones = ["__no_done" in i for i in last_infos]
        last_ignored_done_idx = last_true_in_list(last_ignored_dones)  # last occurrence of True

        assert last_ignored_done_idx is None or last_ignored_dones[last_ignored_done_idx] and not last_dones[last_ignored_done_idx], f"DEBUG: last_ignored_done_idx:{last_ignored_done_idx}, last_ignored_dones:{last_ignored_dones}, last_dones:{last_dones}"

        if last_ignored_done_idx is not None:
            last_done_idx = last_ignored_done_idx  # FIXME: might not work in extreme cases where a done is ignored right after another done

        if last_done_idx is not None:
            replace_hist_before_done(hist=new_act_buf, done_idx_in_hist=last_done_idx - self.start_acts_offset - 1)
            replace_hist_before_done(hist=last_act_buf, done_idx_in_hist=last_done_idx - self.start_acts_offset)
            replace_hist_before_done(hist=poss_new_obs, done_idx_in_hist=last_done_idx - self.start_imgs_offset - 1)
            replace_hist_before_done(hist=poss_last_obs, done_idx_in_hist=last_done_idx - self.start_imgs_offset)

        poss_new_obs = np.ndarray.flatten(poss_new_obs)
        poss_last_obs = np.ndarray.flatten(poss_last_obs)

        last_obs = (self.data[2][idx_last], poss_new_obs, self.data[4][idx_last], *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[6][idx_now])
        new_obs = (self.data[2][idx_now], poss_last_obs, self.data[4][idx_now], *new_act_buf)
        done = self.data[5][idx_now]
        info = self.data[7][idx_now]
        # print(len(last_obs[0]), len(last_obs[1]), len(last_obs[2]))
        return last_obs, new_act, rew, new_obs, done, info

    def load_poss(self, item):
        res = self.data[3][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        return np.stack(res)

    def load_acts(self, item):
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]
        return res

    def append_buffer(self, buffer):
        """
        buffer is a list of samples ( act, obs, rew, done, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """
        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][1] for b in buffer.memory]  # poss
        d4 = [b[1][2] for b in buffer.memory]  # track
        d5 = [b[3] for b in buffer.memory]  # dones
        d6 = [b[2] for b in buffer.memory]  # rewards
        d7 = [b[4] for b in buffer.memory]  # infos

        if self.__len__() > 0:
            for i in d0:
                self.data[0].append(i)
            for i in d1:
                self.data[1].append(i)
            for i in d2:
                self.data[2].append(i)
            for i in d3:
                self.data[3].append(i)
            for i in d4:
                self.data[4].append(i)
            for i in d5:
                self.data[5].append(i)
            for i in d6:
                self.data[6].append(i)
            for i in d7:
                self.data[7].append(i)
            # self.data[0] += d0
            # self.data[1] += d1
            # self.data[2] += d2
            # self.data[3] += d3
            # self.data[4] += d4
            # self.data[5] += d5
            # self.data[6] += d6
            # self.data[7] += d7
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]

        return self

ALG_CONFIG = cfg.TMRL_CONFIG["ALG"]

TRAIN_MODEL = MLPActorCritic

AGENT = partial(
    SAC_Agent,
    device='cuda' if cfg.PRAGMA_CUDA_TRAINING else 'cpu',
    model_cls=partial(TRAIN_MODEL, act_buf_len=cfg.ACT_BUF_LEN),
    lr_actor=ALG_CONFIG["LR_ACTOR"],
    lr_critic=ALG_CONFIG["LR_CRITIC"],
    lr_entropy=ALG_CONFIG["LR_ENTROPY"],
    gamma=ALG_CONFIG["GAMMA"],
    polyak=ALG_CONFIG["POLYAK"],
    learn_entropy_coef=ALG_CONFIG["LEARN_ENTROPY_COEF"],  # False for SAC v2 with no temperature autotuning
    target_entropy=ALG_CONFIG["TARGET_ENTROPY"],  # None for automatic
    alpha=ALG_CONFIG["ALPHA"]  # inverse of reward scale
)

MEM = MemoryTMNFCustom

TMRL_FOLDER = pathlib.Path("D:/GitHub/trackmania-ia-with-tmrl") / "TmrlData"

CHECKPOINTS_FOLDER = TMRL_FOLDER / "checkpoints"

RUN_NAME = 'SAC_4_Custom'

WEIGHTS_FOLDER = TMRL_FOLDER / "weights"

MODEL_PATH_TRAINER = str(WEIGHTS_FOLDER / (RUN_NAME + "_t.pth"))
DATASET_PATH = str(TMRL_FOLDER / "dataset")
CHECKPOINT_PATH = str(CHECKPOINTS_FOLDER / RUN_NAME)

SAMPLE_PREPROCESSOR = None

MEMORY = partial(MEM,
                 memory_size=cfg.TMRL_CONFIG["MEMORY_SIZE"],
                 batch_size=cfg.TMRL_CONFIG["BATCH_SIZE"],
                 sample_preprocessor=SAMPLE_PREPROCESSOR,
                 dataset_path=DATASET_PATH,
                 imgs_obs=cfg.IMG_HIST_LEN,
                 act_buf_len=cfg.ACT_BUF_LEN,
                 crc_debug=cfg.CRC_DEBUG,
                 use_dataloader=False,
                 pin_memory=False)

TRAINER = partial(
        TrainingOffline,
        env_cls=ENV_CLS,
        memory_cls=MEMORY,
        epochs=cfg.TMRL_CONFIG["MAX_EPOCHS"],
        rounds=cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"],
        steps=cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"],
        update_model_interval=cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"],
        update_buffer_interval=cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"],
        max_training_steps_per_env_step=cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"],
        profiling=cfg.PROFILE_TRAINER,
        training_agent_cls=AGENT,
        agent_scheduler=None,  # sac_v2_entropy_scheduler
        start_training=cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"])  # set this > 0 to start from an existing policy (fills the buffer up to this number of samples before starting training)

class Trainer:
    """
    Training entity.
    The `Trainer` object is where RL training happens.
    Typically, it can be located on a HPC cluster.
    """
    def __init__(self,
                 training_cls=TRAINER,
                 server_ip=cfg.SERVER_IP_FOR_TRAINER,
                 model_path=cfg.MODEL_PATH_TRAINER,
                 checkpoint_path=cfg.CHECKPOINT_PATH,
                 dump_run_instance_fn: callable = None,
                 load_run_instance_fn: callable = None):
        """
        Args:
            training_cls (type): training class (subclass of tmrl.training_offline.TrainingOffline)
            server_ip (str): ip of the central `Server`
            model_path (str): path where a local copy of the model will be saved
            checkpoint_path: path where the `Trainer` will be checkpointed (`None` = no checkpointing)
            dump_run_instance_fn (callable): custom serializer (`None` = pickle.dump)
            load_run_instance_fn (callable): custom deserializer (`None` = pickle.load)
        """
        self.checkpoint_path = checkpoint_path
        self.dump_run_instance_fn = dump_run_instance_fn
        self.load_run_instance_fn = load_run_instance_fn
        self.training_cls = training_cls
        self.interface = TrainerInterface(server_ip=server_ip,
                                          model_path=model_path)

    def run(self):
        """
        Runs training.
        """
        run(interface=self.interface,
            run_cls=self.training_cls,
            checkpoint_path=self.checkpoint_path,
            dump_run_instance_fn=self.dump_run_instance_fn,
            load_run_instance_fn=self.load_run_instance_fn)

    def run_with_wandb(self,
                       entity=cfg.WANDB_ENTITY,
                       project=cfg.WANDB_PROJECT,
                       run_id=cfg.WANDB_RUN_ID,
                       key=None):
        """
        Runs training while logging metrics to wandb_.
        .. _wandb: https://wandb.ai
        Args:
            entity (str): wandb entity
            project (str): wandb project
            run_id (str): name of the run
            key (str): wandb API key
        """
        if key is not None:
            os.environ['WANDB_API_KEY'] = key
        run_with_wandb(entity=entity,
                       project=project,
                       run_id=run_id,
                       interface=self.interface,
                       run_cls=self.training_cls,
                       checkpoint_path=self.checkpoint_path,
                       dump_run_instance_fn=self.dump_run_instance_fn,
                       load_run_instance_fn=self.load_run_instance_fn)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--no-wandb', dest='no_wandb', action='store_true', help='(use with --trainer) if you do not want to log results on Weights and Biases, use this option')

    args = parser.parse_args()
    trainer = Trainer(training_cls=TRAINER,
                        server_ip="192.168.0.181",
                        model_path=MODEL_PATH_TRAINER,
                        checkpoint_path=CHECKPOINT_PATH,
                        dump_run_instance_fn=cfg_obj.DUMP_RUN_INSTANCE_FN,
                        load_run_instance_fn=cfg_obj.LOAD_RUN_INSTANCE_FN)
    logging.info(f"--- NOW RUNNING {cfg_obj.ALG_NAME} on TrackMania ---")
    if not args.no_wandb and False:
        print("ok my body")
        trainer.run_with_wandb(entity=cfg.WANDB_ENTITY,
                                project=cfg.WANDB_PROJECT,
                                run_id=cfg.WANDB_RUN_ID)
    else:
        trainer.run()