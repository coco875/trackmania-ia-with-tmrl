import json
from argparse import ArgumentParser

import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.networking import print_with_timestamp, get_connected_socket, poll_and_recv_or_close_socket
from threading import Lock, Thread
from pygbx.headers import Vector3
from common import ENV_CLS, GenericGymEnv, POLICY, collate, Buffer

import torch
import datetime
import os
from requests import get
import socket
import numpy as np
import logging
import pathlib
import time
import select
import pickle
from typing import Mapping, Sequence

parser = ArgumentParser()
parser.add_argument('--test', action='store_true', help='runs inference without training')
parser.add_argument('-d', '--config', type=json.loads, default={}, help='dictionary containing configuration options (modifiers) for the rtgym environment')

args = parser.parse_args()

config = cfg_obj.CONFIG_DICT
config_modifiers = args.config
for k, v in config_modifiers.items():
    config[k] = v

def send_object(sock, obj, ping=False, pong=False, ack=False):
    """
    If ping, this will ignore obj and send the PING request
    If pong, this will ignore obj and send the PONG request
    If ack, this will ignore obj and send the ACK request
    If raw, obj must be a binary string
    Call only after select on a socket with a (long enough) timeout.
    Returns True if sent successfully, False if connection lost.
    """
    if ping:
        msg = bytes(f"{'PING':<{cfg.HEADER_SIZE}}", 'utf-8')
    elif pong:
        msg = bytes(f"{'PONG':<{cfg.HEADER_SIZE}}", 'utf-8')
    elif ack:
        msg = bytes(f"{'ACK':<{cfg.HEADER_SIZE}}", 'utf-8')
    else:
        msg = pickle.dumps(obj)
        msg = bytes(f"{len(msg):<{cfg.HEADER_SIZE}}", 'utf-8') + msg
        if cfg.PRINT_BYTESIZES:
            print_with_timestamp(f"Sending {len(msg)} bytes.")
    try:
        sock.sendall(msg)
    except OSError:  # connection closed or broken
        return False
    return True

def select_and_send_or_close_socket(obj, conn):
    """
    Returns True if success
    False if disconnected (closes sockets)
    """
    print_with_timestamp(f"start select")
    _, wl, xl = select.select([], [conn], [conn], cfg.SELECT_TIMEOUT_OUTBOUND)  # select for writing
    print_with_timestamp(f"end select")
    if len(xl) != 0:
        print_with_timestamp("error when writing, closing socket")
        conn.close()
        return False
    if len(wl) == 0:
        print_with_timestamp("outbound select() timed out, closing socket")
        conn.close()
        return False
    elif not send_object(conn, obj):  # error or timeout
        print_with_timestamp("send_object() failed, closing socket")
        conn.close()
        return False
    return True

TMRL_FOLDER = pathlib.Path("D:/GitHub/trackmania-ia-with-tmrl") / "TmrlData"

RUN_NAME = 'SAC_4_Custom'

WEIGHTS_FOLDER = TMRL_FOLDER / "weights"

MODEL_PATH_WORKER = str(WEIGHTS_FOLDER / (RUN_NAME + ".pth"))

class RolloutWorker:
    """Actor.

    A `RolloutWorker` deploys the current policy in the environment.
    A `RolloutWorker` may connect to a `Server` to which it sends buffered experience.
    Alternatively, it may exist in standalone mode for deployment.
    """
    def __init__(
            self,
            env_cls: GenericGymEnv,
            actor_module_cls,
            sample_compressor: callable = None,
            device="cpu",
            server_ip=None,
            min_samples_per_worker_packet=1,
            max_samples_per_episode=np.inf,
            model_path=cfg.MODEL_PATH_WORKER,
            obs_preprocessor: callable = None,
            crc_debug=False,
            model_path_history=cfg.MODEL_PATH_SAVE_HISTORY,
            model_history=cfg.MODEL_HISTORY,
            standalone=False
    ):
        """
        Args:
            env_cls (type): class of the Gym environment (subclass of tmrl.envs.GenericGymEnv)
            actor_module_cls (type): class of the module containing the policy (subclass of tmrl.actor.ActorModule)
            sample_compressor (callable): compressor for sending samples over the Internet
            device (str): device on which the policy is running
            server_ip (str): ip of the central server
            min_samples_per_worker_packet (int): the worker waits for this number of samples before sending
            max_samples_per_episode (int): if an episode gets longer than this, it is reset
            model_path (str): path where a local copy of the policy will be stored
            obs_preprocessor (callable): utility for modifying samples before forward passes
            crc_debug (bool): can be used for debugging the pipeline
            model_path_history (str): (omit .pth) an history of policies can be stored here
            model_history (int): new policies are saved % model_history (0: not saved)
            standalone (bool): If True, the worker will not try to connect to a server
        """
        self.obs_preprocessor = obs_preprocessor
        self.get_local_buffer_sample = sample_compressor
        self.env = env_cls()
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.model_path = model_path
        self.model_path_history = model_path_history
        self.actor = actor_module_cls(observation_space=obs_space, action_space=act_space).to(device)
        self.device = device
        self.standalone = standalone
        if os.path.isfile(self.model_path):
            logging.debug(f"Loading model from {self.model_path}")
            self.actor.load_state_dict(torch.load(self.model_path, map_location=self.device))
        else:
            logging.debug(f"No model found at {self.model_path}")
        self.buffer = Buffer()
        self.__buffer = Buffer()  # deepcopy for sending
        self.__buffer_lock = Lock()
        self.__weights = None
        self.__weights_lock = Lock()
        self.samples_per_worker_batch = min_samples_per_worker_packet
        self.max_samples_per_episode = max_samples_per_episode
        self.crc_debug = crc_debug
        self.model_history = model_history
        self._cur_hist_cpt = 0

        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'
        self.recv_timeout = cfg.RECV_TIMEOUT_WORKER_FROM_SERVER

        print_with_timestamp(f"local IP: {self.local_ip}")
        print_with_timestamp(f"public IP: {self.public_ip}")
        print_with_timestamp(f"server IP: {self.server_ip}")

        if not self.standalone:
            Thread(target=self.__run_thread, args=(), kwargs={}, daemon=True).start()

    def __run_thread(self):
        """
        Redis thread
        """
        while True:  # main client loop
            ack_time = time.time()
            recv_time = time.time()
            wait_ack = False
            s = get_connected_socket(cfg.SOCKET_TIMEOUT_CONNECT_ROLLOUT, self.server_ip, cfg.PORT_ROLLOUT)
            if s is None:
                print_with_timestamp("get_connected_socket failed in worker")
                continue
            while True:
                # send buffer
                self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
                if len(self.__buffer) >= self.samples_per_worker_batch:  # a new batch is available
                    print_with_timestamp("new batch available")
                    if not wait_ack:
                        obj = self.__buffer
                        if select_and_send_or_close_socket(obj, s):
                            ack_time = time.time()
                            wait_ack = True
                        else:
                            self.__buffer_lock.release()
                            print_with_timestamp("select_and_send_or_close_socket failed in worker")
                            break
                        self.__buffer.clear()  # empty sent batch
                    else:
                        elapsed = time.time() - ack_time
                        print_with_timestamp(f"CAUTION: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= cfg.ACK_TIMEOUT_WORKER_TO_SERVER:
                            print_with_timestamp("ACK timed-out, breaking connection")
                            self.__buffer_lock.release()
                            wait_ack = False
                            break
                self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
                # checks for new weights
                success, obj = poll_and_recv_or_close_socket(s)
                if not success:
                    print_with_timestamp(f"rollout worker poll failed")
                    break
                elif obj is not None and obj != 'ACK':
                    print_with_timestamp(f"rollout worker received obj")
                    recv_time = time.time()
                    self.__weights_lock.acquire()  # WEIGHTS LOCK.......................................................
                    self.__weights = obj
                    self.__weights_lock.release()  # END WEIGHTS LOCK...................................................
                elif obj == 'ACK':
                    wait_ack = False
                    print_with_timestamp(f"transfer acknowledgment received after {time.time() - ack_time}s")
                elif time.time() - recv_time > self.recv_timeout:
                    print_with_timestamp(f"Timeout in RolloutWorker, not received anything for too long")
                    break
                time.sleep(cfg.LOOP_SLEEP_TIME)  # TODO: adapt
            s.close()

    def act(self, obs, test=False):
        """
        Converts inputs to torch tensors and converts outputs to numpy arrays.

        Args:
            obs (nested structure): observation
            test (bool): directly passed to the `act()` method of the `ActorModule`

        Returns:
            action (numpy.array): action computed by the `ActorModule`
        """
        # if self.obs_preprocessor is not None:
        #     obs = self.obs_preprocessor(obs)
        obs = collate([obs], device=self.device)
        with torch.no_grad():
            action = self.actor.act(obs, test=test)
        return action

    def reset(self, collect_samples):
        """
        Starts a new episode.

        Args:
            collect_samples (bool): if True, samples are buffered and sent to the `Server`

        Returns:
            obs (nested structure): observation retrieved from the environment
        """
        obs = None
        act = self.env.default_action.astype(np.float32)
        new_obs = self.env.reset()
        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        rew = 0.0
        done = False
        info = {}
        if collect_samples:
            if self.crc_debug:
                info['crc_sample'] = (obs, act, new_obs, rew, done)
            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(act, new_obs, rew, done, info)
            else:
                sample = act, new_obs, rew, done, info
            print(len(sample[1][0]), len(sample[1][1]), len(sample[1][2]))
            self.buffer.append_sample(sample)
        return new_obs

    def step(self, obs, test, collect_samples, last_step=False):
        """
        Performs a full RL transition.

        A full RL transition is `obs` -> `act` -> `new_obs`, `rew`, `done`, `info`.
        Note that, in the Real-Time RL setting, `act` is appended to a buffer which is part of `new_obs`.
        This is because is does not directly affect the new observation, due to real-time delays.

        Args:
            obs (nested structure): previous observation
            test (bool): passed to the `act()` method of the `ActorModule`
            collect_samples (bool): if True, samples are buffered and sent to the `Server`
            last_step (bool): if True and `done` is False, a '__no_done' entry will be added to the `info` dict

        Returns:
            new_obs (nested structure): new observation
            rew (float): new reward
            done (bool): episode termination signal
            info (dict): information dictionary
        """
        act = self.act(obs, test=test)
        new_obs, rew, done, info = self.env.step(act)
        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        if collect_samples:
            stored_done = done
            if last_step and not done:  # ignore done when stopped by step limit
                info["__no_done"] = True
            if "__no_done" in info:
                stored_done = False
            if self.crc_debug:
                info['crc_sample'] = (obs, act, new_obs, rew, stored_done)
            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(act, new_obs, rew, stored_done, info)
            else:
                sample = act, new_obs, rew, stored_done, info
            # print(len(self.buffer.memory[0][1][0]), len(self.buffer.memory[0][1][1]), len(self.buffer.memory[0][1][2]))
            self.buffer.append_sample(sample)  # CAUTION: in the buffer, act is for the PREVIOUS transition (act, obs(act))
        return new_obs, rew, done, info

    def collect_train_episode(self, max_samples):
        """
        Collects a maximum of n training transitions (from reset to done)

        This method stores the episode and the train return in the local `Buffer` of the worker
        for sending to the `Server`.

        Args:
            max_samples (int): if the environment is not `done` after `max_samples` time steps,
                it is forcefully reset and a '__no_done' entry is added to the `info` dict of the corresponding sample.
        """
        ret = 0.0
        steps = 0
        obs = self.reset(collect_samples=True)
        for i in range(max_samples):
            obs, rew, done, info = self.step(obs=obs, test=False, collect_samples=True, last_step=i == max_samples - 1)
            ret += rew
            steps += 1
            if done:
                break
        self.buffer.stat_train_return = ret
        self.buffer.stat_train_steps = steps

    def run_episodes(self, max_samples_per_episode, nb_episodes=np.inf, train=False):
        """
        Runs `nb_episodes` episodes.

        Args:
            max_samples_per_episode (int): same as run_episode
            nb_episodes (int): total number of episodes to collect
            train (bool): same as run_episode
        """
        counter = 0
        while counter < nb_episodes:
            self.run_episode(max_samples_per_episode, train=train)
            counter += 1

    def run_episode(self, max_samples, train=False):
        """
        Collects a maximum of n test transitions (from reset to done).

        Args:
            max_samples (int): At most `max_samples` samples are collected per episode.
                If the episode is longer, it is forcefully reset and a '__no_done' entry is added to the `info` dict
                of the corresponding sample.
            train (bool): whether the episode is a training or a test episode.
                `step` is called with `test=not train`.
        """
        ret = 0.0
        steps = 0
        obs = self.reset(collect_samples=False)
        for _ in range(max_samples):
            obs, rew, done, info = self.step(obs=obs, test=not train, collect_samples=False)
            ret += rew
            steps += 1
            if done:
                break
        self.buffer.stat_test_return = ret
        self.buffer.stat_test_steps = steps

    def run(self, test_episode_interval=20, nb_episodes=np.inf):  # TODO: check number of collected samples are collected before sending
        """
        Runs the worker for `nb_episodes` episodes.

        This method is for training.
        It collects a test episode each `test_episode_interval` episodes.
        For deployment, use the `run_episodes` method instead.

        Args:
            test_episode_interval (int):
            nb_episodes (int):
        """
        episode = 0
        while episode < nb_episodes:
            if episode % test_episode_interval == 0 and not self.crc_debug:
                print_with_timestamp("running test episode")
                self.run_episode(self.max_samples_per_episode, train=False)
            print_with_timestamp("collecting train episode")
            self.collect_train_episode(self.max_samples_per_episode)
            print_with_timestamp("copying buffer for sending")
            self.send_and_clear_buffer()
            print_with_timestamp("checking for new weights")
            self.update_actor_weights()
            episode += 1
            # if self.crc_debug:
            #     break

    def profile_step(self):
        import torch.autograd.profiler as profiler
        obs = self.reset(collect_samples=True)
        use_cuda = True if self.device == 'cuda' else False
        print_with_timestamp(f"use_cuda:{use_cuda}")
        with profiler.profile(record_shapes=True, use_cuda=use_cuda) as prof:
            obs = collate([obs], device=self.device)
            with profiler.record_function("pytorch_profiler"):
                with torch.no_grad():
                    action_distribution = self.actor(obs)
                    action = action_distribution.sample()
        print_with_timestamp(prof.key_averages().table(row_limit=20, sort_by="cpu_time_total"))

    def run_env_benchmark(self, nb_steps, test=False):
        """
        Benchmarks the environment.

        This method is only compatible with rtgym_ environments.
        Furthermore, the `"benchmark"` option of the rtgym configuration dictionary must be set to `True`.

        .. _rtgym: https://github.com/yannbouteiller/rtgym

        Args:
            nb_steps (int): number of steps to perform to compute the benchmark
            test (int): whether the actor is called in test or train mode
        """
        obs = self.reset(collect_samples=False)
        for _ in range(nb_steps):
            obs, rew, done, info = self.step(obs=obs, test=test, collect_samples=False)
            if done:
                obs = self.reset(collect_samples=False)
        print_with_timestamp(f"Benchmark results:\n{self.env.benchmarks()}")

    def send_and_clear_buffer(self):
        """
        Sends the buffered samples to the `Server`.
        """
        self.__buffer_lock.acquire()  # BUFFER LOCK.....................................................................
        self.__buffer += self.buffer
        self.__buffer_lock.release()  # END BUFFER LOCK.................................................................
        self.buffer.clear()

    def update_actor_weights(self):
        """
        Updates the actor with new weights received from the `Server` when available.
        """
        self.__weights_lock.acquire()  # WEIGHTS LOCK...................................................................
        if self.__weights is not None:  # new weights available
            with open(self.model_path, 'wb') as f:
                f.write(self.__weights)
            if self.model_history:
                self._cur_hist_cpt += 1
                if self._cur_hist_cpt == self.model_history:
                    x = datetime.datetime.now()
                    with open(self.model_path_history + str(x.strftime("%d_%m_%Y_%H_%M_%S")) + ".pth", 'wb') as f:
                        f.write(self.__weights)
                    self._cur_hist_cpt = 0
                    print_with_timestamp("model weights saved in history")
            self.actor.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print_with_timestamp("model weights have been updated")
            self.__weights = None
        self.__weights_lock.release()  # END WEIGHTS LOCK...............................................................

rw = RolloutWorker(env_cls=ENV_CLS,
                    actor_module_cls=POLICY,
                    sample_compressor=None,
                    device='cuda' if cfg.PRAGMA_CUDA_INFERENCE else 'cpu',
                    server_ip=cfg.SERVER_IP_FOR_WORKER,
                    min_samples_per_worker_packet=1000 if not cfg.CRC_DEBUG else cfg.CRC_DEBUG_SAMPLES,
                    max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
                    model_path=MODEL_PATH_WORKER,
                    obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
                    crc_debug=cfg.CRC_DEBUG,
                    standalone=args.test)

rw.run()