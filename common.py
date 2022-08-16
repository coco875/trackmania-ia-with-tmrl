# TMRL import
from tmrl.custom.utils.compute_reward import RewardFunction
from tmrl.custom.utils.control_gamepad import (
    control_gamepad, gamepad_close_finish_pop_up_tm20, gamepad_reset)
from tmrl.custom.utils.control_keyboard import apply_control, keyres
from tmrl.custom.utils.control_mouse import mouse_close_finish_pop_up_tm20, mouse_save_replay_tm20
from tmrl.custom.utils.tools import TM2020OpenPlanetClient
from tmrl.util import partial
from tmrl.actor import ActorModule
from tmrl.wrappers import AffineObservationWrapper
import tmrl.config.config_constants as cfg

#pygbx import
from pygbx import Gbx, GbxType
from pygbx.headers import MapBlock, CGameChallenge, Vector3

#gym import
import gym
import gym.spaces as spaces

#rtgym import
import rtgym
from rtgym import RealTimeGymInterface

#built in import
import logging
from collections import deque
import time
import platform
from typing import Mapping, Sequence
import functools

# torch import
import torch.nn as nn
import torch
from torch.distributions.normal import Normal
import torch.nn.functional as F

import numpy as np

def dict_to_list(l):
    ll = []
    for i in l:
        if type(i) == dict:
            ll += dict_to_list(list(i.values()))
        elif type(i) == Vector3:
            ll += dict_to_list(list(i.__dict__.values()))
        elif type(i) == list:
            ll += dict_to_list(i)
        else:
            ll.append(i)
    return ll

def collate(batch, device=None):
    """Turns a batch of nested structures with numpy arrays as leaves into into a single element of the same nested structure with batched torch tensors as leaves"""
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        # return torch.stack(batch, 0).to(device, non_blocking=non_blocking)
        if elem.numel() < 20000:  # TODO: link to the relevant profiling that lead to this threshold
            return torch.stack(batch).to(device)
        else:
            return torch.stack([b.contiguous().to(device) for b in batch], 0)
    elif isinstance(elem, np.ndarray):
        for b in batch:
            if type(b[0]) != dict:
                return collate(tuple(torch.from_numpy(b)))
            else:
                l = dict_to_list(b)
                l = np.array(l, np.float32)
                l[l >= 1E308] = 1E30
                return collate(tuple(torch.from_numpy(l)))
        # return collate(tuple(torch.from_numpy(b) if type(b[0]) != dict else torch.from_numpy(np.array(i.values() for i in b)) for b in batch), device)
    elif hasattr(elem, '__torch_tensor__'):
        return torch.stack([b.__torch_tensor__().to(device) for b in batch], 0)
    elif isinstance(elem, Sequence):
        transposed = zip(*batch)
        return type(elem)(collate(samples, device) for samples in transposed)
    elif isinstance(elem, Mapping):
        return type(elem)((key, collate(tuple(d[key] for d in batch), device)) for key in elem)
    else:
        l = np.array(batch, np.float32)
        l[l >= 1E308] = 1E30
        return torch.from_numpy(l).to(device)  # we create a numpy array first to work around https://github.com/pytorch/pytorch/issues/24200

def deepmap(f, m):
    """Apply functions to the leaves of a dictionary or list, depending type of the leaf value.
    Example: deepmap({torch.Tensor: lambda t: t.detach()}, x)."""
    for cls in f:
        if isinstance(m, cls):
            return f[cls](m)
    if isinstance(m, Sequence):
        return type(m)(deepmap(f, x) for x in m)
    elif isinstance(m, Mapping):
        return type(m)((k, deepmap(f, m[k])) for k in m)
    else:
        raise AttributeError(f"m is a {type(m)}, not a Sequence nor a Mapping: {m}")

def float64_to_float32(x):
    return np.asarray([x, ], np.float32) if x.dtype == np.float64 else x


def float_to_float32(x: float):
    return np.asarray([x, ], np.float32)

def int_to_float32(x: int):
    return np.asarray([x, ], np.float32)

def Vector3_to_float32(v: Vector3):
    return np.asarray([v.x, v.y, v.z, ], np.float32)

def dict_to_float32(d: dict):
    return np.array(list(d.values()), np.float32)

def mul(a,b) -> int:
    if type(a) == spaces.Dict:
        n = 0
        for i in a:
            n += prod(a[i].shape)
        a = n
    
    if type(b) == spaces.Dict:
        n = 0
        for i in b:
            n += prod(b[i].shape)
        b = n
    
    return a*b

def prod(iterable: tuple[int]) -> int:
    return functools.reduce(mul, iterable, 1) # need to be fix with a dict

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)
class MLPQFunction(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim = sum(prod(s for s in space.shape) for space in obs_space)
        # obs_dim = 315
        act_dim = act_space.shape[0]
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        x = torch.cat((*obs, act), -1)
        q = self.q(x)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.

class SquashedGaussianMLPActor(ActorModule):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU, act_buf_len=0):
        super().__init__(observation_space, action_space)
        dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
        print(dim_obs)
        # dim_obs = 499
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self.net = mlp([dim_obs] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        data = torch.cat(obs, -1)
        np.ma.masked_array(data, ~np.isfinite(data)).filled(0)
        net_out = self.net(data)

        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if test:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        pi_action = pi_action.squeeze()

        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.numpy()

POLICY = partial(SquashedGaussianMLPActor, act_buf_len=cfg.ACT_BUF_LEN)

class Float64ToFloat32(gym.ObservationWrapper):
    """Converts np.float64 arrays in the observations to np.float32 arrays."""

    # TODO: change observation/action spaces to correct dtype
    def observation(self, observation):
        observation = deepmap({np.ndarray: float64_to_float32,
                               float: float_to_float32,
                               np.float32: float_to_float32,
                               np.float64: float_to_float32,
                               int: int_to_float32,
                               dict: dict_to_float32,
                               Vector3: Vector3_to_float32}, observation)
        #print(observation)
        return observation

    def step(self, action):
        s, r, d, info = super().step(action)
        return s, r, d, info

class GenericGymEnv(gym.Wrapper):
    def __init__(self, id: str = "Pendulum-v0", obs_scale: float = 0., gym_kwargs={}):
        """
        Use this wrapper when using the framework with arbitrary environments.

        Args:
            id (str): gym id
            obs_scale (float): change this if wanting to rescale actions by a scalar
            gym_kwargs (dict): keyword arguments of the gym environment (i.e. between -1.0 and 1.0 when the actual action space is something else)
        """
        env = gym.make(id, **gym_kwargs, disable_env_checker=True)
        if obs_scale:
            env = AffineObservationWrapper(env, 0, obs_scale)
        env = Float64ToFloat32(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        # env = NormalizeActionWrapper(env)
        super().__init__(env)

# Globals ==============================================================================================================

NB_OBS_FORWARD = 500  # this allows (and rewards) 50m cuts

# Interface for Trackmania 2020 ========================================================================================

class Buffer:
    """
    Buffer of training samples.

    `Server`, `RolloutWorker` and `Trainer` all have their own `Buffer` to store and send training samples.

    Samples are tuples of the form (`act`, `new_obs`, `rew`, `done`, `info`)
    """
    def __init__(self, maxlen=cfg.BUFFERS_MAXLEN):
        """
        Args:
            maxlen (int): buffer length
        """
        self.memory = []
        self.stat_train_return = 0.0  # stores the train return
        self.stat_test_return = 0.0  # stores the test return
        self.stat_train_steps = 0  # stores the number of steps per training episode
        self.stat_test_steps = 0  # stores the number of steps per test episode
        self.maxlen = maxlen

    def clip_to_maxlen(self):
        lenmem = len(self.memory)
        if lenmem > self.maxlen:
            print_with_timestamp("buffer overflow. Discarding old samples.")
            self.memory = self.memory[(lenmem - self.maxlen):]

    def append_sample(self, sample):
        """
        Appends `sample` to the buffer.

        Args:
            sample (Tuple): a training sample of the form (`act`, `new_obs`, `rew`, `done`, `info`)
        """
        self.memory.append(sample)
        self.clip_to_maxlen()

    def clear(self):
        """
        Clears the buffer but keeps train and test returns.
        """
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def __iadd__(self, other):
        for i in other.memory:
            self.memory.append(i)
        self.clip_to_maxlen()
        self.stat_train_return = other.stat_train_return
        self.stat_test_return = other.stat_test_return
        self.stat_train_steps = other.stat_train_steps
        self.stat_test_steps = other.stat_test_steps
        return self

class TM2020InterfaceCustom(RealTimeGymInterface):
    """
    This is the API needed for the algorithm to control Trackmania2020
    """
    def __init__(self, track_name:str, pos_hist_len:int = 1, gamepad:bool = False, min_nb_steps_before_early_done:int = int(20 * 3.5), record:bool = False, save_replay:bool = False):
        self.track_name = track_name
        self.last_time = None
        # self.digits = None
        self.pos_hist_len = pos_hist_len
        self.pos_hist = None
        self.img = None
        self.reward_function = None
        self.client = None
        self.gamepad = gamepad
        self.j = None
        # self.window_interface = None
        self.small_window = None
        self.min_nb_steps_before_early_done = min_nb_steps_before_early_done
        self.save_replay = save_replay

        self.initialized = False
        self.record = record
        self.track = None
        self.max_track_element = 100

    def initialize_common(self):
        if self.gamepad:
            assert platform.system() == "Windows", "Sorry, Only Windows is supported for gamepad control"
            import vgamepad as vg
            self.j = vg.VX360Gamepad()
            logging.debug(" virtual joystick in use")
        # self.window_interface = WindowInterface("Trackmania")
        # self.window_interface.move_and_resize()
        self.last_time = time.time()
        # self.digits = load_digits()
        self.pos_hist = deque(maxlen=self.pos_hist_len)
        self.img = None
        self.reward_function = RewardFunction(reward_data_path=cfg.REWARD_PATH,
                                              nb_obs_forward=NB_OBS_FORWARD,
                                              nb_obs_backward=10,
                                              nb_zero_rew_before_early_done=10,
                                              min_nb_steps_before_early_done=self.min_nb_steps_before_early_done)
        self.client = TM2020OpenPlanetClient()

        self.track = Gbx(self.track_name)
        b = self.track.find_raw_chunk_id(0x0304301F)
        b.pos -= 4
        b.seen_loopback = True
        self.track._read_node(GbxType.CHALLENGE, -1, b)

    def send_control(self, control):
        """
        Non-blocking function
        Applies the action given by the RL policy
        If control is None, does nothing (e.g. to record)
        Args:
            control: np.array: [forward,backward,right,left]
        """
        if self.gamepad:
            if control is not None:
                control_gamepad(self.j, control)
        else:
            if control is not None:
                actions = []
                if control[0] > 0:
                    actions.append('f')
                if control[1] > 0:
                    actions.append('b')
                if control[2] > 0.5:
                    actions.append('r')
                elif control[2] < -0.5:
                    actions.append('l')
                apply_control(actions)

    def reset_race(self):
        if self.gamepad:
            gamepad_reset(self.j)
        else:
            keyres()

    def reset_common(self):
        if not self.initialized:
            self.initialize()
        self.send_control(self.get_default_action())
        self.reset_race()
        time_sleep = max(0, cfg.SLEEP_TIME_AT_RESET - 0.1) if self.gamepad else cfg.SLEEP_TIME_AT_RESET
        time.sleep(time_sleep)  # must be long enough for image to be refreshed

    def close_finish_pop_up_tm20(self):
        if self.gamepad:
            gamepad_close_finish_pop_up_tm20(self.j)
        else:
            mouse_close_finish_pop_up_tm20(small_window=self.small_window)

    def wait(self):
        """
        Non-blocking function
        The agent stays 'paused', waiting in position
        """
        self.send_control(self.get_default_action())
        self.reset_race()
        time.sleep(0.5)
        self.close_finish_pop_up_tm20()

    def get_action_space(self):
        """
        must return a Box
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(3, ))

    def get_default_action(self):
        """
        initial action at episode start
        """
        return np.array([0.0, 0.0, 0.0], dtype='float32')

    def grab_data_and_track(self):
        data = self.client.retrieve_data()
        challenge:CGameChallenge = self.track.get_class_by_id(GbxType.CHALLENGE)
        block = challenge.blocks
        return data, block

    def initialize(self):
        self.initialize_common()
        self.small_window = False
        self.initialized = True

    def reset(self):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        data, track = self.grab_data_and_track()
        speed = np.array([
            data[0],
        ], dtype='float32')
        pos = [data[2], data[3], data[4]]
        pos = np.array(pos)
        for _ in range(self.pos_hist_len):
            self.pos_hist.append(pos)
        poss = np.array(list(self.pos_hist), dtype='float32')

        block = deque(maxlen=self.max_track_element)
        for i in track:
            block.append({
                "name": int(i.name, base=36) if i.name != "" else 0,
                "rotation": i.rotation,
                "position": i.position,
                "speed": i.speed,
                "flags": i.flags,
                "skin": i.skin,
            })
        
        for i in range(self.max_track_element-len(block)):
            block.append({
                "name": 0,
                "rotation": 0,
                "position": {"x":0, "y":0, "z":0},
                "speed": 0,
                "flags": 0,
                "skin": 0,
            })
        obs = [speed, poss, np.array(block)]
        self.reward_function.reset()
        return obs  # if not self.record else data

    def get_obs_rew_done_info(self):
        """
        returns the observation, the reward, and a done signal for end of episode
        obs must be a list of numpy arrays
        """
        data, track = self.grab_data_and_track()

        pos = [data[2], data[3], data[4]]
        pos = np.array(pos)
        rew, done = self.reward_function.compute_reward(pos=pos)
        rew = np.float32(rew)

        self.pos_hist.append(pos)
        speed = np.array([
            data[0],
        ], dtype='float32')
        poss = np.array(list(self.pos_hist), dtype='float32')
        block = deque(maxlen=self.max_track_element)
        for i in track:
            block.append({
                "name": int(i.name, base=36) if i.name != "" else 0,
                "rotation": i.rotation,
                "position": i.position,
                "speed": i.speed,
                "flags": i.flags,
                "skin": i.skin,
            })
        
        for i in range(self.max_track_element-len(block)):
            block.append({
                "name": 0,
                "rotation": 0,
                "position": {"x":0, "y":0, "z":0},
                "speed": 0,
                "flags": 0,
                "skin": 0,
            })
        obs = [speed, poss, np.array(block)]
        end_of_track = bool(data[8])
        info = {}
        if end_of_track:
            rew += cfg.REWARD_END_OF_TRACK
            done = True
            info["__no_done"] = True
            if self.save_replay:
                mouse_save_replay_tm20()
        rew += cfg.CONSTANT_PENALTY
        return obs, rew, done, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        poss = spaces.Box(low=0.0, high=999999999, shape=(
            self.pos_hist_len,
            3,
        ))  # historic of position
        track = {
            "name": spaces.Box(0, 99999999, (1, ), np.uint64),
            "rotation": spaces.Box(0, 6, (1, ), np.int32),
            "position": spaces.Box(-99999999, 99999999, (3, ), np.int64),
            "speed": spaces.Box(-99999999, 99999999, (1, ), np.int64),
            "flags": spaces.Box(0, 8000000000, (1, ), np.uint64),
            "skin": spaces.Box(0, 8000000000, (1, ), np.uint64),
        }
        
        return spaces.Tuple((speed, poss, spaces.Space((self.max_track_element, spaces.Dict(track)))))

INT = partial(TM2020InterfaceCustom, "C:\\Users\\Corentin\\Documents\\Trackmania2020\\Maps\\My Maps\\tmrl-train.Map.Gbx", pos_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD)

CONFIG_DICT = rtgym.DEFAULT_CONFIG_DICT.copy()
CONFIG_DICT["interface"] = INT
CONFIG_DICT_MODIFIERS = cfg.ENV_CONFIG["RTGYM_CONFIG"]
for k, v in CONFIG_DICT_MODIFIERS.items():
    CONFIG_DICT[k] = v

ENV_CLS = partial(GenericGymEnv, id="real-time-gym-v0", gym_kwargs={"config": CONFIG_DICT})