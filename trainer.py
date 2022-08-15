import os
from argparse import ArgumentParser
import numpy as np
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.custom.custom_algorithms import SpinupSacAgent as SAC_Agent
from tmrl.custom.custom_memories import MemoryTM, last_true_in_list, replace_hist_before_done
from tmrl.custom.custom_models import MLPActorCritic
from tmrl.networking import Buffer, print_with_timestamp, get_connected_socket, select_and_send_or_close_socket, poll_and_recv_or_close_socket
from tmrl.util import partial
import pathlib
from common import ENV_CLS
from threading import Lock, Thread
from tmrl.actor import ActorModule
from tmrl.util import pandas_dict
from dataclasses import dataclass
import torch
from pandas import DataFrame
import socket
import tempfile
import atexit
import logging
import shutil
import time
import json
from copy import deepcopy
from os.path import exists
from requests import get

from function import *

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
    env_cls: type = None  # = GenericGymEnv  # dummy environment, used only to retrieve observation and action spaces if needed
    memory_cls: type = None  # = MemoryDataloading  # replay memory
    training_agent_cls: type = None  # = TrainingAgent  # training agent
    epochs: int = 10  # total number of epochs, we save the agent every epoch
    rounds: int = 50  # number of rounds per epoch, we generate statistics every round
    steps: int = 2000  # number of training steps per round
    update_model_interval: int = 100  # number of training steps between model broadcasts
    update_buffer_interval: int = 100  # number of training steps between retrieving buffered samples
    max_training_steps_per_env_step: float = 1.0  # training will pause when above this ratio
    sleep_between_buffer_retrieval_attempts: float = 0.1  # algorithm will sleep for this amount of time when waiting for needed incoming samples
    profiling: bool = False  # if True, run_epoch will be profiled and the profiling will be printed at the end of each epoch
    agent_scheduler: callable = None  # if not None, must be of the form f(Agent, epoch), called at the beginning of each epoch
    start_training: int = 0  # minimum number of samples in the replay buffer before starting training
    device: str = None  # device on which the model of the TrainingAgent will live (None for automatic)

    total_updates = 0

    def __post_init__(self):
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = 0
        self.memory = self.memory_cls(nb_steps=self.steps, device=device)
        if type(self.env_cls) == tuple:
            observation_space, action_space = self.env_cls
        else:
            with self.env_cls() as env:
                observation_space, action_space = env.observation_space, env.action_space
                print(observation_space)
        self.agent = self.training_agent_cls(observation_space=observation_space,
                                             action_space=action_space,
                                             device=device)
        self.total_samples = len(self.memory)
        logging.info(f" Initial total_samples:{self.total_samples}")

    def update_buffer(self, interface):
        buffer = interface.retrieve_buffer()
        self.memory.append(buffer)
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

class Trainer:
    """
    Training entity.
    The `Trainer` object is where RL training happens.
    Typically, it can be located on a HPC cluster.
    """
    def __init__(self,
                 training_cls=cfg_obj.TRAINER,
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

SAMPLE_PREPROCESSOR = None

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

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

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
            replace_hist_before_done(hist=imgs_new_obs, done_idx_in_hist=last_done_idx - self.start_imgs_offset - 1)
            replace_hist_before_done(hist=imgs_last_obs, done_idx_in_hist=last_done_idx - self.start_imgs_offset)

        imgs_new_obs = np.ndarray.flatten(imgs_new_obs)
        imgs_last_obs = np.ndarray.flatten(imgs_last_obs)

        last_obs = (self.data[2][idx_last], imgs_last_obs, *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[6][idx_now])
        new_obs = (self.data[2][idx_now], imgs_new_obs, *new_act_buf)
        done = self.data[5][idx_now]
        info = self.data[7][idx_now]
        return last_obs, new_act, rew, new_obs, done, info

    def load_imgs(self, item):
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
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
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

MEM = MemoryTMNFCustom

TMRL_FOLDER = pathlib.Path("D:/GitHub/trackmania-ia-with-tmrl") / "TmrlData"

RUN_NAME = 'SAC_4_Custom'

WEIGHTS_FOLDER = TMRL_FOLDER / "weights"

MODEL_PATH_TRAINER = str(WEIGHTS_FOLDER / (RUN_NAME + "_t.pth"))
DATASET_PATH = str(TMRL_FOLDER / "dataset")

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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--no-wandb', dest='no_wandb', action='store_true', help='(use with --trainer) if you do not want to log results on Weights and Biases, use this option')

    args = parser.parse_args()
    trainer = Trainer(training_cls=TRAINER,
                        server_ip="192.168.0.181",
                        model_path=MODEL_PATH_TRAINER,
                        checkpoint_path=cfg.CHECKPOINT_PATH,
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