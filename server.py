import time

from tmrl.networking import Buffer, print_with_timestamp, get_listening_socket, accept_or_close_socket, select_and_send_or_close_socket
import tmrl.config.config_constants as cfg
from threading import Lock, Thread
import socket
import select
import pickle
from requests import get

def poll_and_recv_or_close_socket(conn):
    """
    Returns True, obj is success (obj is None if nothing was in the read buffer when polling)
    False, None otherwise
    """
    rl, _, xl = select.select([conn], [], [conn], 0.0)  # polling read channel
    if len(xl) != 0:
        print_with_timestamp("error when polling, closing sockets")
        conn.close()
        return False, None
    if len(rl) == 0:  # nothing in the recv buffer
        return True, None
    obj = recv_object(conn)
    if obj is None:  # socket error
        print_with_timestamp("error when receiving object, closing sockets")
        conn.close()
        return False, None
    elif obj == 'PINGPONG':
        return True, None
    else:
        return True, obj

def send_ack(sock):
    return send_object(sock, None, ping=False, pong=False, ack=True)

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

def recv_object(sock):
    """
    If the request is PING or PONG, this will return 'PINGPONG'
    If the request is ACK, this will return 'ACK'
    If the request is PING, this will automatically send the PONG answer
    Call only after select on a socket with a (long enough) timeout.
    Returns the object if received successfully, None if connection lost.
    This sends the ACK request back to sock when an object transfer is complete
    """
    # first, we receive the header (inefficient but prevents collisions)
    msg = b''
    l = len(msg)
    while l != cfg.HEADER_SIZE:
        try:
            recv_msg = sock.recv(cfg.HEADER_SIZE - l)
            print(len(recv_msg))
            if len(recv_msg) == 0:  # connection closed or broken
                return None
            msg += recv_msg
        except OSError:  # connection closed or broken
            return None
        l = len(msg)
    if msg[:3] == b'ACK':
        return 'ACK'
    msglen = int(msg[:cfg.HEADER_SIZE])
    # now, we receive the actual data (no more than the data length, again to prevent collisions)
    msg = b''
    l = len(msg)
    while l != msglen:
        try:
            recv_msg = sock.recv(min(cfg.BUFFER_SIZE, msglen - l))  # this will not receive more bytes than required
            if len(recv_msg) == 0:  # connection closed or broken
                return None
            msg += recv_msg
        except OSError:  # connection closed or broken
            return None
        l = len(msg)
    send_ack(sock)
    return pickle.loads(msg)

class Server:
    """
    Central server.

    The `Server` lets 1 `Trainer` and n `RolloutWorkers` connect.
    It buffers experiences sent by workers and periodically sends these to the trainer.
    It also receives the weights from the trainer and broadcasts these to the connected workers.
    """
    def __init__(self, min_samples_per_server_packet=1):
        """
        Args:
            min_samples_per_server_packet (int): Minimum number of samples that the
                server buffers from connected workers before sending to the trainer.
        """
        self.__buffer = Buffer()
        self.__buffer_lock = Lock()
        self.__weights_lock = Lock()
        self.__weights = None
        self.__weights_id = 0  # this increments each time new weights are received
        self.samples_per_server_batch = min_samples_per_server_packet
        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())

        print_with_timestamp(f"INFO SERVER: local IP: {self.local_ip}")
        print_with_timestamp(f"INFO SERVER: public IP: {self.public_ip}")

        Thread(target=self.__rollout_workers_thread, args=('', ), kwargs={}, daemon=True).start()
        Thread(target=self.__trainers_thread, args=('', ), kwargs={}, daemon=True).start()

    def __trainers_thread(self, ip):
        """
        This waits for new potential Trainers to connect
        When a new Trainer connects, this instantiates a new thread to handle it
        """
        while True:  # main server loop
            s = get_listening_socket(cfg.SOCKET_TIMEOUT_ACCEPT_TRAINER, ip, cfg.PORT_TRAINER)
            conn, addr = accept_or_close_socket(s)
            if conn is None:
                continue
            print_with_timestamp(f"INFO TRAINERS THREAD: server connected by trainer at address {addr}")
            Thread(target=self.__trainer_thread, args=(conn, ), kwargs={}, daemon=True).start()  # we don't keep track of this for now
            s.close()

    def __trainer_thread(self, conn):
        """
        This periodically sends the local buffer to the TrainerInterface (when data is available)
        When the TrainerInterface sends new weights, this broadcasts them to all connected RolloutWorkers
        """
        ack_time = time.time()
        wait_ack = False
        while True:
            # send samples
            self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
            if len(self.__buffer) >= self.samples_per_server_batch:
                if not wait_ack:
                    obj = self.__buffer
                    if select_and_send_or_close_socket(obj, conn):
                        wait_ack = True
                        ack_time = time.time()
                    else:
                        print_with_timestamp("failed sending object to trainer")
                        self.__buffer_lock.release()
                        break
                    self.__buffer.clear()
                else:
                    elapsed = time.time() - ack_time
                    print_with_timestamp(f"CAUTION: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                    if elapsed >= cfg.ACK_TIMEOUT_SERVER_TO_TRAINER:
                        print_with_timestamp("ACK timed-out, breaking connection")
                        self.__buffer_lock.release()
                        wait_ack = False
                        break
            self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
            # checks for weights
            success, obj = poll_and_recv_or_close_socket(conn)
            if not success:
                print_with_timestamp("poll failed in trainer thread")
                break
            elif obj is not None and obj != 'ACK':
                print_with_timestamp(f"trainer thread received obj")
                self.__weights_lock.acquire()  # WEIGHTS LOCK.......................................................
                self.__weights = obj
                self.__weights_id += 1
                self.__weights_lock.release()  # END WEIGHTS LOCK...................................................
            elif obj == 'ACK':
                wait_ack = False
                print_with_timestamp(f"transfer acknowledgment received after {time.time() - ack_time}s")
            time.sleep(cfg.LOOP_SLEEP_TIME)  # TODO: adapt

    def __rollout_workers_thread(self, ip):
        """
        This waits for new potential RolloutWorkers to connect
        When a new RolloutWorker connects, this instantiates a new thread to handle it
        """
        while True:  # main server loop
            s = get_listening_socket(cfg.SOCKET_TIMEOUT_ACCEPT_ROLLOUT, ip, cfg.PORT_ROLLOUT)
            conn, addr = accept_or_close_socket(s)
            if conn is None:
                continue
            print_with_timestamp(f"INFO WORKERS THREAD: server connected by worker at address {addr}")
            Thread(target=self.__rollout_worker_thread, args=(conn, ), kwargs={}, daemon=True).start()  # we don't keep track of this for now
            s.close()

    def __rollout_worker_thread(self, conn):
        """
        Thread handling connection to a single RolloutWorker
        """
        # last_ping = time.time()
        worker_weights_id = 0
        ack_time = time.time()
        wait_ack = False
        while True:
            # send weights
            self.__weights_lock.acquire()  # WEIGHTS LOCK...............................................................
            if worker_weights_id != self.__weights_id:  # new weigths
                if not wait_ack:
                    obj = self.__weights
                    if select_and_send_or_close_socket(obj, conn):
                        ack_time = time.time()
                        wait_ack = True
                    else:
                        self.__weights_lock.release()
                        print_with_timestamp("select_and_send_or_close_socket failed in worker thread")
                        break
                    worker_weights_id = self.__weights_id
                else:
                    elapsed = time.time() - ack_time
                    print_with_timestamp(f"object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                    if elapsed >= cfg.ACK_TIMEOUT_SERVER_TO_WORKER:
                        print_with_timestamp("ACK timed-out, breaking connection")
                        self.__weights_lock.release()
                        # wait_ack = False  # not needed since we end the thread
                        break
            self.__weights_lock.release()  # END WEIGHTS LOCK...........................................................
            # checks for samples
            success, obj = poll_and_recv_or_close_socket(conn)
            if not success:
                print_with_timestamp("poll failed in rollout thread")
                break
            elif obj is not None and obj != 'ACK':
                print_with_timestamp(f"rollout worker thread received obj")
                self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
                self.__buffer += obj  # concat worker batch to local batch
                self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
            elif obj == 'ACK':
                wait_ack = False
                print_with_timestamp(f"transfer acknowledgment received after {time.time() - ack_time}s")
            time.sleep(cfg.LOOP_SLEEP_TIME)  # TODO: adapt

Server(min_samples_per_server_packet=1000)
while True:
    time.sleep(1.0)