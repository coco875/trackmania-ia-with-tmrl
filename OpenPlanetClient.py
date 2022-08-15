import socket # main a custom to get track without the file but not now I don't have clud edition
import struct
import time
from threading import Lock, Thread


class TM2020OpenPlanetClient:
    def __init__(self, host='127.0.0.1', port=9000, nb_floats_to_unpack=11):
        # Script attributes:
        self.nb_floats_to_unpack = nb_floats_to_unpack
        self._nb_bytes = self.nb_floats_to_unpack * 4
        self._struct_str = '<' + 'f' * self.nb_floats_to_unpack
        self._host = host
        self._port = port

        # Threading attributes:
        self.__lock = Lock()
        self.__data = None
        self.__t_client = Thread(target=self.__client_thread, args=(), kwargs={}, daemon=True)
        self.__t_client.start()

    def __client_thread(self):
        """
        Thread of the client.
        This listens for incoming data until the object is destroyed
        TODO: handle disconnection
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self._host, self._port))
            data_raw = b''
            while True:  # main loop
                while len(data_raw) < self._nb_bytes:
                    data_raw += s.recv(1024)
                div = len(data_raw) // self._nb_bytes
                data_used = data_raw[(div - 1) * self._nb_bytes:div * self._nb_bytes]
                data_raw = data_raw[div * self._nb_bytes:]
                self.__lock.acquire()
                self.__data = data_used
                self.__lock.release()

    def retrieve_data(self, sleep_if_empty=0.1):
        """
        Retrieves the most recently received data
        Use this function to retrieve the most recently received data
        If block if nothing has been received so far
        """
        c = True
        while c:
            self.__lock.acquire()
            if self.__data is not None:
                data = struct.unpack(self._struct_str, self.__data)
                print(data)
                c = False
            self.__lock.release()
            if c:
                time.sleep(sleep_if_empty)
        return data

t= TM2020OpenPlanetClient()
while True:
    time.sleep(1.0)
    t.retrieve_data()