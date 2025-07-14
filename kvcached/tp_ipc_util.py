import pickle
import threading
import time

import zmq

from kvcached.vmm_ops import map_to_kv_tensors, unmap_from_kv_tensors

# SOCKET_DIR = "/tmp/kvcached-ipc"
ZMQ_PUB_ENDPOINT = "ipc:///tmp/kvcached_ipc_pub"
ZMQ_TOPIC = b"kvcached"

_kvcached_ctx = zmq.Context.instance()
_kvcached_pub_socket = None

# def get_worker_socket_path(rank: int) -> str:
#     """
#     Get the path for the worker socket.
#     The socket is used for inter-process communication.
#     """
#     return os.path.join(SOCKET_DIR, f"worker_{rank}.sock")

# def send_msg(sock: socket.socket, msg: object) -> None:
#     """
#     Send a message through the socket.
#     The message is serialized using pickle.
#     """
#     data = pickle.dumps(msg)
#     sock.sendall(len(data).to_bytes(4, 'big') + data)

# def recv_msg(sock: socket.socket) -> object:
#     """
#     Receive a message from the socket.
#     The message is deserialized using pickle.
#     """
#     length_bytes = sock.recv(4)
#     if not length_bytes:
#         raise ConnectionError("Socket connection closed")
#     if not len(length_bytes) == 4:
#         raise ValueError("Received incomplete length bytes from socket")
#     length = int.from_bytes(length_bytes, 'big')
#     if length <= 0:
#         raise ValueError("Received invalid length for message")
#     data = b""
#     while len(data) < length:
#         chunk = sock.recv(length - len(data))
#         if not chunk:
#             raise ConnectionError(
#                 "Socket connection closed while receiving data")
#         data += chunk
#     if len(data) != length:
#         raise ValueError("Received data length does not match expected length")
#     return pickle.loads(data)


def init_publisher():
    global _kvcached_pub_socket
    if _kvcached_pub_socket is None:
        _kvcached_pub_socket = _kvcached_ctx.socket(zmq.PUB)
        _kvcached_pub_socket.bind(ZMQ_PUB_ENDPOINT)
        # Delay to allow subscribers time to connect
        time.sleep(0.1)


def start_worker_subscriber():

    def listen_loop():
        ctx = zmq.Context().instance()
        sub = ctx.socket(zmq.SUB)
        sub.connect(ZMQ_PUB_ENDPOINT)
        sub.setsockopt(zmq.SUBSCRIBE, ZMQ_TOPIC)

        print("[Worker] IPC subscriber listening")
        while True:
            try:
                topic, data = sub.recv_multipart()
                msg = pickle.loads(data)
                # print(f"[Worker] Received pub msg: {msg}")
                if msg["cmd"] == "map_to_kv_tensors":
                    map_to_kv_tensors(msg["offsets"])
                elif msg["cmd"] == "unmap_from_kv_tensors":
                    unmap_from_kv_tensors(msg["offsets"])
                else:
                    print(f"[Worker] Unknown cmd: {msg['cmd']}")
            except Exception as e:
                print(f"[Worker] Error in subscriber loop: {e}")

    threading.Thread(target=listen_loop, daemon=True).start()


# def start_worker_listerner_thread(rank: int):
#     """
#     Start a thread that listens for messages on the worker socket.
#     The callback is called with the received message.
#     """
#     os.makedirs(SOCKET_DIR, exist_ok=True)
#     socket_path = get_worker_socket_path(rank)

#     if os.path.exists(socket_path):
#         try:
#             os.remove(socket_path)
#         except OSError as e:
#             print(f"Error removing existing socket file {socket_path}: {e}")

#     server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
#     server_sock.bind(socket_path)
#     server_sock.listen()

#     def listen_loop():
#         print(f"Worker {rank} IPC listener started at {socket_path}")
#         while True:
#             conn, _ = server_sock.accept()
#             try:
#                 msg = recv_msg(conn)
#                 # print(f"Worker {rank} received message: {msg}")
#                 if msg["cmd"] == "map_to_kv_tensors":
#                     map_to_kv_tensors(msg["offsets"])
#                     send_msg(conn, {"status": "success"})
#                 elif msg["cmd"] == "unmap_from_kv_tensors":
#                     unmap_from_kv_tensors(msg["offsets"])
#                     send_msg(conn, {"status": "success"})
#                 else:
#                     send_msg(conn, {
#                         "status": "error",
#                         "message": "Unknown command"
#                     })
#             except Exception as e:
#                 print(f"Worker {rank} error processing message: {e}")
#                 send_msg(conn, {"status": "error", "message": str(e)})
#             finally:
#                 conn.close()

#     t = threading.Thread(target=listen_loop, daemon=True)
#     t.start()


def send_pub_message(cmd: str, offsets: list[int]) -> None:
    if _kvcached_pub_socket is None:
        raise RuntimeError(
            "Publisher socket is not initialized. Call init_publisher() first."
        )

    msg = pickle.dumps({"cmd": cmd, "offsets": offsets})
    _kvcached_pub_socket.send_multipart([ZMQ_TOPIC, msg])


# def broadcast_map_to_kv_tensors_to_workers(tp_size: int,
#                                            offsets: list[int]) -> None:
#     for rank in range(tp_size):
#         socket_path = get_worker_socket_path(rank)
#         sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
#         sock.connect(socket_path)
#         try:
#             send_msg(sock, {"cmd": "map_to_kv_tensors", "offsets": offsets})
#             response = recv_msg(sock)
#             if response.get("status") != "success":
#                 raise RuntimeError(f"Worker {rank} failed to map: {response}")
#         finally:
#             sock.close()

# def broadcast_unmap_from_kv_tensors_to_workers(tp_size: int,
#                                                offsets: list[int]) -> None:
#     for rank in range(tp_size):
#         socket_path = get_worker_socket_path(rank)
#         sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
#         sock.connect(socket_path)
#         try:
#             send_msg(sock, {
#                 "cmd": "unmap_from_kv_tensors",
#                 "offsets": offsets
#             })
#             response = recv_msg(sock)
#             if response.get("status") != "success":
#                 raise RuntimeError(f"Worker {rank} failed to unmap {response}")
#         finally:
#             sock.close()
