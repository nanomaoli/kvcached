import os
import pickle
import socket
import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from kvcached.vmm_ops import (kv_tensors_created, map_to_kv_tensors,
                              unmap_from_kv_tensors)

SOCKET_DIR = "/tmp/kvcached-ipc"


def get_worker_socket_path(rank: int) -> str:
    """
    Get the path for the worker socket.
    The socket is used for inter-process communication.
    """
    return os.path.join(SOCKET_DIR, f"worker_{rank}.sock")


def send_msg(sock: socket.socket, msg: object) -> None:
    """
    Send a message through the socket.
    The message is serialized using pickle.
    """
    data = pickle.dumps(msg)
    sock.sendall(len(data).to_bytes(4, 'big') + data)


def recv_msg(sock: socket.socket) -> object:
    """
    Receive a message from the socket.
    The message is deserialized using pickle.
    """
    length_bytes = sock.recv(4)
    if not length_bytes:
        raise ConnectionError("Socket connection closed")
    if not len(length_bytes) == 4:
        raise ValueError("Received incomplete length bytes from socket")
    length = int.from_bytes(length_bytes, 'big')
    if length <= 0:
        raise ValueError("Received invalid length for message")
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise ConnectionError(
                "Socket connection closed while receiving data")
        data += chunk
    if len(data) != length:
        raise ValueError("Received data length does not match expected length")
    return pickle.loads(data)


def start_worker_listerner_thread(rank: int):
    """
    Start a thread that listens for messages on the worker socket.
    The callback is called with the received message.
    """
    os.makedirs(SOCKET_DIR, exist_ok=True)
    socket_path = get_worker_socket_path(rank)

    if os.path.exists(socket_path):
        try:
            os.remove(socket_path)
        except OSError as e:
            print(f"Error removing existing socket file {socket_path}: {e}")

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(socket_path)
    server_sock.listen()

    def listen_loop():
        print(f"Worker {rank} IPC listener started at {socket_path}")
        while True:
            conn, _ = server_sock.accept()
            try:
                msg = recv_msg(conn)
                # print(f"Worker {rank} received message: {msg}")
                if msg["cmd"] == "map_to_kv_tensors":
                    start = time.time()
                    map_to_kv_tensors(msg["offsets"])
                    end = time.time()
                    print(f"[kvcached worker {rank}] map_to_kv_tensors: gpu op took {end - start:.6f}s", flush=True)
                    send_msg(conn, {"status": "success"})
                elif msg["cmd"] == "unmap_from_kv_tensors":
                    start = time.time()
                    unmap_from_kv_tensors(msg["offsets"])
                    end = time.time()
                    print(f"[kvcached worker {rank}] unmap_from_kv_tensors: gpu op took {end - start:.6f}s", flush=True)
                    send_msg(conn, {"status": "success"})
                elif msg["cmd"] == "kv_tensors_created":
                    created: bool = kv_tensors_created()
                    send_msg(conn, {"status": "success", "created": created})
                else:
                    send_msg(conn, {
                        "status": "error",
                        "message": "Unknown command"
                    })
            except Exception as e:
                print(f"Worker {rank} error processing message: {e}")
                send_msg(conn, {"status": "error", "message": str(e)})
            finally:
                conn.close()

    t = threading.Thread(target=listen_loop, daemon=True)
    t.start()

def send_map_cmd_to_worker(rank, offsets):
    socket_path = get_worker_socket_path(rank)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(socket_path)
        send_msg(sock, {"cmd": "map_to_kv_tensors", "offsets": offsets})
        response = recv_msg(sock)
        if response.get("status") != "success":
            raise RuntimeError(f"Worker {rank} failed to map: {response}")
    finally:
        sock.close()

def broadcast_map_to_kv_tensors_to_workers(tp_size: int,
                                           offsets: list[int]) -> None:
    start_time = time.time()
    per_rank_times = []
    
    def timed_send(rank):
        t0 = time.time()
        send_map_cmd_to_worker(rank, offsets)
        t1 = time.time()
        return rank, t1 - t0
    
    with ThreadPoolExecutor(max_workers=tp_size) as executor:
        futures = [executor.submit(timed_send, rank) for rank in range(tp_size)]
        for future in as_completed(futures):
            rank, elapsed = future.result()
            per_rank_times.append((rank, elapsed))
    
    end_time = time.time()
    per_rank_times.sort()
    rank_times = [t for _, t in per_rank_times]
    print(f"[kvcached benchmark] map_to_kv_tensors: total={end_time - start_time:.6f}s, mean_per_rank={sum(rank_times)/tp_size:.6f}s, max={max(rank_times):.6f}s, per_rank_time: {rank_times}s. offsets: {offsets}", flush=True)

# def broadcast_map_to_kv_tensors_to_workers(tp_size: int,
#                                            offsets: list[int]) -> None:
#     start_time = time.time()
#     per_rank_times = []
#     for rank in range(tp_size):
#         t0 = time.time()
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
#         t1 = time.time()
#         per_rank_times.append(t1-t0)
    
#     end_time = time.time()
#     total_time = end_time - start_time
#     mean_time = np.mean(per_rank_times)
#     print(f"[kvcached benchmark] map_to_kv_tensors: total={total_time:.6f}s, mean_per_rank={mean_time:.6f}s, max={max(per_rank_times):.6f}s, per_rank_time: {per_rank_times}s. offsets: {offsets}", flush=True)


def broadcast_unmap_from_kv_tensors_to_workers(tp_size: int,
                                               offsets: list[int]) -> None:
    start_time = time.time()
    per_rank_times = []
    for rank in range(tp_size):
        t0 = time.time()
        socket_path = get_worker_socket_path(rank)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)
        try:
            send_msg(sock, {
                "cmd": "unmap_from_kv_tensors",
                "offsets": offsets
            })
            response = recv_msg(sock)
            if response.get("status") != "success":
                raise RuntimeError(f"Worker {rank} failed to unmap {response}")
        finally:
            sock.close()
        t1 = time.time()
        per_rank_times.append(t1-t0)
    
    end_time = time.time()
    total_time = end_time - start_time
    mean_time = np.mean(per_rank_times)
    print(f"[kvcached benchmark] unmap_from_kv_tensors: total={total_time:.6f}s, mean_per_rank={mean_time:.6f}s, max={max(per_rank_times):.6f}s, per_rank_time: {per_rank_times}s. offsets: {offsets}", flush=True)


def broadcast_kv_tensors_created_to_workers(tp_size: int) -> bool:
    created = True
    for rank in range(tp_size):
        socket_path = get_worker_socket_path(rank)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)
        try:
            send_msg(sock, {"cmd": "kv_tensors_created"})
            response = recv_msg(sock)
            if response.get("status") != "success":
                raise RuntimeError(
                    f"Worker {rank} failed to check KV tensors created: {response}"
                )
            if not response.get("created"):
                created = False
        finally:
            sock.close()

    return created
