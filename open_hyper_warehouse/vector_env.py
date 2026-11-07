"""Serial and Windows-safe process-parallel warehouse environment backends."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection
import traceback
from typing import Any

import numpy as np

from .config import EnvConfig
from .env import OpenWarehouseEnv


def _environment_worker(connection: Connection, env_values: dict[str, Any], seed: int) -> None:
    """Own exactly one simulator so no mutable environment state is shared."""

    try:
        env = OpenWarehouseEnv(EnvConfig(**env_values), seed=seed)
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            if command == "reset":
                connection.send(("ok", env.reset(seed=int(payload))))
            elif command == "step":
                connection.send(("ok", env.step(np.asarray(payload, dtype=np.int64))))
            elif command == "close":
                connection.send(("closed", None))
                break
            else:
                raise KeyError(f"unknown vector-environment command {command!r}")
    except BaseException:  # propagate the worker traceback instead of hanging the parent
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


class SerialVectorEnv:
    backend = "serial"

    def __init__(self, env_cfg: EnvConfig, count: int, seeds: list[int]):
        self.envs = [OpenWarehouseEnv(env_cfg, seed=seeds[i]) for i in range(count)]

    def reset_all(self, seeds: list[int]) -> list[dict[str, np.ndarray]]:
        return [env.reset(seed=seed) for env, seed in zip(self.envs, seeds)]

    def reset_at(self, indexes: list[int], seeds: list[int]) -> dict[int, dict[str, np.ndarray]]:
        return {index: self.envs[index].reset(seed=seed) for index, seed in zip(indexes, seeds)}

    def step(self, actions: np.ndarray):
        return [env.step(actions[index]) for index, env in enumerate(self.envs)]

    def close(self) -> None:
        return None


class ProcessVectorEnv:
    backend = "process"

    def __init__(self, env_cfg: EnvConfig, count: int, seeds: list[int]):
        if count < 1:
            raise ValueError("process vector environment requires at least one worker")
        context = mp.get_context("spawn")
        self.connections: list[Connection] = []
        self.processes: list[mp.Process] = []
        env_values = {name: getattr(env_cfg, name) for name in env_cfg.__dataclass_fields__}
        for index in range(count):
            parent, child = context.Pipe()
            process = context.Process(
                target=_environment_worker,
                args=(child, env_values, seeds[index]),
                name=f"warehouse-env-{index}",
                daemon=True,
            )
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
        for connection in self.connections:
            status, payload = connection.recv()
            if status != "ready":
                self.close()
                raise RuntimeError(f"environment worker failed at startup: {payload}")

    @staticmethod
    def _receive(connection: Connection):
        status, payload = connection.recv()
        if status == "error":
            raise RuntimeError(f"environment worker failed:\n{payload}")
        if status not in {"ok", "closed"}:
            raise RuntimeError(f"unexpected environment worker status {status!r}")
        return payload

    def reset_all(self, seeds: list[int]) -> list[dict[str, np.ndarray]]:
        for connection, seed in zip(self.connections, seeds):
            connection.send(("reset", int(seed)))
        return [self._receive(connection) for connection in self.connections]

    def reset_at(self, indexes: list[int], seeds: list[int]) -> dict[int, dict[str, np.ndarray]]:
        for index, seed in zip(indexes, seeds):
            self.connections[index].send(("reset", int(seed)))
        return {
            index: self._receive(self.connections[index])
            for index in indexes
        }

    def step(self, actions: np.ndarray):
        for index, connection in enumerate(self.connections):
            connection.send(("step", actions[index]))
        return [self._receive(connection) for connection in self.connections]

    def close(self) -> None:
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, EOFError):
                    pass
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                try:
                    self._receive(connection)
                except (BrokenPipeError, EOFError, RuntimeError):
                    pass
            connection.close()
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)


def make_vector_env(env_cfg: EnvConfig, count: int, process_workers: int, seeds: list[int]):
    if process_workers == 0:
        return SerialVectorEnv(env_cfg, count, seeds)
    if process_workers != count:
        raise ValueError(
            "the current process backend requires parallel_env_workers == num_envs"
        )
    return ProcessVectorEnv(env_cfg, count, seeds)
