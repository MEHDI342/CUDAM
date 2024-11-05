from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import threading
import queue
import time
import uuid
from multiprocessing import cpu_count
from contextlib import contextmanager

from ..utils.error_handler import CudaTranslationError
from ..utils.logger import get_logger
from ..metrics import MetricsCollector

logger = get_logger(__name__)

class TaskPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

class TaskStatus(Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    ACQUIRING = "acquiring"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TranslationTask:
    id: str
    priority: TaskPriority
    payload: Any
    dependencies: Set[str]
    resources: Set[str]
    status: TaskStatus
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    error: Optional[Exception]
    result: Optional[Any]
    retries: int = 0

class ResourceGuard:
    def __init__(self):
        self._locks: Dict[str, threading.Lock] = {}
        self._resource_usage: Dict[str, Set[str]] = {}
        self._global_lock = threading.Lock()

    @contextmanager
    def acquire(self, resources: Set[str], task_id: str):
        acquired = []
        try:
            with self._global_lock:
                for resource in resources:
                    if resource not in self._locks:
                        self._locks[resource] = threading.Lock()
                        self._resource_usage[resource] = set()

            for resource in sorted(resources):
                self._locks[resource].acquire()
                acquired.append(resource)
                with self._global_lock:
                    self._resource_usage[resource].add(task_id)
            yield

        finally:
            for resource in reversed(acquired):
                with self._global_lock:
                    self._resource_usage[resource].remove(task_id)
                self._locks[resource].release()

class DependencyGraph:
    def __init__(self):
        self._forward: Dict[str, Set[str]] = {}
        self._backward: Dict[str, Set[str]] = {}
        self._lock = threading.Lock()

    def add_task(self, task_id: str, dependencies: Set[str]):
        with self._lock:
            self._forward[task_id] = dependencies
            for dep in dependencies:
                if dep not in self._backward:
                    self._backward[dep] = set()
                self._backward[dep].add(task_id)

    def remove_task(self, task_id: str):
        with self._lock:
            deps = self._forward.pop(task_id, set())
            for dep in deps:
                self._backward[dep].discard(task_id)
            dependents = self._backward.pop(task_id, set())
            for dependent in dependents:
                self._forward[dependent].discard(task_id)

    def get_ready_tasks(self) -> Set[str]:
        with self._lock:
            return {
                task_id for task_id, deps in self._forward.items()
                if not deps
            }

    def task_completed(self, task_id: str) -> Set[str]:
        with self._lock:
            dependents = self._backward.get(task_id, set()).copy()
            self.remove_task(task_id)
            return {
                dep for dep in dependents
                if not self._forward[dep]
            }

class TaskExecutor:
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._processors: Dict[str, Any] = {}

    def register_processor(self, task_type: str, processor: Any):
        self._processors[task_type] = processor

    async def execute(self, task: TranslationTask) -> Any:
        start_time = time.time()
        task.status = TaskStatus.RUNNING
        task.started_at = start_time

        try:
            processor = self._processors[task.payload.task_type]
            result = await processor.process(task.payload)
            task.result = result
            task.status = TaskStatus.COMPLETED
            return result
        except Exception as e:
            task.error = e
            task.status = TaskStatus.FAILED
            raise
        finally:
            task.completed_at = time.time()
            self.metrics.record_task_execution(
                task_id=task.id,
                duration=task.completed_at - start_time,
                status=task.status
            )

class TranslationThreadPool:
    def __init__(self,
                 max_workers: Optional[int] = None,
                 max_retries: int = 3):
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers or (cpu_count() * 2)
        )
        self.metrics = MetricsCollector()
        self.task_executor = TaskExecutor(self.metrics)
        self.resource_guard = ResourceGuard()
        self.dependency_graph = DependencyGraph()

        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._active_tasks: Dict[str, TranslationTask] = {}
        self._futures: Dict[str, Future] = {}
        self._max_retries = max_retries
        self._lock = threading.Lock()
        self._shutdown = False

        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def submit(self,
                     task_type: str,
                     payload: Any,
                     priority: TaskPriority = TaskPriority.NORMAL,
                     dependencies: Optional[Set[str]] = None,
                     resources: Optional[Set[str]] = None) -> str:
        if self._shutdown:
            raise RuntimeError("ThreadPool is shutting down")

        task_id = str(uuid.uuid4())
        task = TranslationTask(
            id=task_id,
            priority=priority,
            payload=payload,
            dependencies=dependencies or set(),
            resources=resources or set(),
            status=TaskStatus.PENDING,
            created_at=time.time(),
            started_at=None,
            completed_at=None,
            error=None,
            result=None
        )

        with self._lock:
            self._active_tasks[task_id] = task
            self.dependency_graph.add_task(task_id, task.dependencies)
            if not task.dependencies:
                self._queue.put((-priority.value, task_id))

        return task_id

    async def _scheduler_loop(self):
        while not self._shutdown:
            try:
                if not self._queue.empty():
                    _, task_id = self._queue.get_nowait()
                    await self._schedule_task(task_id)
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Scheduler error: {str(e)}")

    async def _schedule_task(self, task_id: str):
        with self._lock:
            task = self._active_tasks[task_id]
            task.status = TaskStatus.SCHEDULED

        try:
            with self.resource_guard.acquire(task.resources, task_id):
                future = self.executor.submit(
                    self._execute_task,
                    task_id
                )
                self._futures[task_id] = future
                await self._monitor_task(task_id, future)
        except Exception as e:
            logger.error(f"Task {task_id} scheduling failed: {str(e)}")
            await self._handle_task_failure(task_id, e)

    def _execute_task(self, task_id: str) -> Any:
        task = self._active_tasks[task_id]
        return asyncio.run(self.task_executor.execute(task))

    async def _monitor_task(self, task_id: str, future: Future):
        try:
            result = await asyncio.wrap_future(future)
            await self._handle_task_completion(task_id, result)
        except Exception as e:
            await self._handle_task_failure(task_id, e)

    async def _handle_task_completion(self, task_id: str, result: Any):
        with self._lock:
            task = self._active_tasks[task_id]
            task.result = result
            task.status = TaskStatus.COMPLETED
            ready_tasks = self.dependency_graph.task_completed(task_id)
            for ready_task in ready_tasks:
                self._queue.put((-self._active_tasks[ready_task].priority.value,
                                 ready_task))

    async def _handle_task_failure(self, task_id: str, error: Exception):
        with self._lock:
            task = self._active_tasks[task_id]
            if task.retries < self._max_retries:
                task.retries += 1
                task.status = TaskStatus.PENDING
                self._queue.put((-task.priority.value, task_id))
            else:
                task.error = error
                task.status = TaskStatus.FAILED
                self.dependency_graph.remove_task(task_id)

    async def wait_for_task(self, task_id: str) -> Any:
        while True:
            with self._lock:
                task = self._active_tasks.get(task_id)
                if task and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    if task.status == TaskStatus.FAILED:
                        raise task.error
                    return task.result
            await asyncio.sleep(0.01)

    async def shutdown(self):
        self._shutdown = True
        self._scheduler_task.cancel()
        self.executor.shutdown(wait=True)