from __future__ import annotations

import threading

__all__ = ["BoardSampler", "DEFAULT_PHASE", "InvalidSampleError"]

DEFAULT_PHASE = "default"

_NVML = None


class InvalidSampleError(ValueError):
    pass


def _coerce_int(value, field):
    if isinstance(value, bool):
        raise InvalidSampleError("boolean field %r" % (field,))
    if isinstance(value, int):
        result = value
    else:
        try:
            result = int(value)
        except (TypeError, ValueError):
            try:
                result = int(float(value))
            except (TypeError, ValueError):
                raise InvalidSampleError("non-integer field %r" % (field,))
    if result < 0:
        raise InvalidSampleError("negative field %r" % (field,))
    return result


def _normalise_sample(raw):
    if raw is None:
        raise InvalidSampleError("sample_fn returned None")
    try:
        total = raw["total_bytes"]
        used = raw["used_bytes"]
        free = raw["free_bytes"]
    except (TypeError, KeyError, IndexError):
        raise InvalidSampleError("sample_fn payload missing required keys")
    return {
        "total_bytes": _coerce_int(total, "total_bytes"),
        "used_bytes": _coerce_int(used, "used_bytes"),
        "free_bytes": _coerce_int(free, "free_bytes"),
    }


class _NvmlBridge:
    _LIBRARY_NAMES = ("libnvidia-ml.so.1", "libnvidia-ml.so")

    def __init__(self):
        import ctypes

        self._ctypes = ctypes
        self._initialised = False
        self._lib = self._load_library(ctypes)
        # One shared struct type: ctypes compares argtypes by identity.
        self._memory_info_type = self._make_memory_info_type()

        self._lib.nvmlInit_v2.restype = ctypes.c_int
        self._lib.nvmlInit_v2.argtypes = []

        self._lib.nvmlShutdown.restype = ctypes.c_int
        self._lib.nvmlShutdown.argtypes = []

        self._lib.nvmlDeviceGetHandleByUUID.restype = ctypes.c_int
        self._lib.nvmlDeviceGetHandleByUUID.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]

        self._lib.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
        self._lib.nvmlDeviceGetMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._memory_info_type),
        ]

        status = self._lib.nvmlInit_v2()
        if status != 0:
            raise RuntimeError("nvmlInit_v2 failed with status %d" % (status,))
        self._initialised = True

    @staticmethod
    def _make_memory_info_type():
        import ctypes

        class _MemoryInfo(ctypes.Structure):
            _fields_ = [
                ("total", ctypes.c_uint64),
                ("free", ctypes.c_uint64),
                ("used", ctypes.c_uint64),
            ]

        return _MemoryInfo

    def _load_library(self, ctypes):
        last_error = None
        for name in self._LIBRARY_NAMES:
            try:
                return ctypes.CDLL(name)
            except OSError as exc:  # pragma: no cover
                last_error = exc
        raise RuntimeError(
            "unable to load libnvidia-ml: %s" % (type(last_error).__name__,)
        )

    def handle_for_uuid(self, device_uuid):
        ctypes = self._ctypes
        handle = ctypes.c_void_p()
        status = self._lib.nvmlDeviceGetHandleByUUID(
            str(device_uuid).encode("utf-8"), ctypes.byref(handle)
        )
        if status != 0:
            raise RuntimeError(
                "nvmlDeviceGetHandleByUUID failed with status %d" % (status,)
            )
        return handle

    def memory_info(self, handle):
        info = self._memory_info_type()
        status = self._lib.nvmlDeviceGetMemoryInfo(handle, self._ctypes.byref(info))
        if status != 0:
            raise RuntimeError(
                "nvmlDeviceGetMemoryInfo failed with status %d" % (status,)
            )
        return {
            "total_bytes": int(info.total),
            "used_bytes": int(info.used),
            "free_bytes": int(info.free),
        }

    def shutdown(self):
        global _NVML
        if not self._initialised:
            return
        self._initialised = False
        if _NVML is self:
            _NVML = None
        status = self._lib.nvmlShutdown()
        if status != 0:
            raise RuntimeError("nvmlShutdown failed with status %d" % (status,))


def _load_nvml():
    global _NVML
    if _NVML is None:
        _NVML = _NvmlBridge()
    return _NVML


class BoardSampler:
    def __init__(self, sample_fn, interval=0.05):
        if not callable(sample_fn):
            raise TypeError("sample_fn must be callable")
        interval = float(interval)
        if interval <= 0:
            raise ValueError("interval must be positive")

        self._sample_fn = sample_fn
        self._interval = interval
        self._shutdown_fn = None

        self._lock = threading.RLock()
        self._transition_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread = None
        self._started = False
        self._stopped = False

        self._phase = DEFAULT_PHASE
        self._baseline = None
        self._by_phase = {}
        self._errors = []
        self._max_used_bytes = None
        self._min_free_bytes = None
        self._samples = 0

    def _is_active(self):
        with self._lock:
            return self._started and not self._stopped

    @property
    def interval(self):
        return self._interval

    @property
    def phase(self):
        with self._lock:
            return self._phase

    @property
    def thread(self):
        with self._lock:
            return self._thread

    @property
    def is_running(self):
        with self._lock:
            return self._started and not self._stopped

    def _record_error(self, exc):
        error_type = type(exc).__name__
        with self._lock:
            self._errors.append(error_type)

    def _sample_once(self):
        try:
            raw = self._sample_fn()
        except Exception as exc:  # noqa: BLE001
            self._record_error(exc)
            return None
        try:
            sample = _normalise_sample(raw)
        except Exception as exc:  # noqa: BLE001
            self._record_error(exc)
            return None

        with self._lock:
            phase = self._phase
            self._samples += 1
            used = sample["used_bytes"]
            free = sample["free_bytes"]
            total = sample["total_bytes"]

            if self._max_used_bytes is None or used > self._max_used_bytes:
                self._max_used_bytes = used
            if self._min_free_bytes is None or free < self._min_free_bytes:
                self._min_free_bytes = free

            bucket = self._by_phase.get(phase)
            if bucket is None:
                self._by_phase[phase] = {
                    "count": 1,
                    "max_used_bytes": used,
                    "min_free_bytes": free,
                    "total_bytes": total,
                }
            else:
                bucket["count"] += 1
                if used > bucket["max_used_bytes"]:
                    bucket["max_used_bytes"] = used
                if free < bucket["min_free_bytes"]:
                    bucket["min_free_bytes"] = free
                if total > bucket["total_bytes"]:
                    bucket["total_bytes"] = total

        return dict(sample)

    def sample_now(self):
        if not self._is_active():
            return None
        return self._sample_once()

    def start(self):
        with self._lock:
            if self._started and not self._stopped:
                return self
            self._started = True
            self._stopped = False
            self._stop_event = threading.Event()

        baseline = self._sample_once()
        with self._lock:
            if baseline is not None:
                self._baseline = dict(baseline)
            stop_event = self._stop_event

        thread = threading.Thread(
            target=self._run,
            args=(stop_event,),
            name="board-sampler",
            daemon=True,
        )
        with self._lock:
            self._thread = thread
        thread.start()
        return self

    def _run(self, stop_event):
        while True:
            try:
                if stop_event.wait(self._interval):
                    return
                with self._transition_lock:
                    if stop_event.is_set():
                        return
                    self._sample_once()
            except Exception as exc:  # noqa: BLE001
                try:
                    self._record_error(exc)
                except Exception:  # noqa: BLE001
                    return

    def set_phase(self, name):
        if name is None:
            name = DEFAULT_PHASE
        name = str(name)

        with self._transition_lock:
            active = self._is_active()
            if active:
                self._sample_once()
            with self._lock:
                self._phase = name
            if active:
                self._sample_once()
        return name

    def stop(self):
        with self._lock:
            already_stopped = self._stopped
            started = self._started
            self._stopped = True
            self._stop_event.set()
            thread = self._thread
            self._thread = None

        if not already_stopped and thread is not None:
            if thread.is_alive() and thread is not threading.current_thread():
                try:
                    thread.join(timeout=max(1.0, self._interval * 4.0))
                except Exception as exc:  # noqa: BLE001
                    self._record_error(exc)

        if not already_stopped and started:
            acquired = False
            try:
                acquired = self._transition_lock.acquire(
                    timeout=max(0.5, self._interval * 2.0)
                )
                if acquired:
                    self._sample_once()
            except Exception as exc:  # noqa: BLE001
                self._record_error(exc)
            finally:
                if acquired:
                    try:
                        self._transition_lock.release()
                    except Exception as exc:  # noqa: BLE001
                        self._record_error(exc)

        shutdown_fn = self._shutdown_fn
        if not already_stopped and shutdown_fn is not None:
            try:
                shutdown_fn()
            except Exception as exc:  # noqa: BLE001
                self._record_error(exc)
            finally:
                self._shutdown_fn = None

        return self.snapshot()

    def snapshot(self):
        with self._lock:
            baseline = dict(self._baseline) if self._baseline is not None else None
            by_phase = {}
            for name, bucket in self._by_phase.items():
                by_phase[str(name)] = {
                    "count": int(bucket["count"]),
                    "max_used_bytes": int(bucket["max_used_bytes"]),
                    "min_free_bytes": int(bucket["min_free_bytes"]),
                    "total_bytes": int(bucket["total_bytes"]),
                }
            return {
                "baseline": baseline,
                "sample_interval_seconds": float(self._interval),
                "by_phase": by_phase,
                "errors": [str(err) for err in self._errors],
                "max_used_bytes": (
                    None if self._max_used_bytes is None else int(self._max_used_bytes)
                ),
                "min_free_bytes": (
                    None if self._min_free_bytes is None else int(self._min_free_bytes)
                ),
                "samples": int(self._samples),
            }

    @classmethod
    def for_uuid(cls, uuid, interval=0.05):
        nvml = _load_nvml()
        handle = nvml.handle_for_uuid(uuid)

        def sample_fn():
            return nvml.memory_info(handle)

        sampler = cls(sample_fn, interval=interval)
        sampler._shutdown_fn = nvml.shutdown
        return sampler
