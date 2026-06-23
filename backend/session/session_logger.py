import threading

class SessionLogger:
    def __init__(self, session_manager):
        self.session_manager = session_manager
        self._lock = threading.Lock()  # ✅ thread-safe lock

    def log(self, data):
        # Run logging in separate thread (no lag)
        # Note: active_session check is done inside the lock in _log_async
        threading.Thread(
            target=self._log_async,
            args=(data,),
            daemon=True
        ).start()

    def _log_async(self, data):
        with self._lock:  # ✅ only one thread writes at a time
            if self.session_manager.active_session:  # ✅ re-check inside lock
                self.session_manager.active_session["data"].append(data)