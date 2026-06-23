import uuid
from datetime import datetime

class SessionManager:
    def __init__(self):
        self.active_session = None

    def start_session(self):
        self.active_session = {
            "id": str(uuid.uuid4()),
            "start_time": datetime.now(),
            "data": []
        }
        print(f"[SESSION STARTED] {self.active_session['id']}")
        return self.active_session["id"]

    def end_session(self):
        if not self.active_session:
            print("[SESSION END] No active session to end")
            return None

        session = self.active_session
        self.active_session = None
        print("[SESSION ENDED]")
        return session