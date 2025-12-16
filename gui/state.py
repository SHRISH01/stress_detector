from collections import deque

class SharedState:
    def __init__(self):
        self.frame = None
        self.hr = None
        self.stress = None

        self.pulse = deque(maxlen=300)
        self.hr_hist = deque(maxlen=100)
        self.stress_hist = deque(maxlen=100)
