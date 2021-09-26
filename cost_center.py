import time


class TimeCostCenter:
    def __init__(self, name):
        self.name = name
        self.total_time = 0

    def add_time(self, t):
        self.total_time += t

    def __sub__(self, other):
        cc = TimeCostCenter(f"{self.name} - {other.name}")
        cc.total_time = self.total_time - other.total_time
        return cc

    def __add__(self, other):
        cc = TimeCostCenter(f"{self.name} + {other.name}")
        cc.total_time = self.total_time + other.total_time
        return cc

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.name}]<{self.total_time:.3f}s>"


class TimedContext:
    def __init__(self, cc):
        self.cc = cc

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, _exc_type, _exc_val, _traceback):
        self.cc.add_time(time.time() - self.t0)
