import time
import warnings

def deprecated(msg):
    def deprecated_decorator(func):
        def deprecated_fun(*args,**kwargs):
            warnings.warn(f"{func.__name__} is a deprecated.{msg}")
            warnings.simplefilter('always',DeprecationWarning)
            return func(*args,**kwargs)
        return deprecated_fun
    return deprecated_decorator

    
class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()

    def get_duration(self):
        if not(self.start_time and self.end_time):
            error_log = "{} is not set".format(
                'start_time' if self.start_time is None else 'end_time')
            raise ValueError(error_log)
        return self.end_time-self.start_time


class AverageMeter:
    def __init__(self, name):
        self.meter = []
        self.reset()

    def reset(self):
        self.sum = 0
        self.avg = 0
        self.curr_val = 0
        self.count = 0

    def update(self, val):
        self.curr_val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

    @property
    def average(self):
        return self.avg


class LogsTracker:
    def __init__(self):
        self.logs = {}

    def update(self, d):

        for key, value in d.items():
            self.logs.setdefault(key, AverageMeter(key)).update(value)

    def get_avg(self):
        return {log_name: logs.average for log_name, logs in self.logs.items()}

    def reset(self):
        self.logs = {}
