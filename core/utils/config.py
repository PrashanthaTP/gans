import json
import os
from datetime import datetime
import logging 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from gans.settings import BASE_DIR
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_curr_run_str():
    now = datetime.now()
    date, time = now.date(), now.time()
    day, month, year = date.day, date.month, date.year
    hour, minutes = time.hour, time.minute
    return 'Run_{}_{}_{}__{}_{}'.format(day, month, year, hour, minutes)


class Config:
    def __init__(self):
        self.vals = {'base_dir': BASE_DIR}
        self.version = get_curr_run_str()

    def __update_vals_from_dict(self, d: dict):

        for key, val in d.items():
            if isinstance(val, str):
                value = val % {'base_dir': self.vals['base_dir'],
                               'version': self.version,
                               'checkpoint_filename': 'checkpoint_'+self.version,
                               'log_filename': 'log_' + self.version
                               }
            else:
                value = val

            # setattr(self,key,val)
            self.vals[key] = value

    @classmethod
    def from_json(cls, json_file_fullpath):
        with open(json_file_fullpath, 'r') as file:
            configs = json.load(file)
            obj = cls()
            obj.__update_vals_from_dict(configs)
            return obj

    def __getitem__(self, item):
        if item not in self.vals:
            logger.error(f'Config object has no key called {item}')
            raise KeyError('Error in config key access')
        if 'fullpath' in item or 'dir' in item:
            path = os.path.dirname(self.vals[item]) if '.' in os.path.basename(
                self.vals[item]) else self.vals[item]
            os.makedirs(path, exist_ok=True)
            logger.debug(f'directory created/accessed : {path}')
        return self.vals[item]
    def __str__(self) -> str:
        vals = ""
        for key,value in self.vals.items():
            vals += f"{key} : {value}\n"
            
        return f"version : {self.version}\n {vals}"

if __name__ == '__main__':
    pass
