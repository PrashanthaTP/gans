import logging
import os
# logging.basicConfig(level=logging.DEBUG)

from gans.core.utils.misc import deprecated 

class Logger:
    def __init__(self, name):
        self.log_fullpath = None
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.writers = {}

    def __add_handler(self, handler: logging.Handler):
        handler.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)

    @classmethod
    def __get_formatter(cls, fmt: str):
        return logging.Formatter(fmt)

    def add_file_handler(self, log_fullpath, fmt: str = None):
        self.log_fullpath = log_fullpath
        os.makedirs(os.path.dirname(self.log_fullpath), exist_ok=True)
        if fmt is None:
            fmt = "[%(levelname)s] : %(asctime)s : %(name)s : [%(funcName)s] : line no: %(lineno)d : %(message)s"
            
        file_handler = logging.FileHandler(log_fullpath)
        file_handler.setFormatter(self.__get_formatter(fmt))
        self.__add_handler(file_handler)

    def add_console_handler(self, fmt: str = None):
        if fmt is None:
            fmt = "[%(levelname)s] %(name)s : [%(funcName)s] : line no: %(lineno)d : %(message)s"
            
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.__get_formatter(fmt))
        self.__add_handler(console_handler)
    @property
    def debug(self):
        return self.logger.debug
    @property
    def info(self):
        return self.logger.info
    @property
    def warning(self):
        return self.logger.warning
    @property
    def error(self):
        return self.logger.error
    @property
    def exception(self):
        return self.logger.exception
    
    @deprecated("Use `add_writer_real` and `add_writer_fake` instead.")
    def add_writer(self,name,writer):
       self.writers[name] = writer 
    
    def add_writer_real(self,writer):
        self.writers['writer_real'] = writer
    
    def add_writer_fake(self,writer):
        self.writers['writer_fake'] = writer
        
    def add_scalars_to_board(self, loss_gen, loss_disc, step):
        writer_fake, writer_real = self.writers['writer_fake'], self.writers['writer_real']
        writer_real.add_scalar('loss/disc', loss_disc, global_step=step)
        writer_fake.add_scalar('loss/gen', loss_gen, global_step=step)

    def add_imgs_to_board(self, img_grid_real,img_grid_fake, step):
 
        writer_fake, writer_real = self.writers['writer_fake'], self.writers['writer_real']
        writer_fake.add_image("MNIST Images/fake",
                              img_grid_fake, global_step=step)
        writer_real.add_image("MNIST Images/real",
                              img_grid_real, global_step=step)
