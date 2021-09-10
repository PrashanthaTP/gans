class BaseState:
    def __init__(self):
        self.state = {}

    def update(self, param, value):
        self.state[param] = value

    def append(self, param, value):
        self.state.setdefault(param, []).append(value)

    def reset(self, param):
        if param not in self.state:
            return
        if isinstance(self.state['param'], list):
            self.state[param] = []
        elif isinstance(self.state[param], str):
            self.state[param] = ''
        else:
            self.state[param] = 0

    def get(self, param):
        if param not in self.state:
            raise KeyError(
                f"State of  {self.__class__.__name__} has no key called {param}")
        return self.state[param]

    def __getitem__(self,idx):
        if idx not in self.state:
            raise KeyError("self state has no key called ",idx)
        return self.state[idx]
    
    def __str__(self):
        s = ''
        for k,v in self.state.items():
            s += f'{k} : {v}\n'
        return s
class State(BaseState):
    pass


class AverageMeter:
    def __init__(self, name):
        self.reset()
        self.name = name

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count
    @property
    def average(self):
        return self.avg 

class Losses(BaseState):
    def __init__(self, loss_names):
        super().__init__()
        self.loss_names = loss_names
        for name in self.loss_names:
            super().update(name, AverageMeter(name))

    def update(self, train_losses: dict, count: int):
        for loss_name, value in train_losses.items():
            # super().update(loss_name,value)

            super().get(loss_name).update(train_losses[loss_name], count)

    def average_of(self, name):
        return super().get(name).avg

    def average(self):
        avgs = {}
        for name in self.loss_names:
            avgs[name] = super().get(name).average
        return avgs

    def reset(self):
        for name in self.loss_names:
            super().reset(name)


class Engine:
    def __init__(self):
        self.state = State()

    def run(self, model,train_dl,val_dl, losses_list, log_every=None):
        steps = 0
        train_losses = Losses(losses_list)
        model.on_train_start()#move models to device 
        for epoch in range(model.hparams['num_epochs']):
            self.state.update('epoch', epoch)
            model.on_epoch_start(state=self.state)#set model in train mode 
            for batch_idx, batch in enumerate(train_dl):
                self.state.update('batch_idx', batch_idx)
                logs = model.train_step(batch, state=self.state)#forward prop and backward prop 
                train_losses.update(logs['loss'], batch[0].shape[0])
                steps += 1
                self.state.update('global_step', steps)
                if log_every and  steps % log_every == 0:
                    model.log(train_losses['loss'], state=self.state)#loss log to tensorboard 
            model.on_epoch_end(train_losses.average(),state=self.state)#log epoch loss and time duration to log file 
            model.validation_step(val_dl,state=self.state) #generate sample images 
        model.on_train_end()#sum all epoch times 
        
        

def test():
    state = State()
    state.update('epoch',1)
    
    print(state['epoch'])
    print(state)

if __name__ == '__main__':
    test()