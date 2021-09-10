from gans.core.engine.base_engine import BaseEngine
from gans.core.utils.misc import Timer,LogsTracker

class GanEngine(BaseEngine):
    def __init__(self):
        super().__init__()

    def run(self, model, dataloaders, log_every):
        self.num_epochs = model.hparams.num_epochs
        train_loader, test_loader = dataloaders.train, dataloaders.test

        epoch_timer = Timer()
        global_timer = Timer()
        train_logs_tracker = LogsTracker()

        timer_logs = {}

        steps = 0
        epoch_timer.start()
        global_timer.start()
        
        model.on_run_start()
        for epoch in range(self.num_epochs):

            model.on_train_epoch_start(epoch)
            
            for batch_idx, batch in enumerate(train_loader):
                logs = model.train_step(batch, batch_idx=batch_idx, epoch=epoch, steps=steps)
                train_logs_tracker.update(logs)
                if (steps % log_every) == 0:
                    model.log(train_logs_tracker.get_avg())
                    train_logs_tracker.reset()
            model.on_train_epoch_end(epoch)

            model.on_eval_epoch_start(epoch)
            for batch_idx, batch in enumerate(test_loader):
                model.eval_step(batch, batch_idx=batch_idx,
                                epoch=epoch, steps=steps)
            model.on_eval_epoch_end(epoch)

            epoch_timer.end()
            epoch_duration = epoch_timer.get_duration()
            timer_logs[f"Epoch {epoch}":(f"{epoch_duration:0.2f} seconds",
                                         f"{epoch_duration/60:0.2f} minutes")]
        global_timer.end()
        run_duration = global_timer.get_duration()
        timer_logs['Run time':(f"{run_duration:0.2f}",
                               f"{run_duration/60:0.2f}")]
        model.on_run_end(timer_logs)
