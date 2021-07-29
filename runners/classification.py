import torch
import torch.nn as nn

from tqdm.auto import tqdm
from .meter import AverageMeter

class Runner():
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, scheduler=None, \
                 epochs=120, eval_interval=10):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.eval_interval = eval_interval
        self.desc = lambda status, progress: f"{status}: {progress}"

        self.device = next(self.model.parameters()).device

    def train_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Train", progress))
        for batch_idx, batch in enumerate(self.train_loader):
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            # loss = nn.functional.cross_entropy(output, target)
            loss = self.criterion(output, target)
            pbar.set_postfix_str("Loss {:.4f}".format(loss.item()))
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pbar.update(1)
        pbar.close()

        return loss_meter.report()

    def evaluate(self, progress):
        self.model.eval()
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()
        with torch.no_grad():
            pbar = tqdm(total=len(self.test_loader), leave=False, desc=self.desc("Eval", progress))
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                loss_meter.update(loss.item())
                pred = output.argmax(dim=1)

                true_positive = (pred == target).sum().item()
                total = pred.shape[0]
                accuracy_meter.update(true_positive, total)
                
                pbar.update(1)
            pbar.close()
        
        return (loss_meter.report(), accuracy_meter.report())

    def train(self):
        (avg_loss, avg_acc) = self.evaluate("Init")
        tqdm.write("Evaluation init, Loss avg. {:.4f}, Acc. {:.4f}".format(avg_loss, avg_acc))

        for epoch_idx in range(self.epochs):
            avg_loss = self.train_step("{}/{}".format(epoch_idx, self.epochs))
            tqdm.write("Training procedure {} (total {}), Loss avg. {:.4f}".format(epoch_idx, self.epochs, avg_loss))
            
            if self.scheduler is not None:
                self.scheduler.step()

            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                avg_loss, avg_acc = self.evaluate("{}/{}".format(epoch_idx, self.epochs))
                tqdm.write("Evaluation {}/{}, Loss avg. {:.4f}, Acc. {:.4f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))

        tqdm.write("Finish training!")