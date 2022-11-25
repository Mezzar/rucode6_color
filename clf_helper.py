import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm
import time
import os
from IPython.display import clear_output

import torch
from torch.utils.data import Dataset, DataLoader

class CLF:
    def __init__(self,
                 model:torch.nn.Module, optimizer:torch.optim.Optimizer,
                 train_loader:DataLoader, val_loader:DataLoader,
                 lr_scheduler=None,
                 name:str=None,
                 device='cuda',
                 clip:float=None):
        '''
            TODO: Document params and a usecase
        '''
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader   
        self.lr_scheduler = lr_scheduler
        self.epoch = 0
        self.history = []
        self.device = device
        self.grad_clip = clip   #maximum norm of gradient. Set for gradient clipping (exeample: 2)
        self.name = name

        self.criterion = torch.nn.CrossEntropyLoss()
        self.model.to(device)
    
    def train(self, n_epoch:int, patience:int=None, silent=False, plot=False,
              save_best_accuracy=False, save_best_loss=False):
        patience_loss = float('inf')
        final_epoch = self.epoch + n_epoch
        for self.epoch in trange(self.epoch + 1, final_epoch + 1):
            time_started = time.time()
            
            train_loss, train_accuracy = self._dataloop(is_training=True)
            val_loss, val_accuracy = self._dataloop(is_training=False)

            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            epoch_seconds = time.time() - time_started
            total_seconds = epoch_seconds + sum(i['epoch_seconds'] for i in self.history)
            if plot: 
                clear_output(wait=True)
            if not silent:
                tqdm.write(f"Total epoch {self.epoch}/{final_epoch} ({total_seconds/60:.1f} min)" + 
                           f"| Train: loss {train_loss:.4f}" +
                           f"| Val: loss {val_loss:.4f} accuracy {val_accuracy*100:.1f}%")
            self.history.append({
                'epoch': self.epoch,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'epoch_seconds': round(epoch_seconds, 1),
                'total_seconds': round(total_seconds),
            })
            
            # Save checkpoint if necessary (needs to be after updating history)
            self._save_checkpoint(save_best_accuracy, save_best_loss)
            
            if plot:
                print()
                self.stats()
                self.plot()
            
            # Stop training if out of patience
            if patience is not None:
                if val_loss < patience_loss:
                    patience_loss = val_loss
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count == patience + 1:
                        print('Patience exceeded. Stopped training')
                        break

    def _dataloop(self, is_training:bool):
        if is_training:
            self.model.train()
            data_loader = self.train_loader
        else:
            self.model.eval()
            data_loader = self.val_loader

        losses = []
        correct = total = 0
        for X_batch, y_batch in tqdm(data_loader, leave=False):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).flatten()
            #need flatten if y is shape (batch,1) instead of (batch,) othersize CrossEntropyLoss returns error

            if is_training:
                pred = self.model(X_batch)
            else:
                with torch.no_grad():
                    pred = self.model(X_batch)
            
            loss = self.criterion(pred, y_batch)
            losses.append(loss.item())

            if is_training:
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()    
            
            correct += (pred.argmax(dim=1) == y_batch).sum().item()
            total += len(y_batch)
        avg_loss = np.mean(losses)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def predict(self, loader:['train','val',DataLoader], return_ground=False):
        logits, labels = self._get_logits_and_labels(loader)
        out = logits.argmax(dim=1)
        if return_ground:
            out = out, labels
        return out

    def predict_proba(self, loader:['train','val',DataLoader], return_ground=False):
        logits, labels = self._get_logits_and_labels(loader)
        out = torch.nn.functional.softmax(logits, dim=1)
        if return_ground:
            out = out, labels
        return out

    def _get_logits_and_labels(self, loader:['train','val',DataLoader]):
        if loader == 'train':
            loader = self.train_loader
        elif loader == 'val':
            loader = self.val_loader
        
        b_logits = []
        b_labels = []
        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                pred = self.model(X_batch)
                b_logits.append(pred)
                b_labels.append(y_batch)
        logits = torch.cat(b_logits, dim=0).cpu()
        labels = torch.cat(b_labels, dim=0).cpu()
        return logits, labels        
    
    def plot(self, start=None, end=None, x_time=False):
        if not self.epoch:
            print('Model not trained (epoch 0)')
            return            
        if end is None:
            history = self.history[: start]
        else:
            history = self.history[start : end]
        x_attr = 'epoch' if not x_time else 'total_seconds'
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,4))
        if self.name:
            fig.suptitle(self.name)

        #accuracy graph
        x = [i[x_attr] for i in history]
        for y_attr in ['train_accuracy', 'val_accuracy']:
            y = [hitem[y_attr] for hitem in history]
            p = ax1.plot(x, y, label=y_attr)
            #max point
            idx = np.argmax(np.array([hitem[y_attr] for hitem in history]))
            ax1.scatter(x=history[idx][x_attr], y=history[idx][y_attr], c=p[0].get_color())

        ax1.set_xlabel(x_attr)
        ax1.set_title('Accuracy')
        ax1.legend()

        #loss graph
        x = [record[x_attr] for record in history]
        for y_attr in ['train_loss', 'val_loss']:
            y = [hitem[y_attr] for hitem in history]
            p = ax2.plot(x, y, label=y_attr)
            #min point
            idx = np.argmin(np.array([hitem[y_attr] for hitem in history]))
            ax2.scatter(x=history[idx][x_attr], y=history[idx][y_attr], c=p[0].get_color())
        
        ax2.set_xlabel(x_attr)
        ax2.set_title('Loss')
        ax2.legend()
        plt.show()
        
    def stats(self):
        if not self.epoch:
            print('Model not trained (epoch 0)')
            return    
        idx_acc = np.argmax([i['val_accuracy'] for i in self.history])
        idx_lss = np.argmin([i['val_loss'] for i in self.history])
        
        print('Best val accuracy {:.2f}% on epoch {} ({:.1f} min) | best val loss {:.4f} on epoch {} ({:.1f} min)'.format(
            self.history[idx_acc]['val_accuracy'] * 100, self.history[idx_acc]['epoch'],
            self.history[idx_acc]['total_seconds'] / 60, 
            self.history[idx_lss]['val_loss'], self.history[idx_lss]['epoch'],
            self.history[idx_lss]['total_seconds'] / 60, 
        ))
        
    def save(self, filepath='checkpoint.pt'):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'history': self.history,
        }
        torch.save(checkpoint, filepath)
        print(f'Saved checkpoint: epoch {self.epoch}')
    
    def load(self, filepath='checkpoint.pt', ignore_error=False):
        if not os.path.isfile(filepath):
            if ignore_error:
                print('Skipping checkpoint load. File not found:', filepath)
                return
            else:
                raise FileNotFoundError(f'Checkpoint file not found: {filepath}')
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.device:
            self.model.to(self.device) #must do this before loading optimizer
        self.model.eval()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch   = checkpoint['epoch']
        self.history = checkpoint['history']
        print(f'Loaded checkpoint: {filepath} (epoch {self.epoch})')

    def _save_checkpoint(self, save_best_accuracy, save_best_loss):
        if save_best_accuracy:
            if self.epoch == 1 or self.history[-1]['val_accuracy'] > max(i['val_accuracy'] for i in self.history[:-1]):
                filename = save_best_accuracy if (type(save_best_accuracy) is str) else 'checkpoint_best_accuracy.pt'
                self.save(filename)
        if save_best_loss:
            if self.epoch == 1 or self.history[-1]['val_loss'] < min(i['val_loss'] for i in self.history[:-1]):
                filename = save_best_loss if (type(save_best_loss) is str) else 'checkpoint_best_loss.pt'
                self.save(filename)


# Дальше идет код от старой версии, он сейчас не используется
# вытащить оттуда сравнение clf - график и таблицу.
# Придумать, как лучше это реализовать, чтобы не добавлять еще кучу функций к CLF

# при определении имени модели в визуализаторе, можно делать так: `name = model.name or str(model.__class__.__name__)`


class Checkpoint():
    def __init__(self, clf, params=None):
        self.save_every_n_epoch = None
        self.save_best_acc = False
        self.save_best_loss = False
        self.silent = True
        self.path_template = "./weights/{0}.pth"

        self.clf = clf
        self.attrs = ['save_every_n_epoch', 'save_best_acc', 'save_best_loss', 'path_template', 'silent']
        self.set_params(params)

    def params(self):
        return {a: getattr(self, a) for a in self.attrs}
        
    def set_params(self, params:dict):
        if not params: 
            return
        for attr, value in params.items():
            if attr in self.attrs:
                setattr(self, attr, value)
            else:
                raise AttributeError()
                
    def save(self, suffix=''):
        if not self.silent:
            print(f'Saving checkpoint: epoch {self.clf.epoch}')
        filename = self.clf.name + suffix
        filepath = self.path_template.format(filename)

        checkpoint = {
            'model_state_dict': self.clf.model.state_dict(),
            'optimizer_state_dict': self.clf.optimizer.state_dict(),
            'epoch': self.clf.epoch,
            'history': self.clf.history,
        }
        torch.save(checkpoint, filepath)
    
    def load(self, suffix=''):
        clf = self.clf

        filename = clf.name + suffix
        filepath = self.path_template.format(filename)
        if not os.path.isfile(filepath):
            if not self.silent:
                print('Skipping checkpoint load. File not found ', filepath)
            return
        
        if not self.silent:
            print('Loading weights: ' + filepath)
        checkpoint = torch.load(filepath, map_location=self.clf.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.device:
            self.model.to(self.device) #делать до загрузки параметров оптимизатора
        self.model.eval()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch   = checkpoint['epoch']
        self.history = checkpoint['history']
        
    def step(self):
        clf = self.clf
        if self.save_every_n_epoch:
            if clf.epoch % self.save_every_n_epoch == 0:
                self.save()
        
        if self.save_best_acc and clf.epoch > 1:
            if clf.history[-1]['val_accuracy'] > max(i['val_accuracy'] for i in clf.history[:-1]):
                self.save(suffix='_best_acc')
        
        if self.save_best_loss and clf.epoch > 1:
            if clf.history[-1]['val_loss'] < min(i['val_loss'] for i in clf.history[:-1]):
                self.save(suffix='_best_loss')
                
#TODO: временами происходит дублирование процедуры сохранения после последней эпохи и в step и finish
       #мб передавать в step количество эпох до конца и делать проверку, нужно ли сохранять в step, если шаг последний и стоит save_on_finish=True
       

class CLFS():
    def __init__(self):
        self.clfs = dict()
    
    def add(self, clf:CLF, label:str=None):
        label = label or clf.name
        self.clfs[label] = clf    
    def get(self, label)->CLF:
        return self.clfs[label]
        
    def train_all(self, max_epoch, auto_checkpoint=False, checkpoint_params=None, silent=False):
        for label, c in self.clfs.items():
            self.train_clf(label, max_epoch, auto_checkpoint=auto_checkpoint, checkpoint_params=checkpoint_params, silent=silent)
                    
    def train_clf(self, label, max_epoch, auto_checkpoint=False, checkpoint_params=None, silent=False):
        clf = self.get(label)
        prev_params = clf.checkpoint.params()
        clf.checkpoint.set_params(checkpoint_params)
        if auto_checkpoint:
            clf.checkpoint.load()        
        
        epoch_left = max_epoch - clf.epoch
        print(f'{label} (epoch {clf.epoch} -> {max_epoch})')
        if epoch_left > 0:
            clf.train(n_epoch=epoch_left, silent=silent)

            if auto_checkpoint:
                clf.checkpoint.save()
        clf.checkpoint.set_params(prev_params)

    def plot(self, *pargs, **kargs):
        return visualizer(self).plot(*pargs, **kargs)

    def stats(self, *pargs, **kargs):
        return visualizer(self).stats(*pargs, **kargs)


class CLF_Visualizer():
    def __init__(self, clf):
        self.clf = clf
    
    def plot(self, slice1=None, slice2=None, x_time=False):
        if not self.clf.epoch:
            print('Model not trained (epoch 0)')
            return            
        if slice2 is None:
            history = self.clf.history[: slice1]
        else:
            history = self.clf.history[slice1 : slice2]
        x_attr = 'epoch' if not x_time else 'total_seconds'
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,4))
        
        #график accuracy
        x = [i[x_attr] for i in history]
        for y_attr in ['train_accuracy', 'val_accuracy']:
            y = [hitem[y_attr] for hitem in history]
            p = ax1.plot(x, y, label=y_attr)
            #точки максимума
            idx = np.argmax(np.array([hitem[y_attr] for hitem in history]))
            ax1.scatter(x=history[idx][x_attr], y=history[idx][y_attr], c=p[0].get_color())

        ax1.set_xlabel(x_attr)
        ax1.set_title('Accuracy')
        ax1.legend()

        #график loss
        x = [record[x_attr] for record in history]
        for y_attr in ['train_loss', 'val_loss']:
            y = [hitem[y_attr] for hitem in history]
            p = ax2.plot(x, y, label=y_attr)
            #точка минимума
            idx = np.argmin(np.array([hitem[y_attr] for hitem in history]))
            ax2.scatter(x=history[idx][x_attr], y=history[idx][y_attr], c=p[0].get_color())

        ax2.set_xlabel(x_attr)
        ax2.set_title('Loss')
        ax2.legend()
        plt.show()

    def stats(self):
        raise NotImplementedError()
                
        
class CLFS_Visualizer():
    def __init__(self, clfs):
        self.clfs = clfs
    
    def plot(self, slice1=None, slice2=None, x_time=False):
        if slice2 is None:
            hist_slice = slice(slice1)
        else:
            hist_slice = slice(slice1, slice2)
        x_attr = 'epoch' if not x_time else 'total_seconds'
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,4))
        
        #график accuracy
        for name, c in self.clfs.clfs.items():
            history = c.history[hist_slice]
            if not history:
                continue
            x = [i[x_attr] for i in history]
            y = [i['val_accuracy'] for i in history]
            p = ax1.plot(x, y, label=name)
            
            #точка максимума
            idx = np.argmax(np.array([i['val_accuracy'] for i in history]))
            ax1.scatter(x=history[idx][x_attr], y=history[idx]['val_accuracy'], c=p[0].get_color())

            #график loss
            x = [i[x_attr] for i in history]
            y = [i['val_loss'] for i in history]
            p = ax2.plot(x, y, label=name)
            
            #точка минимума
            idx = np.argmin(np.array([i['val_loss'] for i in history]))
            ax2.scatter(x=history[idx][x_attr], y=history[idx]['val_loss'], c=p[0].get_color())

        ax1.set_xlabel(x_attr)
        ax1.set_title('Val Accuracy')
        ax1.legend()
        ax2.set_xlabel(x_attr)
        ax2.set_title('Val Loss')
        ax2.legend()
        plt.show()

    def stats(self):
        #таблицы лучих значений
        for metric, compare_fn in [('val_accuracy', np.argmax), ('val_loss', np.argmin)]:
            print(f'\nЛучшая {metric}\n{"": <30}  accuracy   epoch  минут')    
            for name, c in self.clfs.clfs.items():
                idx_best = compare_fn(np.array([i[metric] for i in c.history]))
                best = c.history[idx_best]
                print('   {0: <30} {1:.1f}%    {2:< 3}    {3:.1f}'.format(name, best['val_accuracy'] * 100, best['epoch'], best['total_seconds'] / 60))

def visualizer(obj):
    if isinstance(obj, CLF):
        return CLF_Visualizer(obj)
    elif isinstance(obj, CLFS):
        return CLFS_Visualizer(obj)
    else:
        raise AttributeError()
