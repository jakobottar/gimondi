from itertools import cycle

class SemiSupervisedDataLoader:
    def __init__(self, sup_dataloader, unsup_dataloader, mode = 'semi'):
        if mode == 'semi':
            self._len = max(len(sup_dataloader), len(unsup_dataloader))
        else: 
            self._len = len(sup_dataloader)
        
        self.mode = mode
        self._sup = sup_dataloader
        self._unsup = unsup_dataloader
        self._sup_it = None
        self._unsup_it = None
    
    def __iter__(self):
        if self.mode == 'semi':
            if len(self._unsup) > len(self._sup):
                self._sup_it = cycle(self._sup)
                self._unsup_it = iter(self._unsup)
            else:
                self._sup_it = iter(self._sup)
                self._unsup_it = cycle(self._unsup)
        else:
            self._sup_it = iter(self._sup)

        return self
    
    def __next__(self):
        r"""
        Returns a tuple of the form (Labeled Item, Unlabeled Item)
        """
        if self.mode == 'semi':
            return (next(self._sup_it), next(self._unsup_it))
        else:
            return (next(self._sup_it), None)
    
    def __len__(self):
        return self._len
        