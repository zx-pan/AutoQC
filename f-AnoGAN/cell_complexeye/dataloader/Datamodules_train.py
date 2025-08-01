from torch.utils.data import DataLoader, random_split
import cell_complexeye.dataloader.create_dataset as create_dataset
from typing import Optional
import pandas as pd


class DNA:
    def __init__(self, cfg, fold = None):
        super(DNA, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        # load data paths and indices

        self.cfg.permute = False

        self.imgpath = {}
        self.csvpath_train = cfg.path.DNA.IDs.train[fold]
        self.csvpath_val = cfg.path.DNA.IDs.val[fold]
        self.csv = {}
        states = ['train','val']

        self.csv['train'] = pd.read_csv(self.csvpath_train)
        self.csv['val'] = pd.read_csv(self.csvpath_val)

        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'DNA'

            self.csv[state]['img_path'] = cfg.path.pathBase + '/' +  self.csv[state]['img_path']
            self.csv[state]['seg_path'] = None

            self.csv[state]['label'] = 0

        self.setup()

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'train'):
            if self.cfg.sample_set: # for debugging
                print('loading small dataset')
                self.train = create_dataset.Train(self.csv['train'][0:50],self.cfg) 
                self.val = create_dataset.Train(self.csv['val'][0:50],self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'][0:50],self.cfg)
            else:
                print('loading full dataset')
                self.train = create_dataset.Train(self.csv['train'],self.cfg) 
                self.val = create_dataset.Train(self.csv['val'],self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'],self.cfg)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast',False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)
      
