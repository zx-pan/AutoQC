from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from typing import Optional
import pandas as pd
import src.datamodules.create_dataset as create_dataset


class DNA_ANO(LightningDataModule):
    def __init__(self, cfg, fold= None):
        super(DNA_ANO, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        # load data paths and indices
        self.imgpath = {}
        self.csvpath_val = cfg.path.DNA_ANO.IDs.val[fold]
        self.csvpath_test = cfg.path.DNA_ANO.IDs.test[fold]
        self.csv = {}
        states = ['val','test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)

        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'DNA_ANO'

            self.csv[state]['img_path'] = cfg.path.pathBase + '/' + self.csv[state]['img_path']

            # Handle seg_path safely !!!
            try:
                self.csv[state]['seg_path'] = cfg.path.pathBase + '/' + self.csv[state]['seg_path']
            except KeyError:  # If 'seg_path' column is missing
                self.csv[state]['seg_path'] = None  # Add it with None or NaN values

            # Handle classification label safely !!!
            try:
                self.csv[state]['label'] = self.csv[state]['label']
            except KeyError:  # If 'label' column is missing
                self.csv[state]['label'] = 0

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'val_eval'):
            if self.cfg.sample_set: # for debugging
                self.val_eval = create_dataset.Eval(self.csv['val'][0:1], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:1], self.cfg)
            else :
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

