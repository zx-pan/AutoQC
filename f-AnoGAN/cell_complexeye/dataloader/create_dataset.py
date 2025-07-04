from torch.utils.data import Dataset
import numpy as np
import torch
import SimpleITK as sitk
import torchio as tio

sitk.ProcessObject.SetGlobalDefaultThreader("Platform")
from multiprocessing import Manager


def Train(csv, cfg, preload=True):
    subjects = []
    for _, sub in csv.iterrows():
        subject_dict = {
            'vol': tio.ScalarImage(sub.img_path, reader=sitk_reader),
            'path': sub.img_path
        }

        subject = tio.Subject(subject_dict)
        subjects.append(subject)

    if preload:
        manager = Manager()
        cache = DatasetCache(manager)
        ds = tio.SubjectsDataset(subjects, transform=get_transform(cfg))
        ds = preload_wrapper(ds, cache, augment=get_augment(cfg))
    else:
        ds = tio.SubjectsDataset(subjects, transform=tio.Compose([get_transform(cfg), get_augment(cfg)]))

    return ds


def Eval(csv, cfg):
    subjects = []
    for _, sub in csv.iterrows():
        subject_dict = {
            'vol': tio.ScalarImage(sub.img_path, reader=sitk_reader),
            'vol_orig': tio.ScalarImage(sub.img_path, reader=sitk_reader),
            'seg_available': False,
            'ID': sub.img_path.split('/')[-2],
            'ved_num': sub.img_path.split('/')[-1].split('.')[0],
            'label': sub.label,
            'Dataset': sub.setname,
            'stage': sub.settype,
            'path': sub.img_path}
        if sub.seg_path != None and not isinstance(sub.seg_path, float) :  # if we have segmentations
            subject_dict['seg'] = tio.LabelMap(sub.seg_path, reader=sitk_reader),
            subject_dict['seg_orig'] = tio.LabelMap(sub.seg_path,
                                                    reader=sitk_reader)  # we need the image in original size for evaluation
            subject_dict['seg_available'] = True

        subject = tio.Subject(subject_dict)
        subjects.append(subject)
    ds = tio.SubjectsDataset(subjects, transform=get_transform(cfg, state='eval'))
    return ds


# got it from https://discuss.pytorch.org/t/best-practice-to-cache-the-entire-dataset-during-first-epoch/19608/12
class DatasetCache(object):
    def __init__(self, manager, use_cache=True):
        self.use_cache = use_cache
        self.manager = manager
        self._dict = manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        if not self.use_cache:
            raise AttributeError('Data caching is disabled and get funciton is unavailable! Check your config.')
        return self._dict[str(key)]

    def cache(self, key, subject):
        # only store if full data in memory is enabled
        if not self.use_cache:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = (subject)


class preload_wrapper(Dataset):
    def __init__(self, ds, cache, augment=None):
        self.cache = cache
        self.ds = ds
        self.augment = augment

    def reset_memory(self):
        self.cache.reset()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        if self.cache.is_cached(index):
            subject = self.cache.get(index)
        else:
            subject = self.ds.__getitem__(index)
            self.cache.cache(index, subject)
        if self.augment:
            subject = self.augment(subject)
        return subject


def get_transform(cfg, state='train'):  # only transforms that are applied once before preloading
    h, w, d = tuple(cfg.get('imageDim', (256, 256, 1)))

    if not cfg.resizedEvaluation:
        exclude_from_resampling = ['vol_orig', 'seg_orig']
    else:
        exclude_from_resampling = None

    preprocess = tio.Compose([
        tio.RescaleIntensity((0, 1), percentiles=(cfg.get('perc_low', 0), cfg.get('perc_high', 100))),
        tio.CropOrPad((h, w, d), padding_mode=0),
    ])

    if state == 'eval':
        preprocess = tio.Compose([
            tio.RescaleIntensity(
                (0, 1),
                percentiles=(cfg.get('perc_low', 0), cfg.get('perc_high', 100)),
                include=('vol', 'vol_orig')
            )
        ])

    return preprocess


def get_augment(cfg):  # augmentations that may change every epoch
    augmentations = []

    # individual augmentations
    augment = tio.Compose(augmentations)
    return augment


def sitk_reader(path):
    image_nii = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if not 'mask' in str(path) and not 'seg' in str(path):  # only for volumes / scalar images
        image_nii = sitk.CurvatureFlow(image1=image_nii, timeStep=0.125, numberOfIterations=3)
    # sitk.GetArrayFromImage(image_nii) -> (800, 1280)

    vol = sitk.GetArrayFromImage(image_nii)
    vol = np.expand_dims(vol, axis=2)

    return vol, None
