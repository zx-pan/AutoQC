from torch import nn
import torch
from skimage.measure import regionprops, label
from torchvision.transforms import ToTensor, ToPILImage
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import  confusion_matrix, roc_curve, accuracy_score, precision_recall_fscore_support, auc,precision_recall_curve, average_precision_score
import wandb 
import monai
from torch.nn import functional as F
from PIL import Image

import matplotlib.colors as colors
import torchgeometry as tgm


def _test_step(self, final_volume, data_orig, data_seg, data_mask, batch_idx, ID, label_vol, ved_num) :
        self.healthy_sets = ['DNA']

        # Resize the images if desired
        if not self.cfg.resizedEvaluation: # in case of full resolution evaluation 
            final_volume = F.interpolate(final_volume, size=self.new_size, mode="trilinear",align_corners=True).squeeze() # resize
        else: 
            final_volume = final_volume.squeeze(0).squeeze(0)
        
        # calculate the residual image
        if self.cfg.get('residualmode','l1'): # l1 or l2 residual
            diff_volume = torch.abs((data_orig-final_volume))
        elif self.cfg.get('residualmode','l1') == 'ssim':
            print('using ssim')
            ssim = tgm.losses.SSIM(window_size=self.cfg.get('ssim_window_size', 5),reduction='none')
            diff_volume = ssim(data_orig.squeeze(0).permute(3,0,1,2),final_volume.unsqueeze(0).permute(3,0,1,2))
            diff_volume = diff_volume.permute(1,2,3,0).squeeze(0)
        elif self.cfg.get('residualmode','l1') == 'l1+ssim':
            print('using l1+ssim')
            diff_volume = torch.abs((data_orig.squeeze(0)-final_volume.unsqueeze(0))).squeeze(0)
            ssim = tgm.losses.SSIM(window_size=self.cfg.get('ssim_window_size', 5),reduction='none')
            diff_volume_ssim = ssim(data_orig.squeeze(0).permute(3,0,1,2),final_volume.unsqueeze(0).permute(3,0,1,2))
            diff_volume_ssim = diff_volume_ssim.permute(1,2,3,0).squeeze(0)
            diff_volume_tmp = self.cfg.get('alpha', 0.5) * diff_volume_ssim + (1 - self.cfg.get('alpha', 0.5) ) * diff_volume
            diff_volume = diff_volume_tmp
        else:
            diff_volume = (data_orig-final_volume)**2

        # move data to CPU
        data_seg = data_seg.cpu()
        data_mask = data_mask.cpu()
        diff_volume = diff_volume.cpu()
        data_orig = data_orig.cpu()
        final_volume = final_volume.cpu()
        # binarize the segmentation
        data_seg[data_seg > 0] = 1
        data_mask[data_mask > 0] = 1

        # Filter the DifferenceImage
        if self.cfg['medianFiltering']:
            diff_volume = torch.from_numpy(apply_3d_median_filter(diff_volume.numpy().squeeze(0).squeeze(0),kernelsize=self.cfg.get('kernelsize_median',5))).unsqueeze(0) # bring back to tensor

        ### Compute Metrics per Volume / Step ###
        if self.cfg.evalSeg and self.dataset[0] not in self.healthy_sets and np.array(data_seg.squeeze().flatten()).astype(bool).any(): # only compute metrics if segmentation is available
            # Pixel-Wise Segmentation Error Metrics based on Differenceimage
            AUC, _fpr, _tpr, _threshs = compute_roc(diff_volume.squeeze(0).squeeze(0).flatten(), np.array(data_seg.squeeze(0).squeeze(0).flatten()).astype(bool))
            AUPRC, _precisions, _recalls, _threshs = compute_prc(diff_volume.squeeze(0).squeeze(0).flatten(),np.array(data_seg.squeeze(0).squeeze(0).flatten()).astype(bool))

            # gready search for threshold
            bestDice, bestThresh = find_best_val(np.array(diff_volume.squeeze(0).squeeze(0)).flatten(),  # threshold search with a subset of EvaluationSet
                                                np.array(data_seg.squeeze(0).squeeze(0)).flatten().astype(bool),
                                                val_range=(0, np.max(np.array(diff_volume))),
                                                max_steps=10, 
                                                step=0, 
                                                max_val=0, 
                                                max_point=0)

            if 'test' in self.stage:
                bestThresh = self.threshold['total']

            if self.cfg["threshold"] == 'auto':
                diffs_thresholded = diff_volume > bestThresh
            else: # never used
                diffs_thresholded = diff_volume > self.cfg["threshold"]    
            
            # Connected Components
            if not 'node' in self.dataset[0].lower(): # no 3D data
                diffs_thresholded = filter_3d_connected_components(diffs_thresholded.squeeze(0)) # this is only done for patient-wise evaluation atm

            # save image grid
            if self.cfg['saveOutputImages']:
                log_images(self, diff_volume, data_orig, data_seg, diffs_thresholded, final_volume, ID, ved_num)
            
            # Calculate Dice Score with thresholded volumes
            diceScore = dice(np.array(diffs_thresholded.squeeze(0).squeeze(0)),np.array(data_seg.squeeze(0).squeeze(0).flatten()).astype(bool))
            
            # Other Metrics
            self.eval_dict['lesionSizePerVol'].append(np.count_nonzero(np.array(data_seg.squeeze().flatten()).astype(bool)))
            self.eval_dict['DiceScorePerVol'].append(diceScore)
            self.eval_dict['BestDicePerVol'].append(bestDice)
            self.eval_dict['BestThresholdPerVol'].append(bestThresh)
            self.eval_dict['AUCPerVol'].append(AUC)
            self.eval_dict['AUPRCPerVol'].append(AUPRC)
            self.eval_dict['IDs'].append(ID[0])

            # other metrics from monai:
            if len(data_seg.shape) == 4:
                data_seg = data_seg.unsqueeze(0)
            Haus = monai.metrics.compute_hausdorff_distance(diffs_thresholded.unsqueeze(0).unsqueeze(0),data_seg, include_background=False, distance_metric='euclidean', percentile=None, directed=False)
            self.eval_dict['HausPerVol'].append(Haus.item())

            # compute slice-wise metrics
            for slice in range(data_seg.squeeze().shape[0]):
                # if is normal, we do not want to compute the segmentation metrics
                if np.array(data_seg.squeeze()[slice].flatten()).astype(bool).any():
                    self.eval_dict['DiceScorePerSlice'].append(dice(np.array(diff_volume.squeeze()[slice] > bestThresh),np.array(data_seg.squeeze()[slice].flatten()).astype(bool)))
                    PrecRecF1PerSlice = precision_recall_fscore_support(np.array(data_seg.squeeze()[slice].flatten()).astype(bool),np.array(diff_volume.squeeze()[slice] > bestThresh).flatten(),warn_for=tuple(),labels=[0,1])
                    self.eval_dict['PrecisionPerSlice'].append(PrecRecF1PerSlice[0][1])
                    self.eval_dict['RecallPerSlice'].append(PrecRecF1PerSlice[1][1])
                    self.eval_dict['lesionSizePerSlice'].append(np.count_nonzero(np.array(data_seg.squeeze()[slice].flatten()).astype(bool)))

        if 'val' in self.stage:
            if batch_idx == 0:
                self.diffs_list = np.array(diff_volume.squeeze().flatten())
                self.seg_list = np.array(data_seg.squeeze().flatten()).astype(np.int8)
            else: 
                self.diffs_list = np.append(self.diffs_list,np.array(diff_volume.squeeze().flatten()),axis=0)
                self.seg_list = np.append(self.seg_list,np.array(data_seg.squeeze().flatten()),axis=0).astype(np.int8)

        # sample-Wise Anomalyscores
        if self.cfg.get('use_postprocessed_score', True):
            AnomalyScoreMean_vol = diff_volume.squeeze().mean().item()  # for sample-wise detection
            AnomalyScoreMax_vol = diff_volume.squeeze().max().item()  # for sample-wise detection

            # use sliding window for anomaly score (report the largest anomaly score in the window)
            h, w = diff_volume.squeeze().shape
            window_size = 16
            stride = 8
            AnomalyScorePatch = []
            for i in range(0, h - window_size + 1, stride):
                for j in range(0, w - window_size + 1, stride):
                    AnomalyScorePatch.append(diff_volume.squeeze()[i:i + window_size, j:j + window_size].mean())
            AnomalyScorePatch_vol = np.max(AnomalyScorePatch)

            self.eval_dict['AnomalyScoreMeanPerVol'].append(AnomalyScoreMean_vol)
            self.eval_dict['AnomalyScoreMaxPerVol'].append(AnomalyScoreMax_vol)
            self.eval_dict['AnomalyScorePatchPerVol'].append(AnomalyScorePatch_vol)

            # save to csv with ID and ved_num
            with open('diff_volume.csv', 'a') as f:
                f.write(f'{ID[0]},{ved_num[0]},{AnomalyScoreMean_vol},{AnomalyScoreMax_vol},{AnomalyScorePatch_vol}\n')

        self.eval_dict['labelPerVol'].append(label_vol.item())


def _test_end(self) :
    # average over all test samples

        self.eval_dict['AUPRCPerVolMean'] = np.nanmean(self.eval_dict['AUPRCPerVol'])
        self.eval_dict['AUPRCPerVolStd'] = np.nanstd(self.eval_dict['AUPRCPerVol'])
        self.eval_dict['AUCPerVolMean'] = np.nanmean(self.eval_dict['AUCPerVol'])
        self.eval_dict['AUCPerVolStd'] = np.nanstd(self.eval_dict['AUCPerVol'])

        self.eval_dict['DicePerVolMean'] = np.nanmean(self.eval_dict['DiceScorePerVol'])
        self.eval_dict['DicePerVolStd'] = np.nanstd(self.eval_dict['DiceScorePerVol'])
        self.eval_dict['BestDicePerVolMean'] = np.mean(self.eval_dict['BestDicePerVol'])
        self.eval_dict['BestDicePerVolStd'] = np.std(self.eval_dict['BestDicePerVol'])
        self.eval_dict['BestThresholdPerVolMean'] = np.mean(self.eval_dict['BestThresholdPerVol'])
        self.eval_dict['BestThresholdPerVolStd'] = np.std(self.eval_dict['BestThresholdPerVol'])

        self.eval_dict['HausPerVolMean'] = np.nanmean(np.array(self.eval_dict['HausPerVol'])[np.isfinite(self.eval_dict['HausPerVol'])])
        self.eval_dict['HausPerVolStd'] = np.nanstd(np.array(self.eval_dict['HausPerVol'])[np.isfinite(self.eval_dict['HausPerVol'])])

        self.eval_dict['PrecisionPerVolMean'] = np.mean(self.eval_dict['PrecisionPerVol'])
        self.eval_dict['PrecisionPerVolStd'] =np.std(self.eval_dict['PrecisionPerVol'])
        self.eval_dict['RecallPerVolMean'] = np.mean(self.eval_dict['RecallPerVol'])
        self.eval_dict['RecallPerVolStd'] = np.std(self.eval_dict['RecallPerVol'])
        self.eval_dict['PrecisionPerSliceMean'] = np.mean(self.eval_dict['PrecisionPerSlice'])
        self.eval_dict['PrecisionPerSliceStd'] = np.std(self.eval_dict['PrecisionPerSlice'])
        self.eval_dict['RecallPerSliceMean'] = np.mean(self.eval_dict['RecallPerSlice'])
        self.eval_dict['RecallPerSliceStd'] = np.std(self.eval_dict['RecallPerSlice'])
        self.eval_dict['AccuracyPerVolMean'] = np.mean(self.eval_dict['AccuracyPerVol'])
        self.eval_dict['AccuracyPerVolStd'] = np.std(self.eval_dict['AccuracyPerVol'])

        if 'test' in self.stage:
            # cal classification metrics
            AUC_mean, _fpr, _tpr, _threshs = compute_roc(np.array(self.eval_dict['AnomalyScoreMeanPerVol']), np.array(self.eval_dict['labelPerVol']))
            AUPRC_mean, _precisions, _recalls, _threshs = compute_prc(np.array(self.eval_dict['AnomalyScoreMeanPerVol']),np.array(self.eval_dict['labelPerVol']))
            pred_mean = np.array(self.eval_dict['AnomalyScoreMeanPerVol']) > self.threshold['total_class_mean']
            self.eval_dict['ACCClassificationMeanPerVol'] = accuracy_score(np.array(self.eval_dict['labelPerVol']), pred_mean)
            self.eval_dict['AUCClassificationMeanPerVol'] = AUC_mean
            self.eval_dict['AUPRCClassificationMeanPerVol'] = AUPRC_mean
            AUC_max, _fpr, _tpr, _threshs = compute_roc(np.array(self.eval_dict['AnomalyScoreMaxPerVol']), np.array(self.eval_dict['labelPerVol']))
            AUPRC_max, _precisions, _recalls, _threshs = compute_prc(np.array(self.eval_dict['AnomalyScoreMaxPerVol']),np.array(self.eval_dict['labelPerVol']))
            pred_max = np.array(self.eval_dict['AnomalyScoreMaxPerVol']) > self.threshold['total_class_max']
            self.eval_dict['ACCClassificationMaxPerVol'] = accuracy_score(np.array(self.eval_dict['labelPerVol']), pred_max)
            self.eval_dict['AUCClassificationMaxPerVol'] = AUC_max
            self.eval_dict['AUPRCClassificationMaxPerVol'] = AUPRC_max
            AUC_patch, _fpr, _tpr, _threshs = compute_roc(np.array(self.eval_dict['AnomalyScorePatchPerVol']), np.array(self.eval_dict['labelPerVol']))
            AUPRC_patch, _precisions, _recalls, _threshs = compute_prc(np.array(self.eval_dict['AnomalyScorePatchPerVol']),np.array(self.eval_dict['labelPerVol']))
            pred_patch = np.array(self.eval_dict['AnomalyScorePatchPerVol']) > self.threshold['total_class_patch']
            self.eval_dict['ACCClassificationPatchPerVol'] = accuracy_score(np.array(self.eval_dict['labelPerVol']), pred_patch)
            self.eval_dict['AUCClassificationPatchPerVol'] = AUC_patch
            self.eval_dict['AUPRCClassificationPatchPerVol'] = AUPRC_patch

            del self.threshold

        if 'val' in self.stage: 
            if self.dataset[0] not in self.healthy_sets:
                bestdiceScore, bestThresh = find_best_val((self.diffs_list).flatten(), (self.seg_list).flatten().astype(bool), 
                                        val_range=(0, np.max((self.diffs_list))), 
                                        max_steps=10, 
                                        step=0, 
                                        max_val=0, 
                                        max_point=0)

                self.threshold['total'] = bestThresh
                bestaccmean, bestmeanThresh = find_best_class_threshold(np.array(self.eval_dict['AnomalyScoreMeanPerVol']), np.array(self.eval_dict['labelPerVol']),
                                        val_range=(0, np.max((self.eval_dict['AnomalyScoreMeanPerVol']))),
                                        max_steps=10,
                                        step=0,
                                        max_acc=0,
                                        best_threshold=0)
                self.threshold['total_class_mean'] = bestmeanThresh
                bestaccmax, bestmaxThresh = find_best_class_threshold(np.array(self.eval_dict['AnomalyScoreMaxPerVol']), np.array(self.eval_dict['labelPerVol']),
                                        val_range=(0, np.max((self.eval_dict['AnomalyScoreMaxPerVol']))),
                                        max_steps=10,
                                        step=0,
                                        max_acc=0,
                                        best_threshold=0)
                self.threshold['total_class_max'] = bestmaxThresh
                bestaccpatch, bestpatchThresh = find_best_class_threshold(np.array(self.eval_dict['AnomalyScorePatchPerVol']), np.array(self.eval_dict['labelPerVol']),
                                        val_range=(0, np.max((self.eval_dict['AnomalyScorePatchPerVol']))),
                                        max_steps=10,
                                        step=0,
                                        max_acc=0,
                                        best_threshold=0)
                self.threshold['total_class_patch'] = bestpatchThresh

                if self.cfg.get('KLDBackprop',False): 
                    bestdiceScoreKLComb, bestThreshKLComb = find_best_val((self.diffs_listKLComb).flatten(), (self.seg_list).flatten().astype(bool), 
                        val_range=(0, np.max((self.diffs_listKLComb))), 
                        max_steps=10, 
                        step=0, 
                        max_val=0, 
                        max_point=0)

                    self.threshold['totalKLComb'] = bestThreshKLComb 
                    bestdiceScoreKL, bestThreshKL = find_best_val((self.diffs_listKL).flatten(), (self.seg_list).flatten().astype(bool), 
                        val_range=(0, np.max((self.diffs_listKL))), 
                        max_steps=10, 
                        step=0, 
                        max_val=0, 
                        max_point=0)

                    self.threshold['totalKL'] = bestThreshKL 
            else: # define thresholds based on the healthy validation set
                _, fpr_healthy, _, threshs = compute_roc((self.diffs_list).flatten(), np.zeros_like(self.diffs_list).flatten().astype(int))
                self.threshholds_healthy= {
                        'thresh_1p' : threshs[np.argmax(fpr_healthy>0.01)], # 1%
                        'thresh_5p' : threshs[np.argmax(fpr_healthy>0.05)], # 5%
                        'thresh_10p' : threshs[np.argmax(fpr_healthy>0.10)]} # 10%}
                self.eval_dict['t_1p'] = self.threshholds_healthy['thresh_1p']
                self.eval_dict['t_5p'] = self.threshholds_healthy['thresh_5p']
                self.eval_dict['t_10p'] = self.threshholds_healthy['thresh_10p']


def get_eval_dictionary():
    _eval = {
        'IDs': [],
        'x': [],
        'reconstructions': [],
        'diffs': [],
        'diffs_volume': [],
        'Segmentation': [],
        'reconstructionTimes': [],
        'latentSpace': [],
        'Age': [],
        'AgeGroup': [],
        'HausPerVol': [],

        'PrecisionPerVol': [],
        'RecallPerVol': [],
        'PrecisionPerSlice': [],
        'RecallPerSlice': [],
        'lesionSizePerSlice': [],
        'lesionSizePerVol': [],
        'Dice': [],
        'DiceScorePerSlice': [],
        'DiceScorePerVol': [],
        'BestDicePerVol': [],
        'BestThresholdPerVol': [],
        'AUCPerVol': [],
        'AUPRCPerVol': [],
        'SpecificityPerVol': [],
        'AccuracyPerVol': [],
        'DicegradELBO': [],
        'DiceScorePerVolgradELBO': [],
        'BestDicePerVolgradELBO': [],
        'BestThresholdPerVolgradELBO': [],
        'AUCPerVolgradELBO': [],
        'AUPRCPerVolgradELBO': [],
        'KLD_to_learned_prior':[],

        'AUCAnomalyCombPerSlice': [], # PerVol!!! + Confusionmatrix.
        'AUPRCAnomalyCombPerSlice': [],
        'AnomalyScoreCombPerSlice': [],


        'AUCAnomalyKLDPerSlice': [],
        'AUPRCAnomalyKLDPerSlice': [],
        'AnomalyScoreKLDPerSlice': [],


        'AUCAnomalyRecoPerSlice': [],
        'AUPRCAnomalyRecoPerSlice': [],
        'AnomalyScoreRecoPerSlice': [],
        'AnomalyScoreRecoBinPerSlice': [],
        'AnomalyScoreAgePerSlice': [],
        'AUCAnomalyAgePerSlice': [],
        'AUPRCAnomalyAgePerSlice': [],

        'labelPerSlice' : [],
        'labelPerVol' : [],
        'AnomalyScoreCombPerVol' : [],
        'AnomalyScoreCombiPerVol' : [],
        'AnomalyScoreCombMeanPerVol' : [],
        'AnomalyScoreRegPerVol' : [],
        'AnomalyScoreRegMeanPerVol' : [],
        'AnomalyScoreRecoPerVol' : [],
        'AnomalyScoreCombPriorPerVol': [],
        'AnomalyScoreCombiPriorPerVol': [],
        'AnomalyScoreAgePerVol' : [],
        'AnomalyScoreRecoMeanPerVol' : [],
        'DiceScoreKLPerVol': [],
        'DiceScoreKLCombPerVol': [],
        'BestDiceKLCombPerVol': [],
        'BestDiceKLPerVol': [],
        'AUCKLCombPerVol': [],
        'AUPRCKLCombPerVol': [],
        'AUCKLPerVol': [],
        'AUPRCKLPerVol': [],

        'AUCClassificationMeanPerVol': [],
        'AUPRCClassificationMeanPerVol': [],
        'AUCClassificationMaxPerVol': [],
        'AUPRCClassificationMaxPerVol': [],
        'AUCClassificationPatchPerVol': [],
        'AUPRCClassificationPatchPerVol': [],
        'ACCClassificationMeanPerVol': [],
        'ACCClassificationMaxPerVol': [],
        'ACCClassificationPatchPerVol': [],
        'AnomalyScoreMeanPerVol': [],
        'AnomalyScoreMaxPerVol': [],
        'AnomalyScorePatchPerVol': [],

    }
    return _eval


def apply_3d_median_filter(volume, kernelsize=5):  # kernelsize 5 works quite well
    volume = scipy.ndimage.filters.median_filter(volume, (kernelsize, kernelsize, kernelsize))
    return volume


def apply_2d_median_filter(volume, kernelsize=5):  # kernelsize 5 works quite well
    img = scipy.ndimage.filters.median_filter(volume, (kernelsize, kernelsize))
    return img


def squash_intensities(img):
    # logistic function intended to squash reconstruction errors from [0;0.2] to [0;1] (just an example)
    k = 100
    offset = 0.5
    return 2.0 * ((1.0 / (1.0 + np.exp(-k * img))) - offset)


def filter_3d_connected_components(volume):
    sz = None
    if volume.ndim > 3:
        sz = volume.shape
        volume = np.reshape(volume, [sz[0] * sz[1], sz[2], sz[3]])

    cc_volume = label(volume, connectivity=3)
    props = regionprops(cc_volume)
    for prop in props:
        if prop['filled_area'] <= 7:
            volume[cc_volume == prop['label']] = 0

    if sz is not None:
        volume = np.reshape(volume, [sz[0], sz[1], sz[2], sz[3]])
    return volume


def find_best_class_threshold(x, y, val_range=(0, 1), max_steps=4, step=0, max_acc=0, best_threshold=0):
    """
    Find the threshold that maximizes accuracy for anomaly scores x and labels y.

    Args:
        x (list or np.array): Anomaly scores [x1, x2, ..., xn].
        y (list or np.array): Binary labels [0, 1, 0, ...].
        val_range (tuple): Range of threshold values to search (default is [0, 1]).
        max_steps (int): Maximum number of iterations (default is 4).
        step (int): Current step in the iteration (used for recursion).
        max_acc (float): Maximum accuracy found so far.
        best_threshold (float): Threshold corresponding to max_acc.

    Returns:
        max_acc (float): Maximum accuracy achieved.
        best_threshold (float): Threshold that maximizes accuracy.
    """
    if step == max_steps:
        return max_acc, best_threshold

    # Calculate quartiles within the current range
    bottom, top = val_range
    center = bottom + (top - bottom) * 0.5
    q_bottom = bottom + (top - bottom) * 0.25
    q_top = bottom + (top - bottom) * 0.75

    # Calculate accuracy for the lower quartile threshold
    pred_bottom = (x > q_bottom).astype(int)
    acc_bottom = np.mean(pred_bottom == y)

    # Calculate accuracy for the upper quartile threshold
    pred_top = (x > q_top).astype(int)
    acc_top = np.mean(pred_top == y)

    # Determine which quartile gives better accuracy
    if acc_bottom >= acc_top:
        if acc_bottom > max_acc:
            max_acc = acc_bottom
            best_threshold = q_bottom
        # Narrow the search range to the lower half
        return find_best_class_threshold(x, y, val_range=(bottom, center), max_steps=max_steps, step=step + 1,
                                   max_acc=max_acc, best_threshold=best_threshold)
    else:
        if acc_top > max_acc:
            max_acc = acc_top
            best_threshold = q_top
        # Narrow the search range to the upper half
        return find_best_class_threshold(x, y, val_range=(center, top), max_steps=max_steps, step=step + 1,
                                   max_acc=max_acc, best_threshold=best_threshold)


# From Zimmerer iterative algorithm for threshold search
def find_best_val(x, y, val_range=(0, 1), max_steps=4, step=0, max_val=0, max_point=0):  #x: Image , y: Label
    if step == max_steps:
        return max_val, max_point

    if val_range[0] == val_range[1]:
        val_range = (val_range[0], 1)

    bottom = val_range[0]
    top = val_range[1]
    center = bottom + (top - bottom) * 0.5

    q_bottom = bottom + (top - bottom) * 0.25
    q_top = bottom + (top - bottom) * 0.75
    val_bottom = dice(x > q_bottom, y)
    val_top = dice(x > q_top, y)

    if val_bottom >= val_top:
        if val_bottom >= max_val:
            max_val = val_bottom
            max_point = q_bottom
        return find_best_val(x, y, val_range=(bottom, center), step=step + 1, max_steps=max_steps,
                             max_val=max_val, max_point=max_point)
    else:
        if val_top >= max_val:
            max_val = val_top
            max_point = q_top
        return find_best_val(x, y, val_range=(center, top), step=step + 1, max_steps=max_steps,
                             max_val=max_val,max_point=max_point)


def dice(P, G):
    psum = np.sum(P.flatten())
    gsum = np.sum(G.flatten())
    pgsum = np.sum(np.multiply(P.flatten(), G.flatten()))
    score = (2 * pgsum) / (psum + gsum)
    return score

    
def compute_roc(predictions, labels):
    _fpr, _tpr, _ = roc_curve(labels.astype(int), predictions,pos_label=1)
    roc_auc = auc(_fpr, _tpr)
    return roc_auc, _fpr, _tpr, _


def compute_prc(predictions, labels):
    precisions, recalls, thresholds = precision_recall_curve(labels.astype(int), predictions)
    auprc = average_precision_score(labels.astype(int), predictions)
    return auprc, precisions, recalls, thresholds   


# Dice Score 
def xfrange(start, stop, step):
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1


def tpr(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fn = np.sum(np.multiply(np.invert(P.flatten()), G.flatten()))
    return tp / (tp + fn)


def fpr(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
    return fp / (fp + tp)


def normalize(tensor): # THanks DZimmerer
    tens_deta = tensor.detach().cpu()
    tens_deta -= float(np.min(tens_deta.numpy()))
    tens_deta /= float(np.max(tens_deta.numpy()))

    return tens_deta


def log_images(self, diff_volume, data_orig, data_seg, seg_pred, final_volume, ID, ved_num):
    image_root = os.path.join(os.getcwd(), 'grid')
    save_dir = os.path.join(image_root, 'fold_{}'.format(self.fold))
    save_dir = os.path.join(save_dir, ID[0])
    os.makedirs(save_dir, exist_ok=True)

    img_orig = data_orig.squeeze(0).squeeze(0)[..., 0]
    img_recon = final_volume[..., 0]
    img_diff = diff_volume.squeeze(0).squeeze(0)[..., 0]
    img_seg = data_seg.squeeze(0).squeeze(0)[..., 0]
    img_pred = seg_pred.squeeze(0).squeeze(0)[..., 0]

    def save_single_image(img, cmap, norm, filename, force_gray_bounds=False):
        plt.figure(figsize=(4, 4))

        if force_gray_bounds and cmap == 'gray':
            plt.imshow(img, cmap=cmap, vmin=0, vmax=1)
        else:
            plt.imshow(img, cmap=cmap, norm=norm)

        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    if self.cfg.get('save_to_disc', True):
        save_single_image(img_orig, cmap='gray', norm=None,
                          filename=os.path.join(save_dir, f'{ved_num[0]}_orig.png'))

        save_single_image(img_recon, cmap='gray', norm=None,
                          filename=os.path.join(save_dir, f'{ved_num[0]}_recon.png'))

        save_single_image(img_diff, cmap='inferno',
                          norm=colors.Normalize(vmin=0, vmax=img_diff.max() + .05),
                          filename=os.path.join(save_dir, f'{ved_num[0]}_diff_w_norm.png'))

        save_single_image(img_diff, cmap='inferno',
                          norm=colors.Normalize(vmin=0, vmax=0.6),
                          filename=os.path.join(save_dir, f'{ved_num[0]}_diff_wo_norm.png'))

        save_single_image(img_seg, cmap='gray', norm=None,
                          filename=os.path.join(save_dir, f'{ved_num[0]}_mask.png'),
                          force_gray_bounds=True)

        save_single_image(img_pred, cmap='gray', norm=None,
                          filename=os.path.join(save_dir, f'{ved_num[0]}_pred.png'),
                          force_gray_bounds=True)
