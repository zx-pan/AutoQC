import torch
import torch.nn as nn
from torch.utils.model_zoo import tqdm
from monai.inferers import sliding_window_inference
from utils_eval import _test_step, _test_end, get_eval_dictionary
import torchio as tio
import os
import pprint


class fANOGAN(nn.Module):
    def __init__(self, cfg, prefix=None):
        super().__init__()

        self.cfg = cfg
        self.eval_dict = get_eval_dictionary()
        if not hasattr(self, 'threshold'):
            self.threshold = {}
        self.fold = 0

    def test_anomaly_detection(self, opt, generator, discriminator, encoder,
                                dataloader, device, fold, validate):
            self.fold = fold
            generator.load_state_dict(torch.load("/users/zpan3/afs/Models/f-AnoGAN/cell_complexeye/outputs/2025-02-23/21-08-35/results/generator.pth"))
            discriminator.load_state_dict(torch.load("/users/zpan3/afs/Models/f-AnoGAN/cell_complexeye/outputs/2025-02-23/21-08-35/results/discriminator.pth"))
            encoder.load_state_dict(torch.load("/users/zpan3/afs/Models/f-AnoGAN/cell_complexeye/outputs/2025-02-23/21-08-35/results//encoder.pth"))

            generator.to(device).eval()
            discriminator.to(device).eval()
            encoder.to(device).eval()

            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                self.dataset = batch['Dataset']
                input = batch['vol'][tio.DATA]
                data_orig = batch['vol_orig'][tio.DATA]
                data_seg = batch['seg_orig'][tio.DATA] if batch['seg_available'] else torch.zeros_like(data_orig)
                data_seg[data_seg > 0] = 1
                # data_mask = batch['mask_orig'][tio.DATA]
                data_mask = torch.ones_like(data_orig)

                ID = batch['ID']
                ved_num = batch['ved_num']
                self.stage = batch['stage']
                label = batch['label']

                # reorder depth to batch dimension
                assert input.shape[0] == 1, "Batch size must be 1"
                input = input.squeeze(0).permute(3, 0, 1, 2)  # [B,C,H,W,D] -> [D,C,H,W]

                def generate_rec(input):
                    input = input.to(device)
                    with torch.no_grad():
                        z = encoder(input)
                        rec = generator(z)
                    return rec

                reco = sliding_window_inference(input, predictor=generate_rec, roi_size=(192, 192), sw_batch_size=1, overlap=0.8, mode="gaussian")

                # reassamble the reconstruction volume
                final_volume = reco.clone().squeeze(0)
                final_volume = final_volume.permute(1, 2, 0)  # to HxWxD

                final_volume = final_volume.unsqueeze(0)
                final_volume = final_volume.unsqueeze(0)

                # calculate metrics
                _test_step(self, final_volume, data_orig, data_seg, data_mask, batch_idx, ID, label,
                           ved_num)  # everything that is independent of the model choice
            _test_end(self)

            # save results
            log_dir = '/users/zpan3/afs/Models/f-AnoGAN/cell_complexeye/outputs'
            if not validate:
                out_path = os.path.join(log_dir, f'{fold + 1}_test_preds_dict.txt')
            else:
                out_path = os.path.join(log_dir, f'{fold + 1}_val_preds_dict.txt')

            with open(out_path, 'w') as f:
                pprint.pprint(self.eval_dict, stream=f, width=160, compact=False)