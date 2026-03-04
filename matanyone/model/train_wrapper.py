import logging
from omegaconf import DictConfig
import numpy as np
import torch
from typing import Dict
from torch.nn import functional as F

from einops.layers.torch import Rearrange
from matanyone.model.matanyone import MatAnyone

log = logging.getLogger()


class MatAnyoneTrainWrapper(MatAnyone):
    def __init__(self, cfg: DictConfig, stage_cfg: DictConfig):
        super().__init__(cfg, single_object=(stage_cfg.num_objects == 1))

        self.sensory_dim = cfg.model.sensory_dim
        self.seq_length = stage_cfg.seq_length
        self.num_ref_frames = stage_cfg.num_ref_frames
        self.deep_update_prob = stage_cfg.deep_update_prob
        self.use_amp = stage_cfg.amp

    def forward(self, data: Dict, seg_pass=False, clamp_mat=True, seg_mat=False, sample_id=None):
        self.num_ref_frames = 3
        out = {}
        if sample_id is not None:
            frames = data['rgb'][sample_id] # b t c h w
        else:
            frames = data['rgb'] # b t c h w
        first_frame_gt = data['first_frame_gt'].float()

        b, seq_length = frames.shape[:2]
        self.move_t_out_of_batch = Rearrange('(b t) c h w -> b t c h w', t=seq_length)
        self.move_t_from_batch_to_volume = Rearrange('(b t) c h w -> b c t h w', t=seq_length)
        num_filled_objects = [o.item() for o in data['info']['num_objects']]
        max_num_objects = max(num_filled_objects)
        first_frame_gt = first_frame_gt[:, :, :max_num_objects]
        selector = data['selector'][:, :max_num_objects].unsqueeze(2).unsqueeze(2)

        num_objects = first_frame_gt.shape[2]
        out['num_filled_objects'] = num_filled_objects

        def get_ms_feat_ti(ti):
            return [f[:, ti] for f in ms_feat]

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            frames_flat = frames.view(b * seq_length, *frames.shape[2:]) # (b t) c h w
            ms_feat, pix_feat = self.encode_image(frames_flat, seq_length=seq_length) # f16, f8, f4, f2, f1; p
            
            with torch.cuda.amp.autocast(enabled=False):
                keys, shrinkages, selections = self.transform_key(ms_feat[0].float())

            # ms_feat: tuples of (B*T)*C*H*W -> B*T*C*H*W
            # keys/shrinkages/selections: (B*T)*C*H*W -> B*C*T*H*W
            h, w = keys.shape[-2:]
            keys = self.move_t_from_batch_to_volume(keys)
            shrinkages = self.move_t_from_batch_to_volume(shrinkages)
            selections = self.move_t_from_batch_to_volume(selections)
            ms_feat = [self.move_t_out_of_batch(f) for f in ms_feat]
            
            pix_feat = self.move_t_out_of_batch(pix_feat)

            # zero-init sensory
            sensory = torch.zeros((b, num_objects, self.sensory_dim, h, w), device=frames.device)

            msk_val, sensory, obj_val, _ = self.encode_mask(frames[:, 0], pix_feat[:, 0], sensory,
                                                            first_frame_gt[:, 0])
            masks = first_frame_gt[:, 0] # B N H W

            # init
            last_mask = masks

            # add the time dimension
            msk_values = msk_val.unsqueeze(3)  # B*num_objects*C*T*H*W
            obj_values = obj_val.unsqueeze(2)  # B*num_objects*T*Q*C
        
            for ti in range(0, seq_length):
                
                if ti <= self.num_ref_frames:
                    ref_msk_values = msk_values
                    ref_keys = keys[:, :, :ti]
                    ref_shrinkages = shrinkages[:, :, :ti] if shrinkages is not None else None
                else:
                    
                    ridx = [torch.randperm(ti)[:self.num_ref_frames] for _ in range(b)]
                    ref_msk_values = torch.stack(
                        [msk_values[bi, :, :, ridx[bi]] for bi in range(b)], 0)
                    ref_keys = torch.stack([keys[bi, :, ridx[bi]] for bi in range(b)], 0)
                    ref_shrinkages = torch.stack([shrinkages[bi, :, ridx[bi]] for bi in range(b)],
                                                 0)

                ts_output = None
                
                if ti == 0:
                    readout, aux_input = self.read_first_frame_memory(msk_val, obj_values,
                                                      pix_feat[:, ti], sensory, masks, selector, seg_pass=seg_pass)
                else:
                    readout, aux_input, ts_output = self.read_memory(keys[:, :, ti], selections[:, :,
                                                                                    ti], ref_keys,
                                                        ref_shrinkages, ref_msk_values, obj_values,
                                                        pix_feat[:, ti], sensory, masks, selector, ts_output, seg_pass=seg_pass,
                                                        last_pix_feat=pix_feat[:, ti-1], last_pred_mask=last_mask)
                
                aux_output = self.compute_aux(pix_feat[:, ti], aux_input, selector, seg_pass=seg_pass)
                sensory, logits, masks = self.segment(get_ms_feat_ti(ti),
                                                readout,
                                                sensory,
                                                selector=selector,
                                                seg_pass=seg_pass,
                                                clamp_mat=clamp_mat,
                                                seg_mat=seg_mat)
                # remove background
                masks = masks[:, 1:]

                # update
                last_mask = masks
            
                # No need to encode the last frame
                if ti < (self.seq_length - 1):
                    deep_update = np.random.rand() < self.deep_update_prob
                    msk_val, sensory, obj_val, _ = self.encode_mask(frames[:, ti],
                                                                    pix_feat[:, ti],
                                                                    sensory,
                                                                    masks,
                                                                    deep_update=deep_update)
                                                                    
                    # use predicted to replace first_frame_gt
                    if ti == 0:
                        msk_values = msk_val.unsqueeze(3)  # B*num_objects*C*T*H*W
                        obj_values = obj_val.unsqueeze(2)  # B*num_objects*T*Q*C
                    else:
                        msk_values = torch.cat([msk_values, msk_val.unsqueeze(3)], 3)
                        obj_values = torch.cat([obj_values, obj_val.unsqueeze(2)], 2)
                        
                
                out[f'masks_{ti}'] = masks
                out[f'logits_{ti}'] = logits
                out[f'aux_{ti}'] = aux_output
                out[f'ts_{ti}'] = ts_output

        return out