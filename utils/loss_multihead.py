# ============== MultiHead Loss Implementation ==============
# Added for multi-head detection support
# This is a separate file to avoid modifying the original loss.py

import torch
import torch.nn as nn
from utils.general import bbox_iou
from utils.torch_utils import is_parallel
from utils.loss import smooth_BCE, FocalLoss


class ComputeLossMultiHead:
    """
    Compute loss for MultiHeadDetect (Strategy A)
    Based on ComputeLoss but adapted for multi-head outputs
    """
    
    def __init__(self, model, autobalance=False):
        """Initialize loss computation for multi-head model"""
        super(ComputeLossMultiHead, self).__init__()
        device = next(model.parameters()).device
        h = model.hyp  # hyperparameters
        
        # Define criteria (same as ComputeLoss)
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        
        # Class label smoothing
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))
        
        # Focal loss
        g = h['fl_gamma']
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        
        # Get detection layer
        det = model.model[-1] if is_parallel(model) else model.model[-1]
        
        # Check if multi-head
        from models.yolo import MultiHeadDetect
        if not isinstance(det, MultiHeadDetect):
            raise ValueError("ComputeLossMultiHead requires MultiHeadDetect layer")
        
        self.n_heads = det.n_heads
        self.config = det.config
        
        # Create class masks for each head
        self.class_masks = []
        for head_id in range(self.n_heads):
            mask = self.config.get_head_mask(head_id, device)
            self.class_masks.append(mask)
        
        # Get head weights (normalized)
        self.head_weights = self.config.get_head_weights()
        
        # Balance parameters
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])
        self.ssi = list(det.stride).index(16) if autobalance else 0
        
        # Save attributes
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = \
            BCEcls, BCEobj, model.gr, h, autobalance
        
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))
    
    def __call__(self, p, targets):
        """
        Compute loss for multi-head predictions
        
        Args:
            p: tuple of (reg_obj_outputs, cls_outputs) from MultiHeadDetect
            targets: [image_idx, class, x, y, w, h] ground truth
        
        Returns:
            loss: scalar total loss
            loss_items: tensor [box, obj, cls] losses
        """
        device = targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        
        # Unpack multi-head outputs
        reg_obj_outputs, cls_outputs = p
        
        # Build targets for all heads (shared)
        tcls, tbox, indices, anchors = self.build_targets(reg_obj_outputs, targets)
        
        # Compute shared box and obj losses
        for i, reg_obj in enumerate(reg_obj_outputs):  # layer index
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(reg_obj[..., 0], device=device)  # target obj
            
            n = b.shape[0]  # number of targets
            if n:
                ps = reg_obj[b, a, gj, gi]  # prediction subset
                
                # Box regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox = lbox + (1.0 - iou).mean()
                
                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.gr < 1:
                    score_iou = (1.0 - self.gr) + self.gr * score_iou
                tobj[b, a, gj, gi] = score_iou
            
            obji = self.BCEobj(reg_obj[..., 4], tobj)
            lobj = lobj + obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        
        # Compute classification loss for each head
        for head_id in range(self.n_heads):
            head_lcls = torch.zeros(1, device=device)
            class_mask = self.class_masks[head_id]
            
            for i, cls_pred in enumerate(cls_outputs[head_id]):  # layer index
                b, a, gj, gi = indices[i]
                n = b.shape[0]
                
                if n and self.nc > 1:
                    # Filter targets for this head
                    head_targets_mask = class_mask[tcls[i].long()]
                    
                    if head_targets_mask.sum() > 0:
                        # Get indices of targets this head is responsible for
                        head_indices = head_targets_mask.nonzero(as_tuple=True)[0]
                        
                        # Get predictions for these targets
                        ps = cls_pred[b[head_indices], a[head_indices], 
                                     gj[head_indices], gi[head_indices]]
                        
                        # Class targets
                        t = torch.full_like(ps, self.cn, device=device)
                        t[range(len(head_indices)), tcls[i][head_indices]] = self.cp
                        
                        # Classification loss
                        head_lcls = head_lcls + self.BCEcls(ps, t)
            
            # Apply head weight
            lcls = lcls + head_lcls * self.head_weights[head_id]
        
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        
        # Apply hyperparameter gains
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = reg_obj_outputs[0].shape[0]  # batch size
        
        loss = (lbox + lobj + lcls) * bs
        return loss.squeeze(), torch.cat((lbox, lobj, lcls)).detach()
    
    def build_targets(self, p, targets):
        """
        Build targets for multi-head training
        Same as ComputeLoss.build_targets() - shared for all heads
        """
        # This is identical to the original build_targets
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            ], device=targets.device).float() * g  # offsets
        
        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            
            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                t = t[j]  # filter
                
                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            
            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            
            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3].long() - 1), gi.clamp_(0, gain[2].long() - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
        
        return tcls, tbox, indices, anch

# ============== End of MultiHead Loss ==============