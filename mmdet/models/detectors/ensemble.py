import torch
import torch.nn as nn

# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)

from .base import BaseDetector
from mmdet.core import merge_aug_proposals
from mmdet.models.roi_heads.htc_roi_head import HybridTaskCascadeRoIHead
from mmdet.models.roi_heads.scnet_roi_head import SCNetRoIHead

class EnsembleModel(BaseDetector):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def simple_test(self, img, img_meta, **kwargs):
        pass

    def forward_train(self, imgs, img_metas, **kwargs):
        pass

    def extract_feat(self, imgs):
        pass

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test with augmentations.
        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        
        samples_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(samples_per_gpu)]
      
                
        for model in self.models:
            for x, img_meta in zip(model.extract_feats(imgs), img_metas):
                proposal_list = model.rpn_head.simple_test_rpn(x, img_meta)
                for i, proposals in enumerate(proposal_list):
                    aug_proposals[i].append(proposals)                   

          
        aug_img_metas = []
        for i in range(samples_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)
            
        proposal_list = [
            merge_aug_proposals(proposals, aug_img_meta, self.models[0].rpn_head.test_cfg)
            for proposals, aug_img_meta in zip(aug_proposals, aug_img_metas)
        ]
        
        aug_bboxes = []
        aug_scores = []
        aug_img_metas = []
        rcnn_test_cfg = self.models[0].roi_head.test_cfg
        
        
        for model in self.models:
            img_feats = model.extract_feats(imgs)
            semantic_feats = [model.roi_head.semantic_head(feat)[1] for feat in img_feats]
   

            if isinstance(model.roi_head,HybridTaskCascadeRoIHead):
                glbctx_feats = [None] * len(img_metas)
            else:
                glbctx_feats = [model.roi_head.glbctx_head(feat)[1] for feat in img_feats]
                
            for x, img_meta, semantic_feat, glbctx_feat in zip(
                    img_feats, img_metas, semantic_feats, glbctx_feats):
                # only one image in the batch
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                flip_direction = img_meta[0]['flip_direction']
                proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,scale_factor, flip, flip_direction)
                # "ms" in variable names means multi-stage
                ms_scores = []

                rois = bbox2roi([proposals])
                for i in range(model.roi_head.num_stages):
                    bbox_head = model.roi_head.bbox_head[i]
                    if glbctx_feat is None:
                        bbox_results = model.roi_head._bbox_forward(i,x,rois,semantic_feat=semantic_feat)
                    else:
                        bbox_results = model.roi_head._bbox_forward(i,x,rois,semantic_feat=semantic_feat,glbctx_feat=glbctx_feat)
                    ms_scores.append(bbox_results['cls_score'])
                    if i < model.roi_head.num_stages - 1:
                        bbox_label = bbox_results['cls_score'].argmax(dim=1)
                        rois = bbox_head.regress_by_class(
                            rois, bbox_label, bbox_results['bbox_pred'],
                            img_meta[0])

                cls_score = sum(ms_scores) / float(len(ms_scores))
                bboxes, scores = model.roi_head.bbox_head[-1].get_bboxes(
                    rois,
                    cls_score,
                    bbox_results['bbox_pred'],
                    img_shape,
                    scale_factor,
                    rescale=False,
                    cfg=None)
   

                aug_bboxes.append(bboxes)
                aug_scores.append(scores)
                aug_img_metas.append(img_meta)
                
        merged_bboxes, merged_scores = merge_aug_bboxes(aug_bboxes, aug_scores, aug_img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img,
                                               )

        det_bbox_results = bbox2result(det_bboxes, det_labels,self.models[0].roi_head.bbox_head[-1].num_classes)
                
                
        aug_masks = []
        aug_img_metas = []
        for model in self.models:
            img_feats = model.extract_feats(imgs)
            semantic_feats = [model.roi_head.semantic_head(feat)[1] for feat in img_feats]   
            
            if isinstance(model.roi_head, SCNetRoIHead):
                glbctx_feats = [model.roi_head.glbctx_head(feat)[1] for feat in img_feats]
            else:
                glbctx_feats = [None] * len(img_metas)
            for x, img_meta, semantic_feat, glbctx_feat in zip(img_feats, img_metas, semantic_feats, glbctx_feats):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                flip_direction = img_meta[0]['flip_direction']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,scale_factor, flip,flip_direction)
                mask_rois = bbox2roi([_bboxes])
                # diff htc scnet
                if isinstance(model.roi_head, SCNetRoIHead):
                    bbox_results = model.roi_head._bbox_forward(-1,x,mask_rois,semantic_feat=semantic_feat,glbctx_feat=glbctx_feat)
                    relayed_feat = bbox_results['relayed_feat']
                    relayed_feat = model.roi_head.feat_relay_head(relayed_feat)
                    mask_results = model.roi_head._mask_forward(x,mask_rois,semantic_feat=semantic_feat,glbctx_feat=glbctx_feat,relayed_feat=relayed_feat)
                    mask_pred = mask_results['mask_pred']
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                    aug_img_metas.append(img_meta)       
                else:
                    mask_feats = model.roi_head.mask_roi_extractor[-1](x[:len(model.roi_head.mask_roi_extractor[-1].featmap_strides)],mask_rois)
                    mask_semantic_feat = model.roi_head.semantic_roi_extractor([semantic_feat], mask_rois)
                    if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                        mask_semantic_feat = F.adaptive_avg_pool2d(mask_semantic_feat, mask_feats.shape[-2:])
                    mask_feats += mask_semantic_feat
                    last_feat = None
                    for i in range(model.roi_head.num_stages):
                        mask_head = model.roi_head.mask_head[i]
                        if model.roi_head.mask_info_flow:
                            mask_pred, last_feat = mask_head(mask_feats, last_feat)
                        else:
                            mask_pred = mask_head(mask_feats)
                        aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                        
        merged_masks = merge_aug_masks(aug_masks, aug_img_metas,rcnn_test_cfg)
        ori_shape = img_metas[0][0]['ori_shape']
        if isinstance(self.models[0].roi_head, SCNetRoIHead):
            det_segm_results = self.models[0].roi_head.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                rcnn_test_cfg,
                ori_shape,
                scale_factor=1.0,
                rescale=False)
        elif isinstance(self.models[0].roi_head, HybridTaskCascadeRoIHead):    
                det_segm_results = self.models[0].roi_head.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
                
        return [(det_bbox_results, det_segm_results)]
  
        
