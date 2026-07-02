from mmrotate.models.detectors.oriented_rcnn import OrientedRCNN
from mmrotate.models import ROTATED_DETECTORS
from mmrotate.models.modules.fma import FMA 

@ROTATED_DETECTORS.register_module()
class OrientedRCNNWithFMA(OrientedRCNN):
    def __init__(self, fma_cfg, *args, **kwargs):
        super(OrientedRCNNWithFMA, self).__init__(*args, **kwargs)
        self.fma = FMA(**fma_cfg)

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs):
        x = self.backbone(img)
        enhanced_x = []
        for i, feat in enumerate(x):
            if i == len(x) - 1:
                down_scale = 32.0 
                enhanced_feat = self.fma(feat, gt_bboxes, down_scale)
                enhanced_x.append(enhanced_feat)
            else:
                enhanced_x.append(feat)

        x = self.neck(enhanced_x)

        losses = dict()
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x, img_metas, gt_bboxes, gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore, proposal_cfg=proposal_cfg,** kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = kwargs['proposals']

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                gt_bboxes, gt_labels,
                                                gt_bboxes_ignore, **kwargs)
        losses.update(roi_losses)

        return losses