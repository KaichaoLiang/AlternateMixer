#-------------------------------------------------------------------------
# file: fcos_mixer.py
# author: kaichao liang
# date: 2022.06.06
# discription: variant of Adamixer, the query if from Fcos head selection
#--------------------------------------------------------------------------

from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from .sparse_rcnn import SparseRCNN


@DETECTORS.register_module()
class FcosMixer(SparseRCNN):
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)
        batch_w=img.size(-1)
        batch_h=img.size(-2)
        for meta in img_metas:
            meta['batch_shape']=[batch_w,batch_h]
        losses = dict()
        rpn_losses,proposal_boxes, proposal_features, imgs_whwh = \
        self.rpn_head.forward_train(x,img_metas,gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None)
        roi_losses = self.roi_head.forward_train(
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_masks,
            imgs_whwh=imgs_whwh)
        losses.update(rpn_losses)
        losses.update(roi_losses)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        batch_w=img.size(-1)
        batch_h=img.size(-2)
        for meta in img_metas:
            meta['batch_shape']=[batch_w,batch_h]

        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, img_metas)
        results = self.roi_head.simple_test(
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            imgs_whwh=imgs_whwh,
            rescale=rescale)
        return results
        
       

    
