from __future__ import division

import cv2
import numpy as np


class Segmentator(object):
    def __init__(self, img=None, mask=None):
        self.img = img
        self.mask = mask
        self.segmentation = None
        self.gc_mask = None  # mask for the grabcut algorithm

    def threshold(self, img=None, method='otsu', roi=None):
        if img is None:
            img = self.img

        if roi is not None:
            pts = np.nonzero(roi)
            img_crop = img[pts[0].min():pts[0].max(), pts[1].min():pts[1].max()]
        else:
            img_crop = img
        t, mask = cv2.threshold(img_crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        return mask, t

    def grabcut(self, img=None, mask=None, rect=None):
        if img is None:
            img = self.img
        if mask is not None:
            self.gc_mask = mask
        else:
            # self.gc_mask = self.create_gc_mask(img, rect)
            self.gc_mask = cv2.GC_PR_FGD * rect2roi(rect, img.shape[:2])

        # preparing for the grabcut
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # grab cut
        input_mask = self.gc_mask.copy()
        cv2.grabCut(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), input_mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
        segmentation = np.where((input_mask == 1) + (input_mask == 3), 255, 0).astype('uint8')
        self.segmentation = segmentation

        return segmentation

    def create_gc_mask(self, img, rect, pr_bgd_w=0.3, fgd_w=1.2):
        # create mask from rect
        mask_rect = rect2roi(rect, img.shape[:2])

        # this mask is supposed to be PR_FGD
        mask = cv2.GC_PR_FGD * mask_rect

        # threshold inside of the mask
        _, t = self.threshold(img, roi=mask_rect)
        # print t

        # pixels with significantly lower intensity than the threshold define PR_BGD
        pr_bgd = (img < (pr_bgd_w * t)) * mask_rect
        mask = np.where(pr_bgd, cv2.GC_PR_BGD, mask)

        # # pixels with intensity higher than threshold define PR_FGD
        # pr_fgd = (img > (0.3 * t)) * mask_rect
        # mask = np.where(pr_fgd, cv2.GC_PR_FGD, mask)
        # cv2.imshow('mask PR_FGD', 85 * mask.copy())

        # pixels with intensity significantly higher than threshold define FGD
        fgd = (img > (fgd_w * t)) * mask_rect
        mask = np.where(fgd, cv2.GC_FGD, mask)

        # cv2.imshow('masks', np.hstack((255 * pr_bgd, 255 * pr_fgd, 255 * fgd)))
        # cv2.waitKey(0)
        # visualization
        # cv2.imshow('mask', 80 * mask)
        # cv2.waitKey(0)

        return mask

    # def create_gc_mask_


def rect2roi(rect, shape):
    # define top-left and bootom-right points
    rect = np.array(rect)
    tl = (rect[:, 0].min(), rect[:, 1].min())
    br = (rect[:, 0].max(), rect[:, 1].max())

    # create mask
    mask = np.zeros(shape, dtype=np.uint8)
    mask[tl[1]:br[1], tl[0]:br[0]] = 1

    return mask


if __name__ == '__main__':
    fname = '/home/tomas/projects/memskel/memskel/data/cell.png'
    img = cv2.imread(fname, 0)
    rect = ((40, 38), (355, 380))
    mask = rect2roi(rect, img.shape[:2])

    segmentator = Segmentator()
    # segmentator.grabcut(img, mask)
    segmentator.grabcut(img, rect=rect)

    # visualization
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, rect[0], rect[1], (0, 0, 255), 2)
    c = int(np.floor(255 / segmentator.gc_mask.max())) - 1
    vis_stack = np.hstack((vis,
                           cv2.cvtColor(c * segmentator.gc_mask, cv2.COLOR_GRAY2BGR),
                           cv2.cvtColor(segmentator.segmentation, cv2.COLOR_GRAY2BGR)))
    cv2.imshow('gc', vis_stack)
    cv2.waitKey(0)
