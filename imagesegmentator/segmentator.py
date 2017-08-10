from __future__ import division

import cv2
import numpy as np


class Segmentator(object):
    def __init__(self, img=None, mask=None):
        self.img = img
        self.mask = mask
        self.segmentation = None

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
            if mask is None:
                if self.mask is None:
                    mask = np.ones(self.img.shape[:2], dtype=np.uint8)
                else:
                    mask = self.mask

        # preparing for the grabcut
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        mask = self.create_gc_mask(rect, img.shape[:2])
        # cv2.grabCut(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), mask, (0, 0, 1, 1), bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
        cv2.grabCut(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
        segmentation = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        self.segmentation = segmentation
        return segmentation

    def create_gc_mask(self, rect, shape):
        mask_rect = rect2roi(rect, shape)
        mask = mask_rect * cv2.GC_PR_FGD

        _, t = self.threshold(img, roi=mask_rect)
        prob_bgd = (img < (0.3 * t)) * mask_rect
        fgd = (img > (1.2 * t)) * mask_rect
        mask = np.where(prob_bgd, cv2.GC_PR_BGD, mask)
        mask = np.where(fgd, cv2.GC_FGD, mask)

        cv2.imshow('mask', 80 * mask)
        # cv2.waitKey(0)

        return mask


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
    cv2.imshow('gc', np.hstack((vis, cv2.cvtColor(segmentator.segmentation, cv2.COLOR_GRAY2BGR))))
    cv2.waitKey(0)
