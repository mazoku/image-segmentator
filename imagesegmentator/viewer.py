from __future__ import division

import numpy as np
import cv2
from segmentator import *


class Viewer(object):
    def __init__(self, img):
        self.winname = 'segmentator'
        self.win = cv2.namedWindow(self.winname)
        cv2.setMouseCallback(self.winname, self.mouse_marking_rect)
        self.img = img
        self.vis = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        self.mask = None

        self.marking_F = False  # flag indicating the marking state
        self.marking_type = 'rect'  # types = ('rect', 'line')
        self.FGD_COL = (0, 255, 0)
        self.BGD_COL = (0, 0, 255)
        self.PR_FGD_COL = (150, 255, 150)
        self.PR_BGD_COL = (150, 150, 255)
        self.rect = [None, None]
        self.lines = []
        self.line_pts = []
        self.line_width = 2
        self.seed_types = ('bgd', 'fgd', 'pr_bgd', 'pr_fgd')
        self.seed_labels = (cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD)
        self.seed_colors = (self.BGD_COL, self.FGD_COL, self.PR_BGD_COL, self.PR_FGD_COL)
        self._seed_idx = 0
        self.seed_type = self.seed_types[self.seed_idx]
        self.seed_col = self.seed_colors[self.seed_idx]

        self.segmentator = Segmentator()

    @property
    def seed_idx(self):
        return self._seed_idx

    @seed_idx.setter
    def seed_idx(self, value):
        self._seed_idx = value
        self.seed_type = self.seed_types[value]
        self.seed_col = self.seed_colors[value]

    def create_label(self, scale=0.4):
        cv2.putText(self.vis, 'seed: {}'.format(self.seed_type), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 0, 255), 1)
        cv2.putText(self.vis, 'lwidth: {}'.format(self.line_width), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 0, 255), 1)
        cv2.putText(self.vis, 'mark: {}'.format(self.marking_type), (5, 45), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 0, 255), 1)

    def update_vis(self):
        self.vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.create_label()

        # draw rect
        if self.rect[0] is not None:
            cv2.rectangle(self.vis, self.rect[0], self.rect[1], (0, 0, 255), 1)

        # draw lines
        for seed_idx, line_width, line in self.lines:
            for i in range(len(line) - 1):
                cv2.line(self.vis, line[i], line[i + 1], self.seed_colors[seed_idx], line_width)
        # if self.line_pts:
        for i in range(len(self.line_pts) - 1):
            cv2.line(self.vis, self.line_pts[i], self.line_pts[i + 1], self.seed_col, self.line_width)

        cv2.imshow(self.winname, self.vis)

    def mouse_marking_rect(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.rect[0] is not None:
                self.rect = [None, None]
            self.rect[0] = (x, y)
            self.marking_F = True
        if event == cv2.EVENT_MOUSEMOVE and self.marking_F:
            self.rect[1] = (x, y)
            self.update_vis()
        if event == cv2.EVENT_LBUTTONUP:
            self.rect[1] = (x, y)
            self.marking_F = False
            print 'marked rect: {}'.format(self.rect)

    def mouse_marking_line(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.line_pts.append((x, y))
            self.marking_F = True
        if event == cv2.EVENT_MOUSEMOVE and self.marking_F:
            self.line_pts.append((x, y))
            self.update_vis()
        if event == cv2.EVENT_LBUTTONUP:
            self.lines.append((self.seed_idx, self.line_width, self.line_pts[:]))
            self.line_pts = []
            self.marking_F = False
        if event == cv2.EVENT_MOUSEWHEEL:
            print 'wheel: ',
            if flags < 0:
                self.seed_idx = min(self.seed_idx + 1, len(self.seed_types))
                print '<'
            else:
                self.seed_idx = max(self.seed_idx - 1, 0)
                print '>'
            self.update_vis()

    def create_mask(self):
        if self.mask is None:
            if self.rect[0] is not None:
                self.mask = cv2.GC_PR_FGD * rect2roi(self.rect, self.img.shape[:2])
            else:
                self.mask = cv2.GC_PR_FGD * np.ones(self.img.shape[:2], dtype=np.uint8)
        for seed_idx, line_width, line in self.lines:
            for i in range(len(line) - 1):
                cv2.line(self.mask, line[i], line[i + 1], self.seed_labels[seed_idx], line_width)
        cv2.imshow('mask', 84 * self.mask)
        cv2.waitKey(0)
        cv2.destroyWindow('mask')

    def run(self):
        while True:
            # cv2.imshow(self.winname, self.vis)
            self.update_vis()
            self.create_label()
            key = cv2.waitKey(10) & 0xFF
            if key != 255:
                print key

            if key == ord('l'):  # l = 108 ... switch to line marking
                self.marking_type = 'line'
                cv2.setMouseCallback(self.winname, self.mouse_marking_line)
            elif key == ord('r'):  # r =114 ... switch to rect marking
                self.marking_type = 'rect'
                cv2.setMouseCallback(self.winname, self.mouse_marking_rect)
            elif key == 82:  # up arrow
                self.seed_idx = min(self.seed_idx + 1, len(self.seed_types) - 1)
                self.update_vis()
            elif key == 84:  # down arrow
                self.seed_idx = max(self.seed_idx - 1, 0)
                self.update_vis()
            elif key in (45, 173):  # -
                self.line_width = max(0, self.line_width - 1)
            elif key in (43, 171):  # +
                self.line_width = min(self.line_width + 1, 10)
            elif key == 10:  # return ... start segmentation
                self.create_mask()
                self.segmentator.grabcut(self.img, mask=self.mask)
                cv2.imshow('segmentation', self.segmentator.segmentation)
                cv2.waitKey(0)
                cv2.destroyWindow('segmentation')
            elif key == 27:  # esc ... terminate program
                print 'Program terminated by user.'
                break


if __name__ == '__main__':
    fname = '/home/tomas/projects/memskel/memskel/data/GAP43-CFP-Gi2_c1.png'
    # fname = '/home/tomas/projects/memskel/memskel/data/cell.png'
    img = cv2.imread(fname, 0)

    viewer = Viewer(img)
    viewer.run()