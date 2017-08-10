from __future__ import division

import numpy as np
import cv2


class Viewer(object):
    def __init__(self, img):
        self.winname = 'segmentator'
        self.win = cv2.namedWindow(self.winname)
        cv2.setMouseCallback(self.winname, self.mouse_marking_rect)
        self.img = img
        self.vis = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

        self.marking_F = False  # flag indicating the marking state
        self.marking_type = 'rect'  # types = ('rect', 'line')
        self.FGD_COL = (0, 255, 0)
        self.BGD_COL = (0, 0, 255)
        self.rect = [None, None]
        self.lines = []
        self.line_pts = []
        self.seed_type = 'fgd'  # types = ('fgd', 'bgd')
        self.seed_col = self.FGD_COL

    def create_label(self):
        cv2.putText(self.vis, self.marking_type, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        cv2.putText(self.vis, self.seed_type, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    def update_vis(self):
        self.vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.create_label()

        # draw rect
        if self.rect[0] is not None:
            cv2.rectangle(self.vis, self.rect[0], self.rect[1], (0, 0, 255), 1)

        # draw lines
        for l in self.lines:
            for i in range(len(l) - 1):
                cv2.line(self.vis, l[i], l[i + 1], self.seed_col, 2)
        # if self.line_pts:
        for i in range(len(self.line_pts) - 1):
            cv2.line(self.vis, self.line_pts[i], self.line_pts[i + 1], self.seed_col, 2)

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
            self.lines.append(self.line_pts[:])
            self.line_pts = []
            self.marking_F = False

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
            elif key == 10:  # return ... start segmentation
                pass
            elif key == 27:  # esc ... terminate program
                print 'Program terminated by user.'
                break


if __name__ == '__main__':
    # fname = '/home/tomas/projects/memskel/memskel/data/GAP43-CFP-Gi2_c1.png'
    fname = '/home/tomas/projects/memskel/memskel/data/cell.png'
    img = cv2.imread(fname, 0)

    viewer = Viewer(img)
    viewer.run()