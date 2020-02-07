import os
import sys
import logging
import itertools
from typing import List

import numpy as np
import seaborn as sns
from PyQt4 import Qt, QtCore, QtGui
from PyQt4.QtCore import QRect, QTimer
from PyQt4.QtGui import QLabel, QWidget
from PyQt4.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from shapely.geometry.point import Point

from gui._widget_graph import GraphWidget
from gui._image_loading import retrieve_image
import measurements as m

logger = logging.getLogger('ring.stk.gui')


class StkRingWidget(QWidget):
    images: List[QLabel]
    linePicked = Qt.pyqtSignal()

    def __init__(self, parent=None, stacks=5, n_channels=None, dna_ch=None, rng_ch=None, pix_per_um=None,
                 line_length=4, dl=0.05, lines_to_measure=1, **kwargs):
        super().__init__(parent, **kwargs)
        path = os.path.join(sys.path[0], __package__)

        # layout for images
        self.vLayout = QtGui.QHBoxLayout()
        self.setLayout(self.vLayout)

        self.images = list()
        for i in range(stacks):
            img = QtGui.QLabel()
            img.width = 100
            img.height = 100
            self.images.append(img)
            self.vLayout.addWidget(img)

        self.dnaChannel = dna_ch
        self.rngChannel = rng_ch
        self.nChannels = n_channels
        self.zstacks = stacks
        self.nlines = lines_to_measure
        self.dl = dl
        self.line_length = line_length
        self.pix_per_um = pix_per_um

        self.measurements = None
        self.selectedN = None
        self.selectedZ = None
        self._render = True

        self._images = None
        self._pixmaps = None
        self._nucboundaries = None
        self._colors = sns.husl_palette(lines_to_measure, h=.5).as_hex()

        # graph widget
        self.grph = GraphWidget()
        self.grph.setWindowTitle('Selected line across stack')
        self.grphtimer = QTimer()
        self.grphtimer.setSingleShot(True)

        self.grph.linePicked.connect(self.onLinePickedFromGraph)
        self.grph.linePicked.connect(self.linePicked)
        self.grphtimer.timeout.connect(self._graph)

        self.setWindowTitle('Stack images')
        self.grph.show()
        self.show()
        self.moveEvent(None)

    def moveEvent(self, QMoveEvent):
        logger.debug("moveEvent")
        px = self.geometry().x()
        py = self.geometry().y()
        pw = self.geometry().width()
        ph = self.geometry().height()

        dw = self.grph.width()
        dh = self.grph.height()
        self.grph.setGeometry(px, py + ph + 20, pw, dh)

    def closeEvent(self, event):
        self.grph.close()

    def focusInEvent(self, QFocusEvent):
        logger.debug('focusInEvent')
        self.activateWindow()
        self.grph.activateWindow()

    def showEvent(self, event):
        self.setFocus()

    def mouseReleaseEvent(self, ev):
        for k, im in enumerate(self.images):
            if im.underMouse():
                logger.info(f"image {k} clicked.")
                self.selectedZ = k
                self.linePicked.emit()

        self._repainImages()
        self._graph()
        self.drawMeasurements()

    def _repainImages(self):
        for i in range(len(self._pixmaps)):
            self.images[i].setPixmap(self._pixmaps[i])
            # self.images[i].setPixmap(
            #     self._pixmaps[i].scaled(self.images[i].width, self.images[i].height, QtCore.Qt.KeepAspectRatio))

    def loadImages(self, images, xy=(0, 0), wh=(1, 1)):
        assert (self.dnaChannel is not None and self.rngChannel is not None and
                self.nChannels is not None), "some parameters were not correctly set."
        # assert (len(boundaries) == self.zstacks and len(
        #     lines) == self.zstacks), "boundaries and lines should be the same length as zstacks in the image."

        # crop the image
        x1, x2 = int(xy[0] - wh[0] / 2), int(xy[0] + wh[0] / 2)
        y1, y2 = int(xy[1] - wh[1] / 2), int(xy[1] + wh[1] / 2)
        self._images = images[:, y1:y2, x1:x2]
        if self._images.size == 0:
            logger.warning(f"empty cropped image! x1,x2,y1,y2 ({x1},{x2},{y1},{y2})")

        self._pixmaps = list()

        for i in range(self.zstacks):
            data = retrieve_image(images, channel=self.rngChannel, number_of_channels=self.nChannels,
                                  zstack=i, number_of_zstacks=self.zstacks, frame=0)
            self.dwidth, self.dheight = data.shape

            # map the data range to 0 - 255
            img_8bit = ((data - data.min()) / (data.ptp() / 255.0)).astype(np.uint8)
            qtimage = QtGui.QImage(img_8bit.repeat(4), self.dwidth, self.dheight, QtGui.QImage.Format_RGB32)
            imagePixmap = QPixmap(qtimage)
            rect = QRect(x1, y1, wh[0], wh[1])
            cropped = imagePixmap.copy(rect)
            self._pixmaps.append(cropped)

        self._repainImages()
        self.update()
        return

    def measure(self):
        if self._images.size == 0:
            logger.warning("can't measure on an empty image!")
            return
        logger.debug("computing nuclei boundaries")

        self._nucboundaries = list()
        self.measurements = list()
        for i in range(self.zstacks):
            dnaimg = retrieve_image(self._images, channel=self.dnaChannel, number_of_channels=self.nChannels,
                                    zstack=i, number_of_zstacks=self.zstacks, frame=0)
            width, height = dnaimg.shape
            x, y = int(width / 2), int(height / 2)
            center_pt = Point(x, y)
            lbl, boundaries = m.nuclei_segmentation(dnaimg, simp_px=self.pix_per_um / 2)

            if boundaries is not None:
                nucbnd = [b["boundary"] for b in boundaries if center_pt.within(b["boundary"])][0]
                ring = retrieve_image(self._images, channel=self.rngChannel, number_of_channels=self.nChannels,
                                      zstack=i, number_of_zstacks=self.zstacks, frame=0)

                # measurements around polygon are done from the centroid used to crop all the z-stack
                lines = m.measure_lines_around_polygon(ring, nucbnd, from_pt=center_pt, radius=int(width / 2),
                                                       rng_thick=self.line_length, dl=self.dl, n_lines=self.nlines,
                                                       pix_per_um=self.pix_per_um)
                for k, ((ls, l), colr) in enumerate(zip(lines, itertools.cycle(self._colors))):
                    if ls is not None:
                        self.measurements.append({'n': k, 'x': x, 'y': y, 'z': i, 'l': l, 'c': colr,
                                                  'ls0': ls.coords[0], 'ls1': ls.coords[1],
                                                  'd': max(l) - min(l), 'sum': np.sum(l)})
                logger.debug(f"{len(self.measurements)} lines obtained.")
            else:
                nucbnd = None

            self._nucboundaries.append(nucbnd)

    # @profile
    def drawMeasurements(self, erase_bkg=False):
        if not self.render or not self._nucboundaries: return
        if erase_bkg:
            self._repainImages()
        angle_delta = 2 * np.pi / self.nlines
        nim, width, height = self._images.shape

        for i in range(self.zstacks):
            if i > len(self._nucboundaries) - 1: continue

            painter = QPainter()
            painter.begin(self.images[i].pixmap())
            painter.setRenderHint(QPainter.Antialiasing)

            n = self._nucboundaries[i]
            if not n: continue
            nuc_pen = QPen(QBrush(QColor('white')), 1.1)
            nuc_pen.setStyle(QtCore.Qt.DotLine)
            painter.setPen(nuc_pen)
            dl2 = self.line_length * self.pix_per_um / 2

            try:
                # get nuclei external and internal boundaries as a polygons
                for d in [dl2, -dl2]:
                    nucb_qpoints = [Qt.QPoint(x, y) for x, y in n.buffer(d).exterior.coords]
                    painter.drawPolygon(Qt.QPolygon(nucb_qpoints))
            except Exception as e:
                logger.error(e)

            if self.selectedN is not None:
                alpha = angle_delta * self.selectedN
                x, y = int(width / 2), int(height / 2)
                a = int(width / 2)
                pt1 = Qt.QPoint(x, y)
                pt2 = Qt.QPoint(a * np.cos(alpha) + x, a * np.sin(alpha) + y)
                painter.drawLine(pt1, pt2)

                for me in self.measurements:
                    if me['n'] == self.selectedN:
                        if me['z'] == self.selectedZ and i == self.selectedZ:
                            painter.setPen(QPen(QBrush(QColor(me['c'])), 1 * self.pix_per_um))
                        else:
                            painter.setPen(QPen(QBrush(QColor('gray')), 0.1 * self.pix_per_um))

                        pts = [Qt.QPoint(x, y) for x, y in [me['ls0'], me['ls1']]]
                        painter.drawLine(pts[0], pts[1])

            painter.end()
        self.grphtimer.start(1000)
        self.update()

    def _graph(self, alpha=1.0):
        self.grph.clear()
        if self.measurements is not None:
            for me in self.measurements:
                if me['n'] == self.selectedN:
                    x = np.arange(start=0, stop=len(me['l']) * self.dl, step=self.dl)
                    lw = 0.1 if self.selectedZ is not None and me['z'] != self.selectedZ else 0.5
                    self.grph.ax.plot(x, me['l'], linewidth=lw, linestyle='-', color=me['c'], alpha=alpha, zorder=10,
                                      picker=5, label=me['z'])
            self.grph.format_ax()
            self.grph.canvas.draw()

    @property
    def selectedLine(self):
        if self.selectedN is not None and self.selectedZ is not None:
            for me in self.measurements:
                if me['n'] == self.selectedN and me['z'] == self.selectedZ:
                    return me
        return None

    @QtCore.pyqtSlot()
    def onLinePickedFromGraph(self):
        logger.debug('onLinePickedFromGraph')
        self.selectedZ = self.grph.selectedLine if self.grph.selectedLine is not None else None
        if self.selectedZ is not None:
            logger.debug(f"Z {self.selectedZ} selected")
            self._repainImages()
            self.drawMeasurements()

            # self.emit(QtCore.SIGNAL('linePicked()'))
            # self.linePicked.emit()

    @property
    def render(self):
        return self._render

    @render.setter
    def render(self, value):
        if value is not None:
            self._render = value
            self._repainImages()
            self.drawMeasurements()
