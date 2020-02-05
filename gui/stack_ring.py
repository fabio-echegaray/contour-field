import os
import sys
import logging
import itertools
from typing import List

import numpy as np
import seaborn as sns
from PyQt4 import Qt, QtCore, QtGui
from PyQt4.QtCore import QRect
from PyQt4.QtGui import QLabel, QWidget
from PyQt4.QtGui import QBrush, QColor, QPainter, QPen, QPixmap

from gui._widget_graph import GraphWidget
from gui._image_loading import retrieve_image
import measurements as m

logger = logging.getLogger('ring.stk.gui')


class StkRingWidget(QWidget):
    images: List[QLabel]
    linePicked = Qt.pyqtSignal()

    def __init__(self, parent=None, stacks=5, n_channels=None, dna_ch=None, rng_ch=None, pix_per_um=None,
                 line_length=None, lines_to_measure=1, **kwargs):
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
        self.dl = line_length
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

        self.grph.linePicked.connect(self.onLinePickedFromGraph)
        self.grph.linePicked.connect(self.linePicked)

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
        # self.setGeometry(px, py, pw, ph)
        # logger.debug("stack win size " + str(self.geometry()))
        # logger.debug("graph win size prev " + str(self.grph.geometry()))
        self.grph.setGeometry(px, py + ph + 20, pw, dh)
        # logger.debug("graph win size after " + str(self.grph.geometry()))

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
        self._nucboundaries = list()

        logger.debug("computing nuclei boundaries")
        self.measurements = list()
        for i in range(self.zstacks):
            dnaimg = retrieve_image(self._images, channel=self.dnaChannel, number_of_channels=self.nChannels,
                                    zstack=i, number_of_zstacks=self.zstacks, frame=0)
            width, height = dnaimg.shape
            x, y = int(width / 2), int(height / 2)
            lbl, boundaries = m.nuclei_segmentation(dnaimg, simp_px=self.pix_per_um / 4)

            if boundaries is not None:
                nucbnd = (boundaries[0]["boundary"]
                          .buffer(0.1 * self.pix_per_um, join_style=1)
                          .buffer(-0.1 * self.pix_per_um, join_style=1)
                          )
                ring = retrieve_image(self._images, channel=self.rngChannel, number_of_channels=self.nChannels,
                                      zstack=i, number_of_zstacks=self.zstacks, frame=0)

                lines = m.measure_lines_around_polygon(ring, nucbnd, rng_thick=4, dl=self.dl,
                                                       n_lines=self.nlines, pix_per_um=self.pix_per_um)
                for k, ((ls, l), colr) in enumerate(zip(lines, itertools.cycle(self._colors))):
                    self.measurements.append({'n': k, 'x': x, 'y': y, 'z': i, 'l': l, 'c': colr,
                                              'ls0': ls.coords[0], 'ls1': ls.coords[1],
                                              'd': max(l) - min(l), 'sum': np.sum(l)})
            else:
                nucbnd = None

            self._nucboundaries.append(nucbnd)

    def drawMeasurements(self, erase_bkg=False):
        if not self.render: return
        if erase_bkg:
            self._repainImages()
        for i in range(self.zstacks):
            painter = QPainter()
            painter.begin(self.images[i].pixmap())
            painter.setRenderHint(QPainter.Antialiasing)

            nuc_pen = QPen(QBrush(QColor('red')), 1.1)
            nuc_pen.setStyle(QtCore.Qt.DotLine)
            painter.setPen(nuc_pen)

            n = self._nucboundaries[i]
            # get nuclei boundary as a polygon
            nucb_qpoints = [Qt.QPoint(x, y) for x, y in n.exterior.coords]
            painter.drawPolygon(Qt.QPolygon(nucb_qpoints))

            if self.selectedN is not None:
                for me in self.measurements:
                    if me['n'] == self.selectedN:
                        if me['z'] == self.selectedZ and i == self.selectedZ:
                            painter.setPen(QPen(QBrush(QColor(me['c'])), 1 * self.pix_per_um))
                        else:
                            painter.setPen(QPen(QBrush(QColor('gray')), 0.1 * self.pix_per_um))

                        pts = [Qt.QPoint(x, y) for x, y in [me['ls0'], me['ls1']]]
                        painter.drawLine(pts[0], pts[1])

            painter.end()
        self._graph()
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
