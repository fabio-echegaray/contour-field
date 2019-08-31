import logging

import numpy as np
import pandas as pd
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtGui import QMainWindow, QWidget
from matplotlib.figure import SubplotParams
import matplotlib.ticker as ticker

from gui._ring_label import RingImageQLabel
from gui.gui_mplwidget import MplWidget

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('ring.gui')
sp = SubplotParams(left=0., bottom=0., right=1., top=1.)
mydpi = 72

pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)


class GraphWidget(QWidget):
    graphWidget: MplWidget

    def __init__(self):
        super(GraphWidget, self).__init__()
        uic.loadUi('./gui_ring_graph.ui', self)
        self.graphWidget.clear()
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        self.ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        self.ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
        self.ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))

    @property
    def canvas(self):
        return self.graphWidget.canvas

    @property
    def ax(self):
        return self.canvas.ax


class RingWindow(QMainWindow):
    image: RingImageQLabel
    statusbar: QtGui.QStatusBar

    def __init__(self):
        super(RingWindow, self).__init__()
        uic.loadUi('./gui_ring.ui', self)
        self.move(50, 100)
        self.zSpin.valueChanged.connect(self.on_zvalue_change)
        self.openButton.pressed.connect(self.on_open_button)
        self.addButton.pressed.connect(self.on_add_button)
        self.dnaSpin.valueChanged.connect(self.on_dnaval_change)
        self.actSpin.valueChanged.connect(self.on_actval_change)
        self.dnaChk.toggled.connect(self.on_img_toggle)
        self.actChk.toggled.connect(self.on_img_toggle)
        self.image.clicked.connect(self.on_img_click)

        self.image.dna_channel = self.dnaSpin.value()
        self.image.act_channel = self.actSpin.value()

        self.grph = GraphWidget()
        ph = self.geometry().height()
        px = self.geometry().x()
        py = self.geometry().y()
        dw = self.grph.width()
        dh = self.grph.height()
        self.grph.setGeometry(px, py + ph + 50, dw, dh)
        self.grph.show()

        self.measure_n = 0

        self.df = pd.DataFrame()

        self.file = "/Users/Fabio/data/lab/airyscan/nil.czi"

    def closeEvent(self, event):
        if not self.df.empty:
            self.df.loc[:, "condition"] = self.experimentLineEdit.text()
            self.df.to_csv("out.csv")
        self.grph.close()

    def _graph(self):
        if self.image.measurements is not None:
            self.grph.ax.cla()
            for me in self.image.measurements:
                x = np.arange(start=0, stop=len(me['l']), step=1)
                self.grph.ax.plot(x, me['l'], linewidth=0.5, linestyle='-', color=me['c'])
            self.statusbar.showMessage("ptp: %s" % ["%d " % me['d'] for me in self.image.measurements])
            self.grph.canvas.draw()

    @QtCore.pyqtSlot()
    def on_img_toggle(self):
        logger.info('on_img_toggle')
        if self.dnaChk.isChecked():
            self.image.active_ch = "dna"
        if self.actChk.isChecked():
            self.image.active_ch = "act"

    @QtCore.pyqtSlot()
    def on_open_button(self):
        logger.info('on_open_button')
        qfd = QtGui.QFileDialog()
        path = os.path.dirname(self.file)
        flt = "zeiss(*.czi)"
        f = QtGui.QFileDialog.getOpenFileName(qfd, "Open File", path, flt)
        if len(f) > 0:
            self.image.file = f
            self.image.zstack = self.zSpin.value()
            self.image.dna_channel = self.dnaSpin.value()
            self.nchLbl.setText("%d channels" % self.image.n_channels)
            self.nzsLbl.setText("%d z-stacks" % self.image.n_zstack)
            self.nfrLbl.setText("%d %s" % (self.image.n_frames, "frames" if self.image.n_frames > 1 else "frame"))

    @QtCore.pyqtSlot()
    def on_img_click(self):
        logger.info('on_img_click')
        self._graph()

    @QtCore.pyqtSlot()
    def on_zvalue_change(self):
        logger.info('on_zvalue_change')
        self.image.zstack = self.zSpin.value() % self.image.n_zstack
        self.zSpin.setValue(self.image.zstack)
        self._graph()

    @QtCore.pyqtSlot()
    def on_dnaval_change(self):
        logger.info('on_dnaval_change')
        val = self.dnaSpin.value() % self.image.n_channels
        self.dnaSpin.setValue(val)
        self.image.dna_channel = val
        if self.dnaChk.isChecked():
            self.image.active_ch = "dna"

    @QtCore.pyqtSlot()
    def on_actval_change(self):
        logger.info('on_actval_change')
        val = self.actSpin.value() % self.image.n_channels
        self.actSpin.setValue(val)
        self.image.act_channel = val
        if self.actChk.isChecked():
            self.image.active_ch = "act"

    @QtCore.pyqtSlot()
    def on_add_button(self):
        logger.info('on_add_button')
        if self.image.measurements is not None:
            new = pd.DataFrame(self.image.measurements)
            new.loc[:, "m"] = self.measure_n
            self.df = self.df.append(new, ignore_index=True, sort=False)
            self.measure_n += 1
            print(self.df)


if __name__ == '__main__':
    import sys
    import os

    from PyQt4.QtCore import QT_VERSION_STR
    from PyQt4.Qt import PYQT_VERSION_STR

    base_path = os.path.abspath('%s' % os.getcwd())
    logging.info('Qt version:' + QT_VERSION_STR)
    logging.info('PyQt version:' + PYQT_VERSION_STR)
    logging.info('Working dir:' + os.getcwd())
    logging.info('Base dir:' + base_path)
    os.chdir(base_path)

    app = QtGui.QApplication(sys.argv)

    gui = RingWindow()
    gui.show()

    sys.exit(app.exec_())
