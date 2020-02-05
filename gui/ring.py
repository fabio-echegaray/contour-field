import os
import sys
import logging

import numpy as np
import pandas as pd
import seaborn as sns
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtGui import QMainWindow, QWidget
import matplotlib.ticker as ticker
from matplotlib.ticker import EngFormatter

from gui._ring_label import RingImageQLabel, _nlin
from gui._widget_graph import GraphWidget
from gui.stack_ring import StkRingWidget
import measurements as m

logger = logging.getLogger('ring.gui')

pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)


class RingWindow(QMainWindow):
    image: RingImageQLabel
    statusbar: QtGui.QStatusBar

    def __init__(self):
        super(RingWindow, self).__init__()
        path = os.path.join(sys.path[0], __package__)

        uic.loadUi(os.path.join(path, 'gui_ring.ui'), self)
        self.move(50, 0)

        self.ctrl = QWidget()
        uic.loadUi(os.path.join(path, 'gui_ring_controls.ui'), self.ctrl)
        self.ctrl.show()

        self.ctrl.zSpin.valueChanged.connect(self.onZValueChange)
        self.ctrl.openButton.pressed.connect(self.onOpenButton)
        self.ctrl.addButton.pressed.connect(self.onAddButton)
        self.ctrl.plotButton.pressed.connect(self.onPlotButton)
        self.ctrl.measureButton.pressed.connect(self.onMeasureButton)
        self.ctrl.dnaSpin.valueChanged.connect(self.onDnaValChange)
        self.ctrl.actSpin.valueChanged.connect(self.onActValChange)
        self.ctrl.dnaChk.toggled.connect(self.onImgToggle)
        self.ctrl.actChk.toggled.connect(self.onImgToggle)
        self.ctrl.renderChk.stateChanged.connect(self.onRenderChk)

        self.image.clicked.connect(self.onImgUpdate)
        self.image.lineUpdated.connect(self.onImgUpdate)
        self.image.linePicked.connect(self.onLinePickedFromImage)
        self.image.nucleusPicked.connect(self.onNucleusPickedFromImage)
        self.image.dnaChannel = self.ctrl.dnaSpin.value()
        self.image.actChannel = self.ctrl.actSpin.value()

        self.grph = GraphWidget()
        self.grph.show()

        self.stk = StkRingWidget(linePicked=self.onLinePickedFromStackGraph)
        self.stk.show()

        self.grph.linePicked.connect(self.onLinePickedFromGraph)
        # self.stk.linePicked.connect(self.onLinePickedFromStackGraph)

        self.ctrl.setWindowFlags(self.ctrl.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint)
        self.grph.setWindowFlags(self.grph.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint)
        self.stk.setWindowFlags(self.stk.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint)

        self.image.dnaChannel = self.ctrl.dnaSpin.value()
        self.image.actChannel = self.ctrl.actSpin.value()

        self.measure_n = 0
        self.selectedLine = None

        self.currMeasurement = None
        self.currN = None
        self.currZ = None

        self.df = pd.DataFrame()
        self.file = "/Users/Fabio/data/lab/airyscan/nil.czi"

        self.resizeEvent(None)
        self.moveEvent(None)

    def resizeEvent(self, event):
        # this is a hack to resize everything when the user resizes the main window
        self.grph.setFixedWidth(self.width())
        self.image.setFixedWidth(self.width())
        self.image.setFixedHeight(self.height())
        self.image.resizeEvent(None)
        self.moveEvent(None)

    def moveEvent(self, QMoveEvent):
        px = self.geometry().x()
        py = self.geometry().y()
        pw = self.geometry().width()
        ph = self.geometry().height()

        dw = self.ctrl.width()
        dh = self.ctrl.height()
        self.ctrl.setGeometry(px + pw, py, dw, dh)

        sw = self.stk.width()
        sh = self.stk.height()
        self.stk.setGeometry(px + pw + dw, py, sw, sh)
        self.stk.moveEvent(None)

        dw = self.grph.width()
        dh = self.grph.height()
        self.grph.setGeometry(px, py + ph + 20, dw, dh)
        # super(RingWindow, self).mouseMoveEvent(event)

    def closeEvent(self, event):
        if not self.df.empty:
            self.df.loc[:, "condition"] = self.ctrl.experimentLineEdit.text()
            self.df.loc[:, "l"] = self.df.loc[:, "l"].apply(lambda v: np.array2string(v, separator=','))
            self.df.to_csv(os.path.join(os.path.dirname(self.image.file), "ringlines.csv"))
        self.grph.close()
        self.ctrl.close()
        self.stk.close()

    def focusInEvent(self, QFocusEvent):
        logger.debug('focusInEvent')
        self.ctrl.activateWindow()
        self.grph.activateWindow()
        self.stk.focusInEvent(None)

    def showEvent(self, event):
        self.setFocus()

    def _graphTendency(self):
        df = pd.DataFrame(self.image.measurements).drop(['x', 'y', 'c', 'ls0', 'ls1', 'd', 'sum'], axis=1)
        df.loc[:, "xx"] = df.loc[:, "l"].apply(
            lambda v: np.arange(start=0, stop=len(v) * self.image.dl, step=self.image.dl))
        df = m.vector_column_to_long_fmt(df, val_col="l", ix_col="xx")
        sns.lineplot(x="xx", y="l", data=df, ax=self.grph.ax, color='k', ci="sd", zorder=20)
        self.grph.ax.set_ylabel('')
        self.grph.ax.set_xlabel('')
        self.grph.canvas.draw()

    def _graph(self, alpha=1.0):
        self.grph.clear()
        if self.image.measurements is not None:
            for me in self.image.measurements:
                x = np.arange(start=0, stop=len(me['l']) * self.image.dl, step=self.image.dl)
                lw = 0.1 if self.image.selectedLine is not None and me != self.image.selectedLine else 0.5
                self.grph.ax.plot(x, me['l'], linewidth=lw, linestyle='-', color=me['c'], alpha=alpha, zorder=10,
                                  picker=5, label=me['n'])
            self.grph.format_ax()
            self.statusbar.showMessage("ptp: %s" % ["%d " % me['d'] for me in self.image.measurements])
            self.grph.canvas.draw()

    @QtCore.pyqtSlot()
    def onImgToggle(self):
        logger.debug('onImgToggle')
        if self.ctrl.dnaChk.isChecked():
            self.image.activeCh = "dna"
        if self.ctrl.actChk.isChecked():
            self.image.activeCh = "act"

    @QtCore.pyqtSlot()
    def onRenderChk(self):
        logger.debug('onRenderChk')
        self.image.render = self.ctrl.renderChk.isChecked()
        self.stk.render = self.ctrl.renderChk.isChecked()

    @QtCore.pyqtSlot()
    def onOpenButton(self):
        logger.debug('onOpenButton')
        qfd = QtGui.QFileDialog()
        path = os.path.dirname(self.file)
        if self.image.file is not None:
            self.statusbar.showMessage("current file: %s" % os.path.basename(self.image.file))
        flt = "zeiss(*.czi)"
        f = QtGui.QFileDialog.getOpenFileName(qfd, "Open File", path, flt)
        if len(f) > 0:
            self.image.file = f
            self.image.zstack = self.ctrl.zSpin.value()
            self.image.dnaChannel = self.ctrl.dnaSpin.value()
            self.ctrl.nchLbl.setText("%d channels" % self.image.nChannels)
            self.ctrl.nzsLbl.setText("%d z-stacks" % self.image.nZstack)
            self.ctrl.nfrLbl.setText("%d %s" % (self.image.nFrames, "frames" if self.image.nFrames > 1 else "frame"))
            self.currMeasurement = None
            self.currN = None
            self.currZ = None

            self.stk = StkRingWidget(linePicked=self.onLinePickedFromStackGraph,
                                     stacks=self.image.nZstack,
                                     n_channels=self.image.nChannels,
                                     dna_ch=self.image.dnaChannel,
                                     rng_ch=self.image.actChannel,
                                     line_length=self.image.dl,
                                     lines_to_measure=_nlin,
                                     pix_per_um=self.image.pix_per_um
                                     )
            # self.stk.linePicked.connect(self.onLinePickedFromStackGraph)
            self.stk.loadImages(self.image.images, xy=(100, 100), wh=(200, 200))
            self.stk.hide()
            self.stk.show()
            self.moveEvent(None)

    @QtCore.pyqtSlot()
    def onImgUpdate(self):
        # logger.debug(f"onImgUpdate")
        self.ctrl.renderChk.setChecked(True)
        self.stk.selectedN = self.image.selectedLine['n'] if self.image.selectedLine is not None else 0
        logger.debug(f"onImgUpdate. Selected line is {self.stk.selectedN}")
        self.stk.loadImages(self.image.images, xy=[n[0] for n in self.image.currNucleus.centroid.xy],
                            wh=(30 * self.image.pix_per_um, 30 * self.image.pix_per_um))
        self.stk.drawMeasurements()
        self._graph()

    @QtCore.pyqtSlot()
    def onNucleusPickedFromImage(self):
        logger.debug('onNucleusPickedFromImage')
        self.stk.dnaChannel = self.image.dnaChannel
        self.stk.rngChannel = self.image.actChannel
        self.stk.selectedN = self.image.selectedLine['n'] if self.image.selectedLine is not None else 0

        self.stk.loadImages(self.image.images, xy=[n[0] for n in self.image.currNucleus.centroid.xy],
                            wh=(30 * self.image.pix_per_um, 30 * self.image.pix_per_um))
        try:
            self.stk.measure()
            self.stk.drawMeasurements()
        except Exception as e:
            logger.error(e)

    @QtCore.pyqtSlot()
    def onMeasureButton(self):
        logger.debug('onMeasureButton')
        self.image.paint_measures()
        self._graph(alpha=0.2)
        self._graphTendency()

    @QtCore.pyqtSlot()
    def onZValueChange(self):
        logger.debug('onZValueChange')
        self.image.zstack = self.ctrl.zSpin.value() % self.image.nZstack
        self.ctrl.zSpin.setValue(self.image.zstack)
        self._graph()

    @QtCore.pyqtSlot()
    def onDnaValChange(self):
        logger.debug('onDnaValChange')
        val = self.ctrl.dnaSpin.value() % self.image.nChannels
        self.ctrl.dnaSpin.setValue(val)
        self.image.dnaChannel = val
        if self.ctrl.dnaChk.isChecked():
            self.image.activeCh = "dna"
        self.ctrl.dnaChk.setChecked(True)

    @QtCore.pyqtSlot()
    def onActValChange(self):
        logger.debug('onActValChange')
        val = self.ctrl.actSpin.value() % self.image.nChannels
        self.ctrl.actSpin.setValue(val)
        self.image.actChannel = val
        if self.ctrl.actChk.isChecked():
            self.image.activeCh = "act"
        self.ctrl.actChk.setChecked(True)

    @QtCore.pyqtSlot()
    def onAddButton(self):
        logger.debug('onAddButton')
        if self.currMeasurement is not None and self.currN is not None and self.currZ is not None:
            new = pd.DataFrame(self.currMeasurement)
            new = new.loc[(new["n"] == self.currN) & (new["z"] == self.currZ)]
            new.loc[:, "m"] = self.measure_n
            new.loc[:, "file"] = os.path.basename(self.image.file)
            # new.loc[:, "x"] = new.loc[:, "l"].apply(lambda v: np.arange(start=0, stop=len(v), step=self.image.dl))
            self.df = self.df.append(new, ignore_index=True, sort=False)
            self.measure_n += 1
            self.currMeasurement = None
            self.currN = None
            self.currZ = None

            print(self.df)

    @QtCore.pyqtSlot()
    def onPlotButton(self):
        logger.debug('onPlotButton')
        if self.image.measurements is None: return
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.gridspec import GridSpec

        plt.style.use('bmh')
        pal = sns.color_palette("Blues", n_colors=len(self.image.measurements))
        fig = plt.figure(figsize=(2, 2 * 4), dpi=300)
        gs = GridSpec(nrows=2, ncols=1, height_ratios=[4, 0.5])
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[1, 0])
        self.image.drawMeasurements(ax1, pal)

        lw = 1
        for me, c in zip(self.image.measurements, pal):
            x = np.arange(start=0, stop=len(me['l']) * self.image.dl, step=self.image.dl)
            ax2.plot(x, me['l'], linewidth=lw, linestyle='-', color=c, alpha=1, zorder=10)

        ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax1.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax1.yaxis.set_minor_locator(ticker.MultipleLocator(10))

        ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(1e4))
        ax2.yaxis.set_minor_locator(ticker.MultipleLocator(5e3))
        ax2.yaxis.set_major_formatter(EngFormatter(unit=''))

        fig.savefig(os.path.basename(self.image.file) + ".pdf")

    @QtCore.pyqtSlot()
    def onLinePickedFromGraph(self):
        logger.debug('onLinePickedFromGraph')
        self.selectedLine = self.grph.selectedLine if self.grph.selectedLine is not None else None
        if self.selectedLine is not None:
            self.image.selectedLine = self.selectedLine

            self.currMeasurement = self.image.measurements
            self.currN = self.selectedLine
            self.currZ = self.image.zstack
            self.stk.selectedN = self.currN
            self.stk.selectedZ = self.currZ

            self.statusbar.showMessage("line %d selected" % self.selectedLine)

    @QtCore.pyqtSlot()
    def onLinePickedFromStackGraph(self):
        logger.debug('onLinePickedFromStackGraph')
        self.selectedLine = self.stk.selectedN if self.stk.selectedN is not None else None
        if self.selectedLine is not None:
            self.currMeasurement = self.stk.measurements
            self.currN = self.stk.selectedN
            self.currZ = self.stk.selectedZ

            self.statusbar.showMessage(f"line {self.selectedLine} selected {self.stk.selectedLine}")
            logger.info(f"line {self.selectedLine} selected {self.stk.selectedLine}")

    @QtCore.pyqtSlot()
    def onLinePickedFromImage(self):
        logger.debug('onLinePickedFromImage')
        self.selectedLine = self.image.selectedLine['n'] if self.image.selectedLine is not None else None
        if self.selectedLine is not None:
            self.currMeasurement = self.image.measurements
            self.currN = self.selectedLine
            self.currZ = self.image.zstack
            self.stk.selectedN = self.currN
            self.stk.selectedZ = self.currZ

            self.statusbar.showMessage("line %d selected" % self.selectedLine)


if __name__ == '__main__':
    from PyQt4.QtCore import QT_VERSION_STR
    from PyQt4.Qt import PYQT_VERSION_STR

    base_path = os.path.abspath('%s' % os.getcwd())
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.info('Qt version:' + QT_VERSION_STR)
    logging.info('PyQt version:' + PYQT_VERSION_STR)
    logging.info('Working dir:' + os.getcwd())
    logging.info('Base dir:' + base_path)
    os.chdir(base_path)

    app = QtGui.QApplication(sys.argv)

    gui = RingWindow()
    gui.show()

    sys.exit(app.exec_())
