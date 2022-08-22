from queue import Queue, Empty
from itertools import combinations
from types import SimpleNamespace
from typing import Optional, Iterable, List, Callable
from threading import Timer, Lock
import time
import tqdm
import sys
import bisect
import copy
import numpy as np

from AnyQt.QtCore import QModelIndex, Qt, QAbstractTableModel, QSize, QLineF, \
    QSortFilterProxyModel, QItemSelection, QItemSelectionModel, pyqtSignal as Signal
from AnyQt.QtWidgets import QTableView, QVBoxLayout, QHeaderView, QLabel, QDialog, QLineEdit, \
    QStyleOptionViewItem, QApplication, QStyle
from AnyQt.QtGui import QColor, QPainter, QPen

from Orange.data import Variable, Table
from Orange.preprocess import Discretize
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.visualize.utils import VizRankDialogAttrPair
from Orange.widgets.utils.concurrent import ConcurrentMixin, TaskState
from Orange.widgets.utils.messages import WidgetMessagesMixin
from Orange.widgets.utils.progressbar import ProgressBarMixin
from Orange.widgets.widget import Msg

from orangecontrib.prototypes.widgets.nptablemodel import RankModel
from orangecontrib.prototypes.interactions import Interaction
from orangecontrib.prototypes.widgets.interactions_widget import InteractionItemDelegate


class VizRankWidget(OWWidget, ConcurrentWidgetMixin):
    name = "Interaction Rank"
    want_control_area = False

    # feature = ContextSetting(None)

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.data = None
        self.progress = 0
        self.running = False

        self.setLayout(QVBoxLayout())

        self.model = RankModel(self)
        self.model.setHorizontalHeaderLabels(["Score 1", "Score 2", "Feature 1", "Feature 2"])
        # self.model_proxy = QSortFilterProxyModel(self)
        # self.model_proxy.setSourceModel(self.model)
        self.view = QTableView(selectionBehavior=QTableView.SelectRows,
                               selectionMode=QTableView.SingleSelection,
                               showGrid=False,
                               editTriggers=gui.TableView.NoEditTriggers)
        self.view.setItemDelegate(InteractionItemDelegate())
        self.view.setModel(self.model)
        self.view.setSortingEnabled(True)
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.view.selectionModel().selectionChanged().connect(self.on_selection_changed)

        self.button = gui.button(self, self, "Start", callback=self.toggle, default=True)

        self.layout().addWidget(self.view)
        self.layout().addWidget(self.button)

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.initialize()
        self.button.setEnabled(data is not None)

    def toggle(self):
        if self.running:
            pass
        else:
            pass
        self.running = not self.running

    def initialize(self):
        self.running = False
        pass


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(VizRankWidget).run(Table("iris"))
