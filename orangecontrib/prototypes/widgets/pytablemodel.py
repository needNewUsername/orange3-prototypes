from collections import defaultdict
from numbers import Number, Integral
from queue import Queue, Empty
from itertools import combinations
from types import SimpleNamespace
from threading import Timer, Lock
import time
import tqdm
import sys

import numpy as np

from AnyQt.QtCore import QModelIndex, Qt, QAbstractTableModel, QSize
from AnyQt.QtWidgets import QTableView, QVBoxLayout, QHeaderView, QLabel

from Orange.data import Variable
from Orange.widgets import gui
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState


class RankModel(QAbstractTableModel):
    @staticmethod
    def _RoleData():
        return defaultdict(lambda: defaultdict(dict))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._data = None
        self._columns = 4
        self._rows = 0
        self._max_rows = 1000000000
        self._headers = {}
        self.roleData: defaultdict

        self.initialize()

    def rowCount(self, parent=QModelIndex(), *args, **kwargs):
        return 0 if parent.isValid() else min(self._rows, self._max_rows)

    def columnCount(self, parent=QModelIndex(), *args, **kwargs):
        return 0 if parent.isValid() else self._columns

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return

        row, col = index.row(), index.column()

        role_value = self._roleData.get(row, {}).get(col, {}).get(role)
        if role_value is not None:
            return role_value

        try:
            value = self._data[row][col]
        except IndexError:
            return
        if role != Qt.DisplayRole:
            return

        return value

    def setHorizontalHeaderLabels(self, labels):
        self._headers[Qt.Horizontal] = tuple(labels)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        headers = self._headers.get(orientation)
        if headers and section < len(headers) and role == Qt.DisplayRole:
            return headers[section]
        return super().headerData(section, orientation, role)

    def initialize(self):
        self.beginResetModel()
        self._data = []
        self._roleData = self._RoleData()
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self._data = []
        self._rows = 0
        self._roleData.clear()
        self.endResetModel()

    def append(self, rows):
        count = len(rows)
        insert = self._rows < self._max_rows
        if insert:
            self.beginInsertRows(QModelIndex(), self._rows, min(self._max_rows, self._rows + count) - 1)

        self._data += rows
        self._rows += count

        if insert:
            self.endInsertRows()


class Widget(OWWidget, ConcurrentWidgetMixin):
    name = "TestWidget"
    want_control_area = False

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.setLayout(QVBoxLayout())

        self.model = RankModel()
        self.model.setHorizontalHeaderLabels([
            "Score 1", "Score 2", "Feature 1", "Feature 2"
        ])
        view = QTableView()
        view.setSortingEnabled(True)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        view.setModel(self.model)

        self.button = gui.button(self, self, "do stuff", callback=self.push_button)

        self.layout().addWidget(view)
        self.layout().addWidget(self.button)

        self.N = 1000
        self.progress = 0

    def sizeHint(self):
        return QSize(350, 400)

    def push_button(self):
        self.button.setDisabled(True)
        self.start(do_stuff, compute_score, self.N)

    def on_partial_result(self, result):
        rows = result.get()
        if rows:
            self.model.append(rows)

        self.progress = self.model._rows
        self.progressBarSet(int(self.progress * 100 / (self.N*(self.N-1)//2)))

    def on_done(self, result):
        self.button.setText("done")


class ItemStore(object):
    def __init__(self):
        self.lock = Lock()
        self.items = []

    def put(self, item):
        with self.lock:
            self.items.append(item)

    def get(self):
        with self.lock:
            items, self.items = self.items, []
        return items

    def size(self):
        return len(self.items)


def compute_score(state):
    time.sleep(0.000001)
    return sum(state), -sum(state)


def do_stuff(compute, N, task):
    # result = Queue()
    result = ItemStore()
    can_set_partial_result = True

    def reset_flag():
        nonlocal can_set_partial_result
        can_set_partial_result = True

    for state in tqdm.tqdm(combinations(range(N), 2), total=N*(N-1)//2):
        score = compute(state)
        # result.put_nowait(list(score) + list(state))
        result.put(list(score) + list(state))
        if can_set_partial_result:
            task.set_partial_result(result)
            can_set_partial_result = False
            Timer(0.1, reset_flag).start()
    task.set_partial_result(result)


if __name__ == "__main__":
    WidgetPreview(Widget).run()
