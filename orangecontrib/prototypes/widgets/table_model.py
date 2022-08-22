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


class RankTableModel(QAbstractTableModel):
    @staticmethod
    def _RoleData():
        return defaultdict(lambda: defaultdict(dict))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__sortInd: np.ndarray
        self.__sortColumn = -1
        self.__sortOrder = Qt.DescendingOrder

        self._array: np.ndarray
        self._max_rows = 10
        self._data_rows = 0
        self._array_rows = self._array_growth = 1000
        self._columns = 4
        self._headers = {}
        self._roleData: defaultdict

        self.initialize(np.full((self._array_rows, self._columns), np.nan))

    def rowCount(self, parent=QModelIndex(), *args, **kwargs):
        return 0 if parent.isValid() else min(self._data_rows, self._max_rows)

    def columnCount(self, parent=QModelIndex(), *args, **kwargs):
        return 0 if parent.isValid() else self._columns

    def mapToSourceRows(self, rows):
        if self.__sortInd is not None and \
                (isinstance(rows, (int, type(Ellipsis)))
                 or len(rows)):
            return self.__sortInd[rows]
        return rows

    def resetSorting(self):
        return self.sort(-1)

    def _argsortData(self, data: np.ndarray, order):
        indices = np.argsort(data, kind="mergesort")
        if order == Qt.DescendingOrder:
            indices = np.roll(indices[::-1], self._data_rows)
        return indices

    def _sort(self, column, order):
        if column < 0:
            return

        data = self._array[:, column]
        return self._argsortData(data, order)

    def _setSortIndices(self, indices):
        self.layoutAboutToBeChanged.emit([], QAbstractTableModel.VerticalSortHint)
        self.__sortInd = indices
        self.layoutChanged.emit([], QAbstractTableModel.VerticalSortHint)

    def sort(self, column: int, order=Qt.DescendingOrder):
        indices = self._sort(column, order)
        self.__sortColumn = column
        self.__sortOrder = order
        self._setSortIndices(indices)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return

        row = self.mapToSourceRows(index.row())
        column = index.column()

        role_value = self._roleData.get(row, {}).get(column, {}).get(role)
        if role_value is not None:
            return role_value

        try:
            value = self._array[row, column]
        except IndexError:
            return
        if role == Qt.EditRole:
            return value
        if role == Qt.DecorationRole and isinstance(value, Variable):
            return gui.attributeIconDict[value]
        if role == Qt.DisplayRole:
            if (isinstance(value, Number) and
                    not (np.isnan(value) or np.isinf(value) or isinstance(value, Integral))):
                abs_val = abs(value)
                str_len = len(str(int(abs_val)))
                value = '{:.{}{}}'.format(value,
                                          2 if abs_val < .001 else
                                          3 if str_len < 2 else
                                          1 if str_len < 5 else
                                          0 if str_len < 6 else
                                          3,
                                          'f' if (abs_val == 0 or
                                                  abs_val >= .001 and
                                                  str_len < 6)
                                          else 'e')
            return str(value)
        if role == Qt.TextAlignmentRole and isinstance(value, Number):
            return Qt.AlignRight | Qt.AlignVCenter
        if role == Qt.ToolTipRole:
            return str(value)

    def setHorizontalHeaderLabels(self, labels):
        self._headers[Qt.Horizontal] = tuple(labels)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        headers = self._headers.get(orientation)
        if headers and section < len(headers) and role == Qt.DisplayRole:
            return headers[section]
        return super().headerData(section, orientation, role)

    def initialize(self, array):
        self.beginResetModel()
        self._array = array
        self._roleData = self._RoleData()
        self.resetSorting()
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self._data_rows = 0
        self._array_rows = self._array_growth
        self._array = np.full((self._array_rows, self._columns), np.nan)
        self._roleData.clear()
        self.resetSorting()
        self.endResetModel()

    def extend(self, rows):
        # rows: 2-d list or array
        if self.__sortColumn >= 0:
            self.resetSorting()

        count = len(rows)
        if self._data_rows < self._max_rows:
            insert_row = min(self._data_rows, self._max_rows)
            new_rows = min(count, self._max_rows - self._data_rows)
            self.beginInsertRows(QModelIndex(), insert_row, insert_row + new_rows - 1)

        if self._data_rows + count >= self._array_rows:
            self._pad_array(count)

        self._array[self._data_rows:self._data_rows + count] = rows
        # self.dataChanged.emit(self.index(self._data_rows, 0), self.index(self._data_rows + count, 3))

        if self._data_rows < self._max_rows:
            self.endInsertRows()
        self._data_rows += count

    def _pad_array(self, count):
        self._array_rows = self._data_rows + count + self._array_growth
        new_array = np.full((self._array_rows, self._columns), np.nan)
        new_array[:self._data_rows] = self._array[:self._data_rows]
        self._array = new_array


class Widget(OWWidget, ConcurrentWidgetMixin):
    name = "TestWidget"
    want_control_area = False

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.setLayout(QVBoxLayout())

        self.model = RankTableModel()
        self.model.setHorizontalHeaderLabels([
            "Score 1", "Score 2", "Feature 1", "Feature 2"
        ])
        view = QTableView()
        view.setSortingEnabled(True)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        view.setModel(self.model)

        button = gui.button(self, self, "do stuff", callback=self.push_button)

        self.timer_label = QLabel("click the button")

        self.layout().addWidget(self.timer_label)
        self.layout().addWidget(view)
        self.layout().addWidget(button)

        # self.item_store = ItemStore()
        # self.model_queue = Queue()

    def sizeHint(self):
        return QSize(350, 400)

    def push_button(self):
        self.start(do_stuff, self.compute_score, self.timer_label)

    @staticmethod
    def compute_score(state):
        return sum(state), -sum(state)

    def on_partial_result(self, result_queue):
        items = []
        while not result_queue.empty():
            items.append(result_queue.get_nowait())
        if items:
            self.model.extend(items)

    def on_done(self, result) -> None:
        self.timer_label.setText("done")


def do_stuff(compute, timer_label: QLabel, task: TaskState):
    # item_store = ItemStore()
    result_queue = Queue()
    can_set_partial_result = True

    def reset_flag():
        nonlocal can_set_partial_result
        can_set_partial_result = True

    for state in tqdm.tqdm(combinations(range(500), 2), total=500*499//2):
        timer_label.setText("working: " + str(np.random.random()))
        score = compute(state)
        result_queue.put_nowait(list(score) + list(state))
        if can_set_partial_result:
            task.set_partial_result(result_queue)
            can_set_partial_result = False
            Timer(0.1, reset_flag).start()

    """
    result = ItemStore()
    can_set_partial_result = True

    def reset_flag():
        nonlocal can_set_partial_result
        can_set_partial_result = True

    for state in tqdm.tqdm(combinations(range(N), 2), total=N*(N-1)//2):
        score = compute(state)
        result.put(list(score) + list(state))
        if can_set_partial_result:
            task.set_partial_result(result)
            can_set_partial_result = False
            Timer(0.05, reset_flag).start()
    task.set_partial_result(result)
    """


if __name__ == "__main__":
    WidgetPreview(Widget).run()
