import copy
from numbers import Number
from queue import Queue, Empty
from itertools import combinations
from types import SimpleNamespace
from typing import Iterable
from threading import Timer, Lock
import time
import numpy as np

from AnyQt.QtCore import QModelIndex, Qt, QAbstractTableModel, QSortFilterProxyModel
from AnyQt.QtWidgets import QTableView, QVBoxLayout, QHeaderView

from Orange.data import Variable, Table
from Orange.widgets import gui
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from Orange.widgets.utils.signals import Input


class RankModel(QAbstractTableModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.__sortInd: np.ndarray
        # self.__sortColumn = -1
        # self.__sortOrder = Qt.DescendingOrder

        self._data = None  # type: np.ndarray
        self._columns = 0
        self._rows = 0
        self._max_rows = int(1e9)
        self._headers = {}

    def set_domain(self, table):
        self._domain = table.domain
        n_attrs = len(table.domain.attributes)
        self._n_comb = n_attrs * (n_attrs - 1) // 2

    def set_scorer(self, scorer):
        self.scorer = scorer

    def rowCount(self, parent=QModelIndex(), *args, **kwargs):
        return 0 if parent.isValid() else min(self._rows, self._max_rows)

    def columnCount(self, parent=QModelIndex(), *args, **kwargs):
        return 0 if parent.isValid() else self._columns

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return

        # row = self.mapToSourceRows(index.row())
        row = index.row()
        col = index.column()

        try:
            value = self._data[row, col]
            if role == Qt.EditRole:
                return value
            if col >= self._columns - 2:
                value = self._domain[value]
        except IndexError:
            return
        if role == Qt.DecorationRole and isinstance(value, Variable):
            return gui.attributeIconDict[value]
        if role == Qt.DisplayRole:
            if isinstance(value, Number):
                return "{:.1f}%".format(100 * value)
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

    def initialize(self, data):
        self.beginResetModel()
        self._data = np.array(data)
        self._rows, self._columns = self._data.shape
        # self.unsort()
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self._data = None
        self._rows = self._columns = 0
        # self.unsort()
        self.endResetModel()

    def append(self, rows):
        if not isinstance(self._data, np.ndarray):
            return self.initialize(rows)

        n_rows = len(rows)
        n_data = len(self._data)
        insert = self._rows < self._max_rows

        if insert:
            self.beginInsertRows(QModelIndex(), self._rows, min(self._max_rows, self._rows + n_rows) - 1)

        if self._rows + n_rows >= n_data:
            n_data = min(max(n_data + n_rows, 2 * n_data), self._n_comb)
            ar = np.full((n_data, self._columns), np.nan)
            ar[:self._rows] = self._data[:self._rows]
            self._data = ar

        self._data[self._rows:self._rows + n_rows] = rows
        self._rows += n_rows

        if insert:
            self.endInsertRows()

        # if self.__sortColumn >= 0:
        #     self.sort_new(n_rows)

    def __len__(self):
        return self._rows

    def source_data(self):
        return self._data


class ModelProxy(QSortFilterProxyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__sortInd: np.ndarray
        self.__sortColumn = -1
        self.__sortOrder = Qt.DescendingOrder

    def setSourceModel(self, model):
        super().setSourceModel(model)
        self._data = model.source_data()

    def sort(self, column: int, order=Qt.DescendingOrder):
        if self._data is None:
            return
        indices = self._sort(column, order)
        self.__sortColumn = column
        self.__sortOrder = order
        self._setSortIndices(indices)

    def _setSortIndices(self, indices):
        self.layoutAboutToBeChanged.emit([], QAbstractTableModel.VerticalSortHint)
        self.__sortInd = indices
        self.layoutChanged.emit([], QAbstractTableModel.VerticalSortHint)

    def _sort(self, column, order):
        if column < 0:
            return

        data = self._data[:self._rows, column]
        return self._argsortData(data, order)

    def _argsortData(self, data, order):
        indices = np.argsort(data, kind="mergesort")
        if order == Qt.DescendingOrder:
            indices = indices[::-1]
        return indices

    def sort_new(self, n_rows):
        data = self._data[:self._rows, self.__sortColumn]
        old_rows = self._rows - n_rows
        ind = np.arange(old_rows, self._rows)
        order = 1 if self.__sortOrder == Qt.AscendingOrder else -1
        loc = np.searchsorted(data[:old_rows],
                              data[old_rows:self._rows],
                              sorter=self.__sortInd[::order])
        indices = np.insert(self.__sortInd[::order], loc, ind)[::order]
        self._setSortIndices(indices)

    def unsort(self):
        return self.sort(-1)

    def mapToSourceRows(self, rows):
        if self.__sortInd is not None and \
                (isinstance(rows, (int, type(Ellipsis)))
                 or len(rows)):
            return self.__sortInd[rows]
        return rows

    def mapToSource(self, index):
        return index


class Widget(OWWidget, ConcurrentWidgetMixin):
    name = "npTestWidget"
    want_control_area = False

    class Inputs:
        data = Input("Data", Table)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.attrs = 0
        self.progress = 0
        self.running = False
        self.saved_state = None

        self.setLayout(QVBoxLayout())

        self.model = RankModel()
        self.model.setHorizontalHeaderLabels([
            "Score 1", "Feature 1", "Feature 2"
        ])
        self.proxy = ModelProxy()
        self.proxy.setSourceModel(self.model)
        view = QTableView(selectionBehavior=QTableView.SelectRows,
                          selectionMode=QTableView.SingleSelection,
                          showGrid=False,
                          editTriggers=gui.TableView.NoEditTriggers)
        view.setSortingEnabled(True)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        view.setModel(self.proxy)

        self.button = gui.button(self, self, "do stuff", callback=self.toggle)

        self.layout().addWidget(view)
        self.layout().addWidget(self.button)

    def toggle(self):
        self.running = not self.running
        if self.running:
            self.button.setText("Pause")
            self.button.repaint()
            self.progressBarInit()
            self.start(run, compute_score,
                       self.attrs, self.iterate_states,
                       self.saved_state, self.progress)
        else:
            self.button.setText("Continue")
            self.button.repaint()
            self.cancel()
            self.progressBarFinished()

    def on_partial_result(self, result):
        rows = []
        try:
            while True:
                queued = result.get_nowait()
                self.saved_state = queued.next_state
                row_item = list(queued.score) + list(queued.state)
                rows.append(row_item)
        except Empty:
            if rows:
                self.model.append(rows)

        self.progress = len(self.model)
        self.progressBarSet(int(self.progress * 100 / (self.attrs*(self.attrs-1)//2)))

    def on_done(self, result):
        self.button.setText("done")

    @Inputs.data
    def set_data(self, data):
        self.model.set_domain(data)
        self.attrs = len(data.domain.attributes)

    def iterate_states(self, initial_state):
        si, sj = initial_state or (0, 0)
        for i in range(si, self.attrs):
            for j in range(sj, i):
                yield i, j
            sj = 0


class QueuedScore(SimpleNamespace):
    score = None  # type: float
    state = None  # type: Iterable
    next_state = None  # type: Iterable


def compute_score(state):
    time.sleep(1e-7)
    return sum(state),


def run(compute, attrs, iterate_states, saved_state, progress, task):
    task.set_status("Getting combinations...")
    task.set_progress_value(0.1)
    states = iterate_states(saved_state)

    task.set_status("Getting scores...")
    queue = Queue()
    can_set_partial_result = True

    def do_work(st, next_st):
        try:
            score = compute(st)
            if score is not None:
                queue.put_nowait(QueuedScore(score=score, state=st, next_state=next_st))
        except Exception:
            pass

    def reset_flag():
        nonlocal can_set_partial_result
        can_set_partial_result = True

    state = None
    next_state = next(states)
    try:
        while True:
            if task.is_interruption_requested():
                return queue
            task.set_progress_value(progress * 100 // (attrs*(attrs-1)//2))
            progress += 1
            state = copy.copy(next_state)
            next_state = copy.copy(next(states))
            do_work(state, next_state)
            if can_set_partial_result:
                task.set_partial_result(queue)
                can_set_partial_result = False
                Timer(0.05, reset_flag).start()
    except StopIteration:
        do_work(state, None)
        task.set_partial_result(queue)
    return queue


if __name__ == "__main__":
    WidgetPreview(Widget).run(Table("/Users/noah/Nextcloud/Fri/tables/GDS3713-small.tab"))
