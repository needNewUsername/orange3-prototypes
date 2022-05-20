from numbers import Number
from queue import Queue, Empty
from itertools import combinations
from types import SimpleNamespace
from typing import Iterable
from threading import Timer, Lock
import time
import tqdm
import numpy as np

from AnyQt.QtCore import QModelIndex, Qt, QAbstractTableModel
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

        self.__sortInd: np.ndarray
        self.__sortColumn = -1
        self.__sortOrder = Qt.DescendingOrder

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

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return

        row = self.mapToSourceRows(index.row())
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
        self.unsort()
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self._data = None
        self._rows = self._columns = 0
        self.unsort()
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

        if self.__sortColumn >= 0:
            self.sort_new(n_rows)

    def __len__(self):
        return self._rows


class Widget(OWWidget, ConcurrentWidgetMixin):
    name = "npTestWidget"
    want_control_area = False

    class Inputs:
        data = Input("Data", Table)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.N = 0
        self.progress = 0
        self.add_to_model = Queue()

        self.setLayout(QVBoxLayout())

        self.model = RankModel()
        self.model.setHorizontalHeaderLabels([
            "Score 1", "Feature 1", "Feature 2"
        ])
        view = QTableView(selectionBehavior=QTableView.SelectRows,
            selectionMode=QTableView.SingleSelection,
            showGrid=False,
            editTriggers=gui.TableView.NoEditTriggers)
        view.setSortingEnabled(True)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        view.setModel(self.model)

        self.button = gui.button(self, self, "do stuff", callback=self.push_button)

        self.layout().addWidget(view)
        self.layout().addWidget(self.button)

    def push_button(self):
        self.button.setDisabled(True)
        self.start(do_stuff, compute_score, self.N)

    def on_partial_result(self, result):
        try:
            while True:
                queued = result.queue.get_nowait()
                self.add_to_model.put_nowait(queued)
        except Empty:
            pass

        rows = []
        try:
            while True:
                queued = self.add_to_model.get_nowait()
                row_item = list(queued.score) + list(queued.state)
                rows.append(row_item)
        except Empty:
            if rows:
                self.model.append(rows)

        self.progress = self.model._rows
        self.progressBarSet(int(self.progress * 100 / (self.N*(self.N-1)//2)))

    def on_done(self, result):
        self.button.setText("done")

    @Inputs.data
    def set_data(self, data):
        self.model.set_domain(data)
        self.N = len(data.domain.attributes)


class ItemStore(object):
    def __init__(self):
        self.lock = Lock()
        self.items = []

    def put(self, item):
        with self.lock:
            self.items.append(item)

    def put_nowait(self, item):
        return self.put(item)

    def get(self):
        with self.lock:
            items, self.items = self.items, []
        return items


class Result(SimpleNamespace):
    queue = None  # type: Queue[QueuedScore, ...]
    # scores = None  # type: Optional[List[float, ...]]


class QueuedScore(SimpleNamespace):
    # position = None  # type: int
    score = None  # type: float
    state = None  # type: Iterable
    # next_state = None  # type: Iterable


def compute_score(state):
    time.sleep(0.0000001)
    return sum(state),


def do_stuff(compute, N, task):
    task.set_status("Getting scores...")
    res = Result(queue=Queue())
    can_set_partial_result = True

    def reset_flag():
        nonlocal can_set_partial_result
        can_set_partial_result = True

    for state in tqdm.tqdm(combinations(range(N), 2), total=N*(N-1)//2):
        score = compute(state)
        res.queue.put_nowait(QueuedScore(score=score,
                                         state=state))
        if can_set_partial_result:
            task.set_partial_result(res)
            can_set_partial_result = False
            Timer(0.05, reset_flag).start()
    task.set_partial_result(res)


if __name__ == "__main__":
    WidgetPreview(Widget).run(Table("/Users/noah/Nextcloud/Fri/tables/GDS3713-small.tab"))
