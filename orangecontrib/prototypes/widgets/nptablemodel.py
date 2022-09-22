import copy
from numbers import Number
from typing import Iterable, Callable
from threading import Timer, Lock
import numpy as np

from AnyQt.QtCore import QModelIndex, Qt, QAbstractTableModel
from AnyQt.QtWidgets import QTableView, QVBoxLayout, QHeaderView, QLineEdit

from Orange.data import Variable, Table
from Orange.preprocess import Discretize
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.signals import Input
from Orange.widgets.utils.itemmodels import DomainModel

from orangecontrib.prototypes.interactions import InteractionScorer


MAX_ROWS = int(1e9)


class RankModel(QAbstractTableModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__sortInd = ...  # type: np.ndarray
        self.__sortColumn = -1
        self.__sortOrder = Qt.DescendingOrder

        self.__filterInd = ...  # type: np.ndarray
        self.__filterAttrs = None

        self._data = None  # type: np.ndarray
        self._columns = 0
        self._rows = 0
        self._max_rows = MAX_ROWS
        self._headers = []

    def set_domain(self, domain):
        self.domain = domain
        n_attrs = len(domain.attributes)
        self.n_comb = n_attrs * (n_attrs - 1) // 2
        self.attr_names = [attr.name.lower() for attr in domain.attributes]

    def set_scorer(self, scorer):
        self.scorer = scorer

    def rowCount(self, parent=QModelIndex(), *args, **kwargs):
        return 0 if parent.isValid() else min(self._rows, self._max_rows)

    def columnCount(self, parent=QModelIndex(), *args, **kwargs):
        return 0 if parent.isValid() else self._columns

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return

        row = self.mapToSourceRows(index.row())
        col = index.column()

        try:
            value = self._data[row, col]
            if role == Qt.EditRole:
                return value
            if col >= self._columns - 2:
                value = self.domain[value]
        except IndexError:
            return
        if role == Qt.DecorationRole and isinstance(value, Variable):
            return gui.attributeIconDict[value]
        if role == Qt.DisplayRole:
            if isinstance(value, Number):
                absval = abs(value)
                strlen = len(str(int(absval)))
                value = '{:.{}{}}'.format(value,
                                          2 if absval < .001 else
                                          3 if strlen < 2 else
                                          1 if strlen < 5 else
                                          0 if strlen < 6 else
                                          3,
                                          'f' if (absval == 0 or
                                                  absval >= .001 and
                                                  strlen < 6)
                                          else 'e')
            return str(value)
        if role == Qt.TextAlignmentRole and isinstance(value, Number):
            return Qt.AlignRight | Qt.AlignVCenter
        if role == Qt.ToolTipRole:
            return str(value)

    def setHorizontalHeaderLabels(self, labels: Iterable):
        self._headers = labels

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal and section < len(self._headers):
                return self._headers[section]
            if orientation == Qt.Vertical:
                return section + 1
        return

    def initialize(self, data):
        self.beginResetModel()
        self._data = np.array(data)
        self._rows, self._columns = self._data.shape
        self.reset_sort()
        self.reset_filter()
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self._data = None
        self._rows = self._columns = 0
        self.reset_sort()
        self.reset_filter()
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
            n_data = min(max(n_data + n_rows, 2 * n_data), self.n_comb)
            ar = np.full((n_data, self._columns), np.nan)
            ar[:self._rows] = self._data[:self._rows]
            self._data = ar

        self._data[self._rows:self._rows + n_rows] = rows
        self._rows += n_rows

        if insert:
            self.endInsertRows()

        if self.__sortColumn >= 0:
            self.sort_new(n_rows)

        if isinstance(self.__filterInd, np.ndarray):
            self.reset_filter()

    def __len__(self):
        return self._rows

    def sort(self, column: int, order=Qt.DescendingOrder):
        if self._data is None:
            return
        indices = self._sort(column, order)
        self.__sortColumn = column
        self.__sortOrder = order
        if isinstance(self.__filterInd, np.ndarray):
            self._filterAndSort(indices)
        else:
            self._setSortIndices(indices)

    def _setSortIndices(self, indices):
        self.layoutAboutToBeChanged.emit([], QAbstractTableModel.VerticalSortHint)
        self.__sortInd = indices
        self.layoutChanged.emit([], QAbstractTableModel.VerticalSortHint)

    def _sort(self, column, order):
        if column < 0:
            return ...

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

    def reset_sort(self):
        return self.sort(-1)

    def set_filter(self, text):
        if self._data is None:
            return

        self.layoutAboutToBeChanged.emit([])
        if not text:
            self.__filterInd = ...
            self._max_rows = MAX_ROWS
        else:
            self.__filterAttrs = [i for i, name in enumerate(self.attr_names) if str(text).lower() in name]
            self.__filterInd = np.isin(self._data[:, -2:][self.__sortInd], self.__filterAttrs).any(axis=1).nonzero()[0]
            self._max_rows = len(self.__filterInd)
        self.layoutChanged.emit([])

    def reset_filter(self):
        return self.set_filter("")

    def _filterAndSort(self, indices):
        self.layoutAboutToBeChanged.emit([])
        self.__sortInd = indices
        self.__filterInd = np.isin(self._data[:, -2:][self.__sortInd], self.__filterAttrs).any(axis=1).nonzero()[0]
        self.layoutChanged.emit([])

    def mapToSourceRows(self, rows):
        if isinstance(rows, (int, type(Ellipsis))) or len(rows):
            if isinstance(self.__filterInd, np.ndarray):
                rows = self.__filterInd[rows]
            if isinstance(self.__sortInd, np.ndarray):
                rows = self.__sortInd[rows]
        return rows


class Widget(OWWidget, ConcurrentWidgetMixin):
    name = "npTestWidget"
    want_control_area = False

    class Inputs:
        data = Input("Data", Table)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.keep_running = True
        self.saved_state = None
        self.progress = 0

        self.data = None
        self.pp_data = None
        self.attrs = 0

        self.setLayout(QVBoxLayout())

        self.feature = self.feature_index = None
        self.feature_model = DomainModel(
            order=DomainModel.ATTRIBUTES, separators=False,
            placeholder="(All combinations)")
        gui.comboBox(
            self, self, "feature", callback=self.feature_combo_changed,
            model=self.feature_model, searchable=True
        )

        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter ...")
        self.filter.textChanged.connect(self.filter_changed)
        self.setFocus(Qt.ActiveWindowFocusReason)

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

        self.button = gui.button(self, self, "Start", callback=self.toggle)

        self.layout().addWidget(self.filter)
        self.layout().addWidget(view)
        self.layout().addWidget(self.button)

    def filter_changed(self, text):
        self.model.set_filter(text)

    def feature_combo_changed(self):
        self.feature_index = self.feature and self.pp_data.domain.index(self.feature)
        self.initialize()

    def initialize(self):
        if self.task is not None:
            self.keep_running = False
            self.cancel()
        self.keep_running = True
        self.saved_state = None
        self.progress = 0
        self.progressBarFinished()
        self.model.clear()
        self.button.setText("Start")
        self.button.setEnabled(True)

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.pp_data = Discretize()(self.data)
        self.score = InteractionScorer(self.pp_data)
        self.attrs = len(data.domain.attributes)
        self.model.set_domain(self.pp_data.domain)
        self.feature_model.set_domain(self.pp_data.domain)
        self.initialize()

    def toggle(self):
        self.keep_running = not self.keep_running
        if not self.keep_running:
            self.button.setText("Pause")
            self.button.repaint()
            self.progressBarInit()
            self.filter.setText("")
            self.filter.setEnabled(False)
            self.start(run, self.compute_score, self.row_for_state,
                       self.iterate_states, self.saved_state,
                       self.state_count(), self.progress)
        else:
            self.button.setText("Continue")
            self.button.repaint()
            self.cancel()
            self.progressBarFinished()
            self.filter.setEnabled(True)

    def on_partial_result(self, result):
        add_to_model, latest_state = result
        self.saved_state = latest_state
        self.model.append(add_to_model)
        self.progress = len(self.model)
        self.progressBarSet(self.progress * 100 // self.state_count())

    def on_done(self, result):
        self.button.setText("Finished")
        self.button.setEnabled(False)
        self.filter.setEnabled(True)

    def iterate_states(self, initial_state):
        if self.feature is not None:
            return self._iterate_by_feature(initial_state)
        return self._iterate_all(initial_state)

    def _iterate_all(self, initial_state):
        i0, j0 = initial_state or (0, 0)
        for i in range(i0, self.attrs):
            for j in range(j0, i):
                yield i, j
            j0 = 0

    def _iterate_by_feature(self, initial_state):
        _, j0 = initial_state or (0, 0)
        for j in range(j0, self.attrs):
            if j != self.feature_index:
                yield self.feature_index, j

    def state_count(self):
        return self.attrs if self.feature is not None else self.attrs * (self.attrs - 1) // 2

    def row_for_state(self, score, state):
        return [score] + list(state)

    def compute_score(self, state):
        return self.score(*state)


class ModelQueue:
    def __init__(self):
        self.lock = Lock()
        self.model = []
        self.state = None

    def put(self, row, state):
        with self.lock:
            self.model.append(row)
            self.state = state

    def get(self):
        with self.lock:
            model, self.model = self.model, []
            state, self.state = self.state, None
        return model, state


def run(compute_score: Callable, row_for_state: Callable,
        iterate_states: Callable, saved_state: Iterable,
        state_count: int, progress: int, task: TaskState):
    task.set_status("Getting combinations...")
    task.set_progress_value(0.1)
    states = iterate_states(saved_state)

    task.set_status("Getting scores...")
    queue = ModelQueue()
    can_set_partial_result = True

    def do_work(st, next_st):
        try:
            score = compute_score(st)
            if score is not None:
                queue.put(row_for_state(score, st), next_st)
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
                return queue.get()
            task.set_progress_value(progress * 100 // state_count)
            progress += 1
            state = copy.copy(next_state)
            next_state = copy.copy(next(states))
            do_work(state, next_state)
            if can_set_partial_result:
                task.set_partial_result(queue.get())
                can_set_partial_result = False
                Timer(0.05, reset_flag).start()
    except StopIteration:
        do_work(state, None)
        task.set_partial_result(queue.get())
    return queue.get()


if __name__ == "__main__":
    # WidgetPreview(Widget).run(Table("iris"))
    WidgetPreview(Widget).run(Table("aml-1k"))
