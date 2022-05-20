from queue import Queue, Empty
from types import SimpleNamespace
from typing import Optional, Iterable, Callable
from threading import Timer
import copy

from AnyQt.QtCore import QModelIndex, Qt, QLineF, QSortFilterProxyModel, pyqtSignal as Signal
from AnyQt.QtWidgets import QTableView, QVBoxLayout, QHeaderView, QDialog, QLineEdit, \
    QStyleOptionViewItem, QApplication, QStyle
from AnyQt.QtGui import QColor, QPainter, QPen

from Orange.data import Table
from Orange.preprocess import Discretize
from Orange.widgets import gui
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.visualize.utils import VizRankDialogAttrPair
from Orange.widgets.utils.concurrent import ConcurrentMixin, TaskState
from Orange.widgets.utils.messages import WidgetMessagesMixin
from Orange.widgets.widget import Msg

from orangecontrib.prototypes.widgets.nptablemodel import RankModel
from orangecontrib.prototypes.interactions import Interaction


class InteractionItemDelegate(gui.TableBarItem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.r = QColor("#ffaa7f")
        self.g = QColor("#aaf22b")
        self.b = QColor("#46befa")
        self.__line = QLineF()
        self.__pen = QPen(self.b, 5, Qt.SolidLine, Qt.RoundCap)

    def paint(
            self, painter: QPainter, option: QStyleOptionViewItem,
            index: QModelIndex
    ) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = option.widget
        style = QApplication.style() if widget is None else widget.style()
        pen = self.__pen
        line = self.__line
        self.__style = style
        text = opt.text
        opt.text = ""
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)
        textrect = style.subElementRect(
            QStyle.SE_ItemViewItemText, opt, widget)

        # interaction is None for attribute items ->
        # only draw bars for first column
        interaction = self.cachedData(index, Qt.EditRole)
        if index.column() == 0 and interaction is not None:
            rect = option.rect
            pw = self.penWidth
            textoffset = pw + 2
            baseline = rect.bottom() - textoffset / 2
            origin = rect.left() + 3 + pw / 2  # + half pen width for the round line cap
            width = rect.width() - 3 - pw

            def draw_line(start, length):
                line.setLine(origin + start, baseline, origin + start + length, baseline)
                painter.drawLine(line)

            # negative information gains stem from issues in interaction calculation
            # may cause bars reaching out of intended area
            model = index.model()
            attr1 = model.data(index.siblingAtColumn(2), Qt.EditRole)
            attr2 = model.data(index.siblingAtColumn(3), Qt.EditRole)
            h = model.scorer.class_h
            l_bar, r_bar = model.scorer.gains[int(attr1)], model.scorer.gains[int(attr2)]
            l_bar, r_bar = width * max(l_bar, 0) / h, width * max(r_bar, 0) / h
            interaction *= width

            pen.setColor(self.b)
            pen.setWidth(pw)
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(pen)
            draw_line(0, l_bar)
            draw_line(l_bar + interaction, r_bar)
            pen.setColor(self.g if interaction >= 0 else self.r)
            painter.setPen(pen)
            draw_line(l_bar, interaction)
            # draw_line(0, interaction)
            painter.restore()
            textrect.adjust(0, 0, 0, -textoffset)

        opt.text = text
        self.drawViewItemText(style, painter, opt, textrect)


class InteractionVizRank(VizRankDialogAttrPair):
    captionTitle = ""

    processingStateChanged = Signal(int)
    progressBarValueChanged = Signal(float)
    messageActivated = Signal(Msg)
    messageDeactivated = Signal(Msg)
    selectionChanged = Signal(object)

    def __init__(self, master):
        """Initialize the attributes and set up the interface"""
        QDialog.__init__(self, master, windowTitle=self.captionTitle)
        WidgetMessagesMixin.__init__(self)
        ConcurrentMixin.__init__(self)
        self.setLayout(QVBoxLayout())

        self.insert_message_bar()
        self.layout().insertWidget(0, self.message_bar)
        self.master = master

        self.keep_running = False
        self.scheduled_call = None
        self.saved_state = None
        self.saved_progress = 0

        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter ...")
        self.filter.textChanged.connect(self.filter_changed)
        self.layout().addWidget(self.filter)
        # Remove focus from line edit
        self.setFocus(Qt.ActiveWindowFocusReason)

        self.rank_model = RankModel(self)
        self.rank_model.setHorizontalHeaderLabels([
            "Score 1", "Score 2", "Feature 1", "Feature 2"
        ])
        self.model_proxy = QSortFilterProxyModel(
            self, filterCaseSensitivity=Qt.CaseInsensitive
        )
        self.model_proxy.setSourceModel(self.rank_model)
        self.rank_table = view = QTableView(
            selectionBehavior=QTableView.SelectRows,
            selectionMode=QTableView.SingleSelection,
            showGrid=False,
            editTriggers=gui.TableView.NoEditTriggers)
        view.setItemDelegate(InteractionItemDelegate())
        view.setModel(self.rank_model)
        view.selectionModel().selectionChanged.connect(
            self.on_selection_changed)
        view.setSortingEnabled(True)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout().addWidget(view)

        self.button = gui.button(self, self, "Start", callback=self.toggle, default=True)

        self.attrs = []
        manual_change_signal = getattr(master, "xy_changed_manually", None)
        if manual_change_signal:
            manual_change_signal.connect(self.on_manual_change)

        self.heuristic = None
        self.use_heuristic = False
        self.sel_feature_index = None

    def initialize(self):
        if self.task is not None:
            self.keep_running = False
            self.cancel()
        self.keep_running = False
        self.scheduled_call = None
        self.saved_state = None
        self.saved_progress = 0
        self.progressBarFinished()
        self.rank_model.clear()
        self.button.setText("Start")
        self.button.setEnabled(self.check_preconditions())

        data = self.master.data
        self.attrs = data and data.domain.attributes
        self.heuristic = None
        self.use_heuristic = False
        self.sel_feature_index = None  # self.master.feature or data.domain.index(self.master.feature)
        if data:
            self.interaction = Interaction(self.master.data)
            self.rank_model.set_domain(self.master.data)
            self.rank_model.set_scorer(self.interaction)

    def compute_score(self, state):
        attr1, attr2 = state
        h = self.interaction.class_h
        score = self.interaction(attr1, attr2) / h
        gain1 = self.interaction.gains[attr1] / h
        gain2 = self.interaction.gains[attr2] / h
        return score, gain1, gain2

    def row_for_state(self, score, state):
        return [score[0], sum(score)] + list(state)

    def check_preconditions(self):
        return self.master.data is not None

    def on_selection_changed(self, selected, deselected):
        pass

    def on_manual_change(self, attr1, attr2):
        pass

    def iterate_states(self, initial_state):
        if self.sel_feature_index is not None:
            return self.iterate_states_by_feature(initial_state)
        elif self.use_heuristic:
            return self.heuristic.get_states(initial_state)
        else:
            return self.iterate_all_states(initial_state)

    def iterate_states_by_feature(self, initial_state):
        _, sj = initial_state or (0, 0)
        for j in range(sj, len(self.attrs)):
            if j != self.sel_feature_index:
                yield self.sel_feature_index, j

    def iterate_all_states(self, initial_state):
        si, sj = initial_state or (0, 0)
        for i in range(si, len(self.attrs)):
            for j in range(sj, i):
                yield i, j
            sj = 0

    def state_count(self):
        n = len(self.attrs)
        return n * (n - 1) / 2 if self.sel_feature_index is None else n - 1

    def on_partial_result(self, result: Queue):
        rows = []
        try:
            while True:
                queued = result.get_nowait()
                self.saved_state = queued.next_state
                row = self.row_for_state(queued.score, queued.state)
                rows.append(row)
        except Empty:
            if rows:
                self.rank_model.append(rows)

        self.saved_progress = len(self.rank_model)
        self._update_progress()

    def _update_model(self):
        pass

    def toggle(self):
        self.keep_running = not self.keep_running
        if self.keep_running:
            self.button.setText("Pause")
            self.button.repaint()
            self.progressBarInit()
            self.before_running()
            self.start(run_vizrank, self.compute_score,
                       self.iterate_states, self.saved_state,
                       self.saved_progress, self.state_count())
        else:
            self.button.setText("Continue")
            self.button.repaint()
            self.cancel()
            self._stopped()

    def _connect_signals(self, state):
        super()._connect_signals(state)
        state.progress_changed.connect(self.master.progressBarSet)
        state.status_changed.connect(self.master.setStatusMessage)

    def _disconnect_signals(self, state):
        super()._disconnect_signals(state)
        state.progress_changed.disconnect(self.master.progressBarSet)
        state.status_changed.disconnect(self.master.setStatusMessage)

    def _on_task_done(self, future):
        super()._on_task_done(future)
        self.__set_state_ready()

    def __set_state_ready(self):
        self._set_empty_status()
        self.master.setBlocking(False)

    def __set_state_busy(self):
        self.master.progressBarInit()
        self.master.setBlocking(True)

    def _set_empty_status(self):
        self.master.progressBarFinished()
        self.master.setStatusMessage("")


def run_vizrank(compute_score: Callable, iterate_states: Callable,
                saved_state: Optional[Iterable], progress: int,
                state_count: int, task: TaskState):
    task.set_status("Getting combinations...")
    task.set_progress_value(0.1)
    states = iterate_states(saved_state)

    task.set_status("Getting scores...")
    queue = Queue()
    can_set_partial_result = True

    def do_work(st, next_st):
        try:
            score = compute_score(st)
            if score is not None:
                queue.put_nowait(Score(score=score, state=st, next_state=next_st))
        except Exception:  # ignore current state in case of any problem
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
            task.set_progress_value(progress * 100 // max(1, state_count))
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


class Score(SimpleNamespace):
    score = None  # type: Iterable
    state = None  # type: Iterable
    next_state = None  # type: Iterable


class InteractionWidget(OWWidget):
    name = "Interaction Rank"
    want_main_area = False
    want_control_area = True

    feature = None  # ContextSetting(None)

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    def __init__(self):
        OWWidget.__init__(self)

        self.data = None  # type: Table
        self.attr_color = None

        box = gui.vBox(self.controlArea)
        self.vizrank, _ = InteractionVizRank.add_vizrank(
            None, self, None, self._vizrank_selection_changed)

        box.layout().addWidget(self.vizrank.filter)
        box.layout().addWidget(self.vizrank.rank_table)
        box.layout().addWidget(self.vizrank.button)

    @Inputs.data
    def set_data(self, data):
        self.data = Discretize()(data)
        self.selection = []
        self.apply()
        self.vizrank.button.setEnabled(data is not None)

    def _vizrank_selection_changed(self, *args):
        self.selection = list(args)
        self.commit()

    def apply(self):
        self.vizrank.initialize()
        if self.data is not None:
            self.vizrank.toggle()

    def commit(self):
        pass


if __name__ == "__main__":  # pragma: no cover
    # WidgetPreview(InteractionWidget).run(Table("iris"))
    WidgetPreview(InteractionWidget).run(Table("/Users/noah/Nextcloud/Fri/tables/mushrooms.tab"))
