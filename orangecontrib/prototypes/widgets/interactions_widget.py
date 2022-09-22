from AnyQt.QtCore import QModelIndex, Qt, QLineF
from AnyQt.QtWidgets import QTableView, QVBoxLayout, QHeaderView, QLineEdit, \
    QStyleOptionViewItem, QApplication, QStyle
from AnyQt.QtGui import QColor, QPainter, QPen

from Orange.data import Table
from Orange.preprocess import Discretize
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, AttributeList
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.settings import Setting, ContextSetting

from orangecontrib.prototypes.widgets.nptablemodel import RankModel, run
from orangecontrib.prototypes.interactions import InteractionScorer, HeuristicType, Heuristic


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
            h = model.scorer.class_entropy
            l_bar, r_bar = model.scorer.information_gain[int(attr1)], model.scorer.information_gain[int(attr2)]
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
            painter.restore()
            textrect.adjust(0, 0, 0, -textoffset)

        opt.text = text
        self.drawViewItemText(style, painter, opt, textrect)


class InteractionWidget(OWWidget, ConcurrentWidgetMixin):
    name = "Interaction Rank"
    want_control_area = False

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        features = Output("Features", AttributeList)

    selection = ContextSetting([])
    heuristic_type = Setting(0)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.keep_running = True
        self.saved_state = None
        self.progress = 0

        self.data = None  # type: Table
        self.pp_data = None  # type: Table
        self.n_attrs = 0

        self.score = None
        self.heuristic = None

        self.setLayout(QVBoxLayout())

        self.heuristic_combo = gui.comboBox(
            self, self, "heuristic_type", items=HeuristicType.items(),
            callback=self.on_heuristic_combo_changed,
        )

        self.feature = self.feature_index = None
        self.feature_model = DomainModel(
            order=DomainModel.ATTRIBUTES, separators=False,
            placeholder="(All combinations)")
        feature_combo = gui.comboBox(
            self, self, "feature", callback=self.on_feature_combo_changed,
            model=self.feature_model, searchable=True
        )

        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter ...")
        self.filter.textChanged.connect(self.on_filter_changed)
        self.setFocus(Qt.ActiveWindowFocusReason)

        self.model = RankModel()
        self.model.setHorizontalHeaderLabels((
            "Interaction", "Information Gain", "Feature 1", "Feature 2"
        ))
        view = QTableView(selectionBehavior=QTableView.SelectRows,
                          selectionMode=QTableView.SingleSelection,
                          showGrid=False,
                          editTriggers=gui.TableView.NoEditTriggers)
        view.setSortingEnabled(True)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        view.setItemDelegate(InteractionItemDelegate())
        view.setModel(self.model)
        view.selectionModel().selectionChanged.connect(self.on_selection_changed)

        self.button = gui.button(self, self, "Start", callback=self.toggle)

        self.layout().addWidget(feature_combo)
        self.layout().addWidget(self.filter)
        self.layout().addWidget(view)
        self.layout().addWidget(self.button)

    @Inputs.data
    def set_data(self, data):
        self.selection = []
        self.pp_data = self.data = data
        if data is not None:
            if any(attr.is_continuous for attr in self.data.domain):
                self.pp_data = Discretize()(self.data)
            self.n_attrs = len(data.domain.attributes)
            self.model.set_domain(self.pp_data.domain)
            self.feature_model.set_domain(self.pp_data.domain)
            self.score = InteractionScorer(self.pp_data)
            self.model.set_scorer(self.score)
            self.heuristic = Heuristic(self.score.information_gain, self.heuristic_type)
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

    def commit(self):
        if self.data is None:
            self.Outputs.features.send(None)
            return

        self.Outputs.features.send(AttributeList(
            [self.data.domain[attr] for attr in self.selection]))

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

    def on_selection_changed(self, selected):
        self.selection = [self.model.data(ind) for ind in selected.indexes()[-2:]]
        self.commit()

    def on_filter_changed(self, text):
        self.model.set_filter(text)

    def on_feature_combo_changed(self):
        self.feature_index = self.feature and self.data.domain.index(self.feature)
        self.initialize()

    def on_heuristic_combo_changed(self):
        self.heuristic = Heuristic(self.score.information_gain, self.heuristic_type)
        self.initialize()

    def compute_score(self, state):
        scores = (self.score(*state),
                  self.score.information_gain[state[0]],
                  self.score.information_gain[state[1]])
        return tuple(self.score.normalize(score) for score in scores)

    @staticmethod
    def row_for_state(score, state):
        return [score[0], sum(score)] + list(state)

    def iterate_states(self, initial_state):
        if self.feature is not None:
            return self._iterate_by_feature(initial_state)
        if self.heuristic is not None:
            return self.heuristic.get_states(initial_state)
        return self._iterate_all(initial_state)

    def _iterate_all(self, initial_state):
        i0, j0 = initial_state or (0, 0)
        for i in range(i0, self.n_attrs):
            for j in range(j0, i):
                yield i, j
            j0 = 0

    def _iterate_by_feature(self, initial_state):
        _, j0 = initial_state or (0, 0)
        for j in range(j0, self.n_attrs):
            if j != self.feature_index:
                yield self.feature_index, j

    def state_count(self):
        if self.feature is None:
            return self.n_attrs * (self.n_attrs - 1) // 2
        return self.n_attrs

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


if __name__ == "__main__":  # pragma: no cover
    # WidgetPreview(InteractionWidget).run(Table("iris"))
    WidgetPreview(InteractionWidget).run(Table("aml-1k"))
