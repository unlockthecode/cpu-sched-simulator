"""
CPU Scheduling Algorithms Simulator — TUI Edition
by: Mennard Ezekiel M. Manlutac
Keyboard shortcuts:
    R          — Run simulation
    A          — Add process row
    D          — Remove last row
    C          — Clear all rows
    1          — Toggle Gantt chart panel
    2          — Toggle Results table panel
    Q / Ctrl+C — Quit
"""

# Standard library and Textual TUI framework imports; exits early if textual is missing
from __future__ import annotations
from turtle import pos
from typing import List, Tuple, Dict
import sys

try:
    from textual.app import App, ComposeResult
    from textual.widgets import Header, Footer, Input, Static, Label, Switch, Button
    from textual.containers import Horizontal, Vertical, ScrollableContainer
    from textual import on, events
    from textual.reactive import reactive
    from textual.css.query import NoMatches
    from rich.text import Text
except ImportError:
    print("\n[ERROR] textual is required. Install with:  pip install textual\n")
    sys.exit(1)

# Process count limits enforced during input validation
MIN_PROCESSES = 3
MAX_PROCESSES = 12

# Maps algorithm keys to their full display names shown in the UI
ALGORITHMS: Dict[str, str] = {
    "FCFS"    : "First-Come, First-Served (Non-preemptive)",
    "SJF"     : "Shortest Job First (Non-preemptive)",
    "SRT"     : "Shortest Remaining Time (Preemptive)",
    "RR"      : "Round Robin",
    "PRI-NP"  : "Priority — Non-preemptive",
    "PRI-PRE" : "Priority — Preemptive",
    "PRI-RR"  : "Priority + Round Robin",
}

# Color pool cycled per process for Gantt bar coloring; wraps around if >12 processes
_ANSI_COLORS = [
    "bright_red", "bright_yellow", "bright_green", "bright_cyan",
    "bright_blue", "bright_magenta", "orange1", "deep_pink2",
    "aquamarine1", "chartreuse1", "dodger_blue2", "light_coral",
]

# Data model for a single process; holds all input fields and computed scheduling results
class Process:
    __slots__ = ("pid","arrival","burst","priority","remaining",
                 "start_time","finish_time","waiting_time","turnaround_time")
    def __init__(self, pid, arrival, burst, priority=0):
        self.pid = pid; self.arrival = arrival; self.burst = burst
        self.priority = priority; self.remaining = burst
        self.start_time = -1; self.finish_time = 0
        self.waiting_time = 0; self.turnaround_time = 0

GanttItem = Tuple[str, int, int]

# Deep-copies a process list so algorithms never mutate the original input
def _clone(procs):
    return [Process(p.pid, p.arrival, p.burst, p.priority) for p in procs]

# Stamps completion time, turnaround, and waiting time onto a process once it finishes
def _finish(p, time):
    p.finish_time = time
    p.turnaround_time = time - p.arrival
    p.waiting_time = p.turnaround_time - p.burst

# FCFS — runs processes in arrival order; inserts IDLE gaps when CPU is free
def fcfs(processes):
    procs = sorted(_clone(processes), key=lambda p: (p.arrival, p.pid))
    gantt, time = [], 0
    for p in procs:
        if time < p.arrival:
            gantt.append(("IDLE", time, p.arrival)); time = p.arrival
        p.start_time = time
        gantt.append((p.pid, time, time + p.burst))
        time += p.burst; _finish(p, time)
    return procs, gantt

# SJF (non-preemptive) — picks the shortest burst among all currently arrived processes
def sjf(processes):
    procs = _clone(processes); gantt, time = [], 0; pool, done = list(procs), []
    while pool:
        ready = [p for p in pool if p.arrival <= time]
        if not ready:
            nxt = min(p.arrival for p in pool)
            gantt.append(("IDLE", time, nxt)); time = nxt; continue
        p = min(ready, key=lambda x: (x.burst, x.arrival, x.pid)); pool.remove(p)
        p.start_time = time
        gantt.append((p.pid, time, time + p.burst))
        time += p.burst; _finish(p, time); done.append(p)
    return done, gantt

# SRT (preemptive SJF) — re-evaluates at every future arrival and preempts if a shorter job arrives
def srt(processes):
    procs = _clone(processes); gantt, time, done = [], 0, []
    while len(done) < len(procs):
        ready = [p for p in procs if p.arrival <= time and p.remaining > 0]
        if not ready:
            nxt = min(p.arrival for p in procs if p.remaining > 0)
            gantt.append(("IDLE", time, nxt)); time = nxt; continue
        cur = min(ready, key=lambda x: (x.remaining, x.arrival, x.pid))
        if cur.start_time == -1: cur.start_time = time
        future = [p.arrival for p in procs if p.arrival > time and p.remaining > 0]
        run_for = min(cur.remaining, min(future) - time) if future else cur.remaining
        if gantt and gantt[-1][0] == cur.pid:
            gantt[-1] = (cur.pid, gantt[-1][1], time + run_for)
        else:
            gantt.append((cur.pid, time, time + run_for))
        cur.remaining -= run_for; time += run_for
        if cur.remaining == 0: _finish(cur, time); done.append(cur)
    return done, gantt

# Round Robin — gives each process a fixed time quantum in FIFO order, re-queuing unfinished ones
def round_robin(processes, quantum):
    procs = _clone(processes); gantt, time, done = [], 0, []
    by_arr = sorted(procs, key=lambda p: (p.arrival, p.pid))
    ready = []; admitted = set()
    def admit(t):
        for p in by_arr:
            if p.arrival <= t and p.pid not in admitted:
                ready.append(p); admitted.add(p.pid)
    admit(time)
    while len(done) < len(procs):
        if not ready:
            pending = [p for p in by_arr if p.pid not in admitted]
            if not pending: break
            nxt = min(p.arrival for p in pending)
            gantt.append(("IDLE", time, nxt)); time = nxt; admit(time); continue
        p = ready.pop(0)
        if p.start_time == -1: p.start_time = time
        run_for = min(quantum, p.remaining)
        if gantt and gantt[-1][0] == p.pid:
            gantt[-1] = (p.pid, gantt[-1][1], time + run_for)
        else:
            gantt.append((p.pid, time, time + run_for))
        p.remaining -= run_for; time += run_for; admit(time)
        if p.remaining == 0: _finish(p, time); done.append(p)
        else: ready.append(p)
    return done, gantt

# Priority non-preemptive — picks highest-priority ready process and runs it to completion
def priority_np(processes, higher_is_better=False):
    procs = _clone(processes); gantt, time, pool, done = [], 0, list(procs), []
    while pool:
        ready = [p for p in pool if p.arrival <= time]
        if not ready:
            nxt = min(p.arrival for p in pool)
            gantt.append(("IDLE", time, nxt)); time = nxt; continue
        key = (lambda x: (-x.priority, x.arrival, x.pid)) if higher_is_better \
              else (lambda x: (x.priority, x.arrival, x.pid))
        p = min(ready, key=key); pool.remove(p); p.start_time = time
        gantt.append((p.pid, time, time + p.burst))
        time += p.burst; _finish(p, time); done.append(p)
    return done, gantt

# Priority preemptive — re-evaluates priority at every arrival and preempts the running process if outranked
def priority_pre(processes, higher_is_better=False):
    procs = _clone(processes); gantt, time, done = [], 0, []
    key = (lambda x: (-x.priority, x.arrival)) if higher_is_better \
          else (lambda x: (x.priority, x.arrival))
    while len(done) < len(procs):
        ready = [p for p in procs if p.arrival <= time and p.remaining > 0]
        if not ready:
            nxt = min(p.arrival for p in procs if p.remaining > 0)
            gantt.append(("IDLE", time, nxt)); time = nxt; continue
        cur = min(ready, key=key)
        if cur.start_time == -1: cur.start_time = time
        future = [p.arrival for p in procs if p.arrival > time and p.remaining > 0]
        run_for = min(cur.remaining, min(future) - time) if future else cur.remaining
        if gantt and gantt[-1][0] == cur.pid:
            gantt[-1] = (cur.pid, gantt[-1][1], time + run_for)
        else:
            gantt.append((cur.pid, time, time + run_for))
        cur.remaining -= run_for; time += run_for
        if cur.remaining == 0: _finish(cur, time); done.append(cur)
    return done, gantt

# Priority + Round Robin — groups by priority, runs each group in round-robin order
def priority_rr(processes, quantum, higher_is_better=False):
    procs = _clone(processes); gantt, time, done = [], 0, []
    by_arr = sorted(procs, key=lambda p: (p.arrival, p.pid))
    ready = []; admitted = set()
    pri_key = (lambda p: -p.priority) if higher_is_better else (lambda p: p.priority)
    def admit(t):
        changed = False
        for p in by_arr:
            if p.arrival <= t and p.pid not in admitted and p.remaining > 0:
                ready.append(p); admitted.add(p.pid); changed = True
        if changed: ready.sort(key=pri_key)
    admit(time)
    while len(done) < len(procs):
        if not ready:
            pending = [p for p in by_arr if p.pid not in admitted and p.remaining > 0]
            if not pending: break
            nxt = min(p.arrival for p in pending)
            gantt.append(("IDLE", time, nxt)); time = nxt; admit(time); continue
        p = ready.pop(0)
        if p.start_time == -1: p.start_time = time
        run_for = min(quantum, p.remaining)
        if gantt and gantt[-1][0] == p.pid:
            gantt[-1] = (p.pid, gantt[-1][1], time + run_for)
        else:
            gantt.append((p.pid, time, time + run_for))
        p.remaining -= run_for; time += run_for; admit(time)
        if p.remaining == 0:
            _finish(p, time); done.append(p)
        else:
            same = [i for i, q in enumerate(ready) if pri_key(q) == pri_key(p)]
            if same:
                pos = same[-1] + 1
            else:
                lower = [i for i, q in enumerate(ready) if pri_key(q) > pri_key(p)]
                pos = lower[0] if lower else len(ready)
            ready.insert(pos, p)
    return done, gantt

# Computes average waiting time and average turnaround time across all finished processes
def compute_stats(processes):
    n = len(processes)
    return {
        "avg_wt" : sum(p.waiting_time    for p in processes) / n,
        "avg_tat": sum(p.turnaround_time for p in processes) / n,
    }

# Dispatcher — routes the selected algorithm key to the correct scheduling function
def run_algorithm(algo, processes, quantum, higher_is_better):
    if   algo == "FCFS"    : return fcfs(processes)
    elif algo == "SJF"     : return sjf(processes)
    elif algo == "SRT"     : return srt(processes)
    elif algo == "RR"      : return round_robin(processes, quantum)
    elif algo == "PRI-NP"  : return priority_np(processes, higher_is_better)
    elif algo == "PRI-PRE" : return priority_pre(processes, higher_is_better)
    elif algo == "PRI-RR"  : return priority_rr(processes, quantum, higher_is_better)
    else: raise ValueError(f"Unknown algorithm: {algo}")

# Returns the assigned color for a PID, falling back to green if not in the map
def _pid_color(pid, color_map):
    return color_map.get(pid, "bright_green")

# Builds the colored Gantt bar, time axis tick marks, and process legend as a Rich Text object
def render_gantt(gantt, color_map, algo, width=100):
    if not gantt:
        return Text("  No results yet. Press R to run.", style="#008F11")
    max_t = max(e for _, _, e in gantt)
    usable = max(width - 6, 30)
    scale = max(1, min(4, usable // max(max_t, 1)))
    out = Text()
    out.append(f"\n  Gantt Chart  --  {ALGORITHMS.get(algo, algo)}\n\n", style="bold #00FF41")
    out.append("  ")
    for pid, s, e in gantt:
        dur = e - s; w = max(dur * scale, 1)
        label = (pid if len(pid) <= w else pid[:w]).center(w)
        if pid == "IDLE":
            out.append(label, style="bold #444444 on #111111")
        else:
            out.append(label, style=f"bold black on {_pid_color(pid, color_map)}")
    out.append("\n")
    ticks = sorted({t for _, s, e in gantt for t in (s, e)})
    total_w = max_t * scale
    tick_row = [" "] * (total_w + 8)
    for t in ticks:
        pos = t * scale
        for i, ch in enumerate(str(t)):
            if pos + i < len(tick_row): tick_row[pos + i] = ch
    out.append("  " + "".join(tick_row) + "\n", style="#008F11")
    out.append("\n  Legend:  ", style="#008F11")
    seen = set()
    for pid, _, _ in gantt:
        if pid not in seen:
            seen.add(pid)
            if pid == "IDLE":
                out.append(f" {pid} ", style="bold #444444 on #111111")
            else:
                out.append(f" {pid} ", style=f"bold black on {_pid_color(pid, color_map)}")
            out.append("  ")
    out.append("\n")
    return out

# Builds the per-process results table (arrival, burst, wait, TAT) with averages row at the bottom
def render_table(done, color_map, algo, higher_is_better):
    if not done:
        return Text("  No results yet.", style="#008F11")
    out = Text()
    conv = "Higher # = Higher priority" if higher_is_better else "Lower # = Higher priority"
    out.append(f"\n  Results  --  {ALGORITHMS.get(algo, algo)}\n", style="bold #00FF41")
    out.append(f"  Priority: {conv}\n\n", style="#008F11")
    hdr = f"  {'PID':<6} {'Arrival':>7} {'Burst':>6} {'Pri':>4} {'Start':>6} {'Completion':>10} {'Wait':>6} {'TAT':>6}"
    sep = "  " + "-" * (len(hdr) - 2)
    out.append(hdr + "\n", style="bold #00FF41")
    out.append(sep + "\n", style="#003B00")
    for p in sorted(done, key=lambda x: x.pid):
        col = _pid_color(p.pid, color_map)
        st = str(p.start_time) if p.start_time != -1 else "-"
        out.append(
            f"  {p.pid:<6} {p.arrival:>7} {p.burst:>6} {p.priority:>4}"
            f" {st:>6} {p.finish_time:>7} {p.waiting_time:>6} {p.turnaround_time:>6}\n",
            style=col,
        )
    n = len(done)
    avg_wt = sum(p.waiting_time for p in done) / n
    avg_tat = sum(p.turnaround_time for p in done) / n
    out.append(sep + "\n", style="#003B00")
    out.append(
        f"  {'Avg':<6}                              "
        f"       {avg_wt:>6.2f} {avg_tat:>6.2f}\n",
        style="bold #00FF41",
    )
    return out

# Textual CSS — defines the full terminal theme for all widgets and layout
CSS = """
Screen { background: #000000; color: #00FF41; }
Header { background: #000000; color: #00FF41; text-style: bold; height: 1; }
Footer { background: #000000; color: #006600; height: 1; }
#title-area {
    background: #000000; height: 2; min-height: 2; max-height: 2;
    padding: 0; margin: 0; border-bottom: solid #003B00;
    align: center middle; content-align: center middle;
}
#title-area Static { margin: 0; padding: 0; height: 1; }
#app-subtitle { color: #008F11; height: 1; margin: 0; padding: 0; content-align: center middle; }
#toolbar {
    background: #000000; height: 3; min-height: 3; max-height: 3;
    margin: 0; padding: 0 1; border-bottom: solid #003B00; align: left middle;
}
#toolbar Label {
    color: #008F11; padding: 0 1; height: 3; min-height: 3;
    align: center middle; content-align: center middle;
}
#toolbar-spacer { width: 1fr; }
#status-label { display: none; }
#status-banner {
    color: #00FF41; width: 1fr; height: 3; min-height: 3;
    align: left middle; content-align: left middle; padding: 0 1;
}
#quantum-label, #pri-label {
    width: auto; height: 3; min-height: 3;
    align: center middle; content-align: center middle;
}
AlgoSelect {
    width: 19; height: 1; background: #001500;
    color: #00FF41; padding: 0; margin: 1 1;
}
AlgoSelect:focus { background: #002800; }
AlgoSelect:hover { background: #001a00; }
#quantum-input {
    width: 7; height: 1; background: #001500; color: #00FF41;
    border: none; margin: 1 1; padding: 0 1;
}
#quantum-input:focus { background: #002800; border: none; }
Switch {
    background: #000000; border: none; height: 1; margin: 1 1;
    align: center middle; content-align: center middle;
}
#pri-switch { width: 12; height: 1; align: center middle; content-align: center middle; }
Switch > .switch--slider { color: #003B00; }
Switch.-on > .switch--slider { color: #00AA00; }
#main-columns { height: 1fr; background: #000000; }
#left-panel {
    width: 50; background: #000000;
    border-right: solid #003B00; padding: 0 1;
}
#panel-title { color: #00FF41; text-style: bold; height: 1; padding: 1 0 0 0; }
#col-headers {
    height: 3; background: #000000; padding: 0;
    border-bottom: solid #002200; margin-top: 0; align: left middle;
}
#col-headers Static {
    width: 7; height: 3; margin: 0 1 0 0;
    content-align: center middle; color: #00FF41; text-style: bold;
}
#col-headers Static.header-dot { width: 3; margin-right: 1; }
#proc-scroll { height: 1fr; background: #000000; }
#right-panel { width: 1fr; background: #000000; }
#results-scroll { height: 1fr; background: #000000; }

/* Section toggle headers */
.section-header {
    height: 1;
    background: #001500;
    color: #00FF41;
    text-style: bold;
    border-bottom: solid #003B00;
    padding: 0 1;
    content-align: left middle;
}
.section-header.collapsed {
    color: #005500;
    background: #000800;
    border-bottom: solid #001500;
}

/* Output bodies — hidden-body hides them */
#gantt-body, #table-body {
    background: #000000;
    padding: 0 1;
    color: #00FF41;
}
#gantt-body.hidden-body, #table-body.hidden-body { display: none; }

#gantt-section, #table-section {
    background: #000000;
    border-bottom: solid #002200;
}
"""

# Single-line clickable/keyboard cycler widget for selecting the scheduling algorithm
class AlgoSelect(Static, can_focus=True):
    _ALGOS = ["FCFS", "SJF", "SRT", "RR", "PRI-NP", "PRI-PRE", "PRI-RR"]
    _LABELS = {
        "FCFS": "FCFS", "SJF": "SJF", "SRT": "SRT", "RR": "Round Robin",
        "PRI-NP": "Priority NP", "PRI-PRE": "Priority Pre", "PRI-RR": "Priority + RR",
    }
    idx: reactive[int] = reactive(0)
    # Renders the current algorithm label with left/right arrow indicators
    def render(self) -> Text:
        key = self._ALGOS[self.idx]; label = self._LABELS[key].ljust(14)
        t = Text()
        t.append(" < ", style="bold #007700")
        t.append(label, style="bold #00FF41")
        t.append("> ", style="bold #007700")
        return t

    @property
    def value(self) -> str:
        return self._ALGOS[self.idx]
    # Arrow/space keys step through the algorithm list
    def on_key(self, event: events.Key) -> None:
        if event.key in ("left", "up"):
            self.idx = (self.idx - 1) % len(self._ALGOS); event.stop()
        elif event.key in ("right", "down", "space"):
            self.idx = (self.idx + 1) % len(self._ALGOS); event.stop()

    def on_click(self) -> None:
        self.idx = (self.idx + 1) % len(self._ALGOS); self.focus()

# One editable input row in the process table: colored dot, PID, Arrival, Burst, Priority fields
class ProcessRow(Horizontal):
    DEFAULT_CSS = """
    ProcessRow {
        height: 3; background: #000000;
        border-bottom: solid #001200; align: left middle;
    }
    ProcessRow Static.dot {
        width: 3; height: 3; content-align: center middle;
        text-style: bold; margin-right: 1;
    }
    ProcessRow Input {
        height: 3; width: 7; background: #000000; color: #00FF41;
        border: tall #003B00; margin: 0 1 0 0; padding: 0 1;
    }
    ProcessRow Input:focus { border: tall #00FF41; background: #001200; }
    """
    def __init__(self, row_id, pid, arrival, burst, priority, color):
        super().__init__()
        self.row_id = row_id; self._pid = pid; self._arrival = arrival
        self._burst = burst; self._priority = priority; self.color = color

    def compose(self) -> ComposeResult:
        yield Static(f"[{self.color}]>[/]", classes="dot")
        yield Input(value=self._pid,           id=f"pid-{self.row_id}", placeholder="PID")
        yield Input(value=str(self._arrival),  id=f"arr-{self.row_id}", placeholder="Arr")
        yield Input(value=str(self._burst),    id=f"bst-{self.row_id}", placeholder="Bst")
        yield Input(value=str(self._priority), id=f"pri-{self.row_id}", placeholder="Pri")

# Main Textual application — owns the full layout, all state, and coordinates user actions with scheduling logic
class CPUSchedulerTUI(App):
    TITLE = "CPU Scheduling Simulator"
    CSS = CSS

    BINDINGS = [
        ("r",      "run_sim",      "Run [R]"),
        ("a",      "add_row",      "Add [A]"),
        ("d",      "remove_row",   "Del [D]"),
        ("c",      "clear_rows",   "Clear [C]"),
        ("1",      "toggle_gantt", "Gantt [1]"),
        ("2",      "toggle_table", "Table [2]"),
        ("q",      "quit",         "Quit [Q]"),
    ]

    def __init__(self):
        super().__init__()
        self._rows = []
        self._color_map = {}
        self._higher_is_better = False
        self._gantt_visible = True
        self._table_visible = True
    # Declares the full widget tree: header, toolbar, left input panel, right results panels, footer
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="title-area"):
            yield Static("by: Mennard Ezekiel M. Manlutac", id="app-subtitle")
        with Horizontal(id="toolbar"):
            yield Label("Scheduling Algorithm:")
            yield AlgoSelect(id="algo-select")
            yield Label("", id="status-label")
            yield Static("", id="status-banner")
            yield Label("", id="toolbar-spacer")
            yield Label("Quantum:", id="quantum-label")
            yield Input(value="2", id="quantum-input", placeholder="2")
            yield Label("Higher # = Better Priority:", id="pri-label")
            yield Switch(value=False, id="pri-switch")
        with Horizontal(id="main-columns"):
            with Vertical(id="left-panel"):
                yield Static(">> Process Input", id="panel-title")
                with Horizontal(id="col-headers"):
                    yield Static("   ", classes="header-dot")
                    yield Static("PID", classes="header-cell")
                    yield Static("Arr", classes="header-cell")
                    yield Static("Bst", classes="header-cell")
                    yield Static("Pri", classes="header-cell")
                with ScrollableContainer(id="proc-scroll"):
                    pass
            with Vertical(id="right-panel"):
                with ScrollableContainer(id="results-scroll"):
                    with Vertical(id="gantt-section"):
                        yield Static(
                            "[ 1 ] Gantt Chart  (press 1 to hide/show)",
                            classes="section-header", id="gantt-header",
                        )
                        yield Static(
                            "\n  [#006600]Run a simulation -- press R[/]",
                            id="gantt-body", expand=True,
                        )
                    with Vertical(id="table-section"):
                        yield Static(
                            "[ 2 ] Results Table  (press 2 to hide/show)",
                            classes="section-header", id="table-header",
                        )
                        yield Static(
                            "\n  [#006600]Run a simulation to see results.[/]",
                            id="table-body", expand=True,
                        )
        yield Footer()
    # Seeds the process table with three default processes on startup
    def on_mount(self) -> None:
        for pid, arr, bst, pri in [("P1",0,6,3), ("P2",2,4,1), ("P3",4,8,2)]:
            self._add_row(pid, arr, bst, pri)
    # Picks a color from the pool by row index; wraps around using modulo
    def _color_for(self, idx):
        return _ANSI_COLORS[idx % len(_ANSI_COLORS)]
    # Updates the status banner text in the toolbar
    def _set_status(self, msg):
        self.query_one("#status-banner", Static).update(msg)
    # Creates and mounts a new ProcessRow; enforces the MAX_PROCESSES cap
    def _add_row(self, pid="", arrival=0, burst=5, priority=1):
        if len(self._rows) >= MAX_PROCESSES:
            self._set_status(f"[red]Max {MAX_PROCESSES} processes.[/]"); return
        idx = len(self._rows); color = self._color_for(idx)
        pid = pid or f"P{idx + 1}"
        row = ProcessRow(idx, pid, arrival, burst, priority, color)
        self._rows.append(row)
        scroll = self.query_one("#proc-scroll", ScrollableContainer)
        scroll.mount(row); scroll.scroll_end(animate=False)
    # Reads all input fields from the process table, validates them, and returns a list of Process objects
    def _parse_processes(self):
        processes = []; self._color_map = {}; seen = set()
        for i, row in enumerate(self._rows):
            try:
                pid_w = self.query_one(f"#pid-{row.row_id}", Input)
                arr_w = self.query_one(f"#arr-{row.row_id}", Input)
                bst_w = self.query_one(f"#bst-{row.row_id}", Input)
                pri_w = self.query_one(f"#pri-{row.row_id}", Input)
            except NoMatches:
                continue
            pid = pid_w.value.strip()
            if not pid: raise ValueError(f"Row {i+1}: PID is empty.")
            if pid in seen: raise ValueError(f"Duplicate PID '{pid}'.")
            seen.add(pid)
            try:
                arr = int(arr_w.value); bst = int(bst_w.value); pri = int(pri_w.value)
            except ValueError:
                raise ValueError(f"{pid}: fields must be integers.")
            if arr < 0: raise ValueError(f"{pid}: Arrival >= 0.")
            if bst <= 0: raise ValueError(f"{pid}: Burst > 0.")
            processes.append(Process(pid, arr, bst, pri))
            self._color_map[pid] = row.color
        return processes
    # Syncs section header labels and body visibility with the current gantt/table toggle state
    def _refresh_toggles(self):
        gh = self.query_one("#gantt-header", Static)
        th = self.query_one("#table-header", Static)
        gb = self.query_one("#gantt-body",   Static)
        tb = self.query_one("#table-body",   Static)
        if self._gantt_visible:
            gh.remove_class("collapsed")
            gb.remove_class("hidden-body")
            gh.update("[ 1 ] Gantt Chart  (press 1 to hide)")
        else:
            gh.add_class("collapsed")
            gb.add_class("hidden-body")
            gh.update("[ 1 ] Gantt Chart  (hidden — press 1 to show)")
        if self._table_visible:
            th.remove_class("collapsed")
            tb.remove_class("hidden-body")
            th.update("[ 2 ] Results Table  (press 2 to hide)")
        else:
            th.add_class("collapsed")
            tb.add_class("hidden-body")
            th.update("[ 2 ] Results Table  (hidden — press 2 to show)")

    # Keyboard action handlers — wired to BINDINGS above; delegate to core methods
    def action_run_sim(self)    -> None: self.run_simulation()
    def action_add_row(self)    -> None: self._add_row()
    def action_remove_row(self) -> None:
        if len(self._rows) <= 1:
            self._set_status("[red]Keep at least one row.[/]"); return
        self._rows.pop().remove()
    def action_clear_rows(self) -> None:
        for r in self._rows: r.remove()
        self._rows.clear()
    def action_toggle_gantt(self) -> None:
        self._gantt_visible = not self._gantt_visible
        self._refresh_toggles()
    def action_toggle_table(self) -> None:
        self._table_visible = not self._table_visible
        self._refresh_toggles()

    # Listens for the priority convention toggle switch and updates the app state flag
    @on(Switch.Changed, "#pri-switch")
    def _on_switch(self, event: Switch.Changed) -> None:
        self._higher_is_better = event.value
    # Core simulation entry point: validates input, runs the chosen algorithm, renders results, resets panel visibility
    def run_simulation(self) -> None:
        if len(self._rows) < MIN_PROCESSES:
            self._set_status(
                f"[red]Need >= {MIN_PROCESSES} processes (have {len(self._rows)}).[/]"
            ); return
        try:
            processes = self._parse_processes()
        except ValueError as e:
            self._set_status(f"[red]{e}[/]"); return
        try:
            q = int(self.query_one("#quantum-input", Input).value)
            if q < 1: raise ValueError()
        except ValueError:
            self._set_status("[red]Quantum must be a positive integer.[/]"); return
        algo = self.query_one("#algo-select", AlgoSelect).value
        higher = self._higher_is_better
        try:
            done, gantt = run_algorithm(algo, processes, q, higher)
        except Exception as e:
            self._set_status(f"[red]Error: {e}[/]"); return
        stats = compute_stats(done)
        self._set_status(
            f"[#00FF41]OK | {algo} | Avg WT={stats['avg_wt']:.2f}"
            f" | Avg TAT={stats['avg_tat']:.2f}[/]"
        )
        w = max(self.size.width - 44, 40)
        self.query_one("#gantt-body", Static).update(
            render_gantt(gantt, self._color_map, algo, w))
        self.query_one("#table-body", Static).update(
            render_table(done, self._color_map, algo, higher))
        self._gantt_visible = True
        self._table_visible = True
        self._refresh_toggles()
        self.query_one("#results-scroll", ScrollableContainer).scroll_home(animate=False)

# Entry point — instantiates and runs the Textual app
def main():
    CPUSchedulerTUI().run()

if __name__ == "__main__":
    main()
