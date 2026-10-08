"""Abstract TUI: a constellation of labs showing who is *here* with the message, and who is not."""

from __future__ import annotations

import asyncio
import random

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Input, RichLog, Static

from .backends import Backend
from .circuit import Circuit
from .nodes import Lab
from .router import Event, Relay, TelephoneRouter

ROLE_TAG = {"guard": "GUARD", "middle": "MIDDLE", "exit": "EXIT", None: ""}


class NodeCard(Static):
    """One lab. States: away (not in circuit) / standby / here / echo (persona inside Z)."""

    state = reactive("away")
    role = reactive(None)
    pulse = reactive(False)

    def __init__(self, lab: Lab):
        super().__init__(id=f"node-{lab.key}", classes="node")
        self.lab = lab

    def render(self) -> Text:
        glyph, note, style = {
            "away": ("·", "not here", "grey35"),
            "standby": ("○", "on route", "grey70"),
            "here": ("◉" if self.pulse else "●", "HERE", "bold bright_green"),
            "echo": ("◌" if self.pulse else "◎", "echo in Z", "bold magenta"),
        }[self.state]
        t = Text(justify="center")
        t.append(f"{glyph}\n", style=f"{style}")
        t.append(f"{self.lab.name}\n", style="bold" if self.state != "away" else "grey35 strike")
        t.append(f"{ROLE_TAG[self.role] or '—'}\n", style="cyan" if self.role else "grey35")
        t.append(note, style=style)
        return t

    def watch_state(self, state: str) -> None:
        self.set_class(state in ("here", "echo"), "lit")
        self.set_class(state == "away", "away")


class ZCore(Static):
    state = reactive("idle")
    revealed = reactive(False)
    pulse = reactive(False)

    def __init__(self, z: Lab):
        super().__init__(id="zcore")
        self.z = z

    def render(self) -> Text:
        busy = self.state == "here"
        t = Text(justify="center")
        bar = ("▓▒░" if self.pulse else "░▒▓") * 4 if busy else "░" * 12
        t.append(f"{bar}\n", style="bold yellow" if busy else "grey35")
        t.append("Z  ", style="bold yellow")
        t.append(self.z.name if self.revealed else "identity withheld",
                 style="bold red" if self.revealed else "grey62 italic")
        t.append("\nHERE, answering" if busy else "\nwaiting", style="yellow" if busy else "grey50")
        return t


class RouteStrip(Static):
    STOPS = 9  # you G M E Z E M G you
    pos = reactive(-1)
    labels: reactive[list[str]] = reactive(list)

    def render(self) -> Text:
        names = self.labels or ["you", "?", "?", "?", "Z", "?", "?", "?", "you"]
        t = Text(justify="center")
        for i, n in enumerate(names):
            if i:
                t.append(" ─▶ " if i <= 4 else " ◀─ ", style="grey42")
            style = "reverse bold bright_green" if i == self.pos else (
                "white" if 0 <= self.pos and i < self.pos else "grey50")
            t.append(f" {n} ", style=style)
        return t


class TelephoneApp(App):
    TITLE = "telephone · onion"
    CSS = """
    Screen { background: #0b0d10; }
    #constellation { height: 8; padding: 0 1; }
    .node { width: 1fr; height: 7; border: round #2a2f36; content-align: center middle; margin: 0 1; }
    .node.lit { border: heavy #39d98a; }
    .node.away { border: dashed #1c2026; }
    #zcore { height: 5; border: double #5c4a12; content-align: center middle; margin: 0 2; }
    #route { height: 3; content-align: center middle; }
    #panes { height: 1fr; }
    #log { width: 1fr; border: round #2a2f36; }
    #answer { width: 1fr; border: round #2a2f36; padding: 0 1; }
    #prompt { margin: 0 1; }
    """
    BINDINGS = [
        Binding("ctrl+r", "reveal", "reveal Z"),
        Binding("ctrl+n", "new_z", "new random Z"),
        Binding("ctrl+t", "toggle_relay", "relay mode"),
        Binding("ctrl+q", "quit", "quit"),
    ]

    def __init__(self, backend: Backend, labs: tuple[Lab, ...], z: Lab, relay: Relay):
        super().__init__()
        self.backend, self.labs, self.z, self.relay = backend, labs, z, relay
        self.pace = 0.35

    def compose(self) -> ComposeResult:
        with Horizontal(id="constellation"):
            for lab in self.labs:
                yield NodeCard(lab)
        yield ZCore(self.z)
        yield RouteStrip(id="route")
        with Horizontal(id="panes"):
            yield RichLog(id="log", markup=True, wrap=True, min_width=20)
            with Vertical(id="answer"):
                yield Static("[grey50]answers arrive here, origin unknown[/]", id="answer-text")
        yield Input(placeholder="ask anything, enter to route it through a random circuit", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(0.4, self._tick)
        self._status()

    def _tick(self) -> None:
        for w in (*self.query(NodeCard), self.query_one(ZCore)):
            w.pulse = not w.pulse

    def _status(self) -> None:
        self.sub_title = f"backend={self.backend.name} · relay={self.relay}"

    def _log(self, msg: str) -> None:
        self.query_one("#log", RichLog).write(msg)

    # ------------------------------------------------------------------ actions
    def action_reveal(self) -> None:
        zc = self.query_one(ZCore)
        zc.revealed = not zc.revealed

    def action_new_z(self) -> None:
        self.z = random.SystemRandom().choice(self.labs)
        zc = self.query_one(ZCore)
        zc.z, zc.revealed = self.z, False
        zc.refresh()
        self._log("[yellow]Z re-drawn, identity withheld[/]")

    def action_toggle_relay(self) -> None:
        self.relay = "nodes" if self.relay == "z" else "z"
        self._status()
        self._log(f"[cyan]relay mode → {self.relay}[/]")

    async def on_input_submitted(self, ev: Input.Submitted) -> None:
        if ev.value.strip():
            self.route(ev.value.strip())
            ev.input.value = ""

    # ------------------------------------------------------------------ routing
    @work(exclusive=True)
    async def route(self, prompt: str) -> None:
        router = TelephoneRouter(self.backend, self.labs, self.z, self.relay)
        self.query_one("#answer-text", Static).update("[grey50]…in transit[/]")
        try:
            await router.ask(prompt, self._on_event)
        except Exception as e:  # surface backend errors without killing the TUI
            self._log(f"[red]error: {e}[/]")
            self._reset_nodes()

    def _reset_nodes(self) -> None:
        for card in self.query(NodeCard):
            card.state, card.role = "away", None
        self.query_one(ZCore).state = "idle"

    async def _on_event(self, ev: Event) -> None:
        cards = {c.lab.key: c for c in self.query(NodeCard)}
        strip = self.query_one(RouteStrip)
        zc = self.query_one(ZCore)

        if ev.kind == "circuit":
            c: Circuit = ev.circuit
            for card in cards.values():
                card.role = c.role_of(card.lab)
                card.state = "standby" if card.role else "away"
            g, m, e = (h.name.split()[0] for h in c.hops)
            strip.labels = ["you", g, m, e, "Z", e, m, g, "you"]
            strip.pos = 0
            absent = [l.name for l in self.labs if not c.role_of(l)]
            self._log(f"[bold]circuit[/] {g} → {m} → {e}   [grey50]absent: {', '.join(absent)}[/]")
            await asyncio.sleep(self.pace)
            return

        if ev.kind == "output":
            strip.pos = 8
            self.query_one("#answer-text", Static).update(ev.text)
            self._log("[bright_green]delivered[/] origin: [italic]unknown[/]")
            await asyncio.sleep(self.pace)
            for card in cards.values():
                card.state = "standby" if card.role else "away"
            return

        order = [h.key for h in self._circuit_hops()]
        if ev.node == "Z":
            if ev.kind == "enter":
                strip.pos = 4
                zc.state = "here"
                self._log(f"[yellow]Z[/] {ev.phase}: called with nested prompt "
                          f"[grey50]({len(ev.text)} chars)[/]")
            else:
                zc.state = "idle"
            return

        card = cards[ev.node]
        if ev.kind == "enter":
            card.state = "echo" if ev.virtual else "here"
            idx = order.index(ev.node)
            strip.pos = 1 + idx if ev.phase == "forward" else 7 - idx
            how = "persona inside Z" if ev.virtual else "real call, restating"
            self._log(f"  {ev.phase:7} [bold]{card.lab.name}[/] here [grey50]({how})[/]")
            await asyncio.sleep(self.pace)
        else:
            card.state = "standby"

    def _circuit_hops(self) -> list[Lab]:
        by_role = {c.role: c.lab for c in self.query(NodeCard) if c.role}
        return [by_role["guard"], by_role["middle"], by_role["exit"]]
