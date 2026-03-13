"""
Rich Terminal Dashboard — Module 13.
Live-updating terminal UI showing portfolio, signals, whale alerts, and stats.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from core.portfolio import PortfolioState
from strategies.signals import TradeSignal
from tracking.whale_tracker import WhaleAlert
from utils.helpers import format_duration, fmt_pct, fmt_usdc, now_ts

logger = logging.getLogger(__name__)


def _color_pnl(value: float) -> str:
    if value > 0:
        return f"[green]+{fmt_usdc(value)}[/green]"
    elif value < 0:
        return f"[red]{fmt_usdc(value)}[/red]"
    return f"[white]{fmt_usdc(value)}[/white]"


def _color_pct(value: float) -> str:
    if value > 0:
        return f"[green]+{value:.2f}%[/green]"
    elif value < 0:
        return f"[red]{value:.2f}%[/red]"
    return f"[white]{value:.2f}%[/white]"


def _signal_color(score: float) -> str:
    if score >= 8.0:
        return "green"
    elif score >= 6.0:
        return "yellow"
    elif score >= 4.0:
        return "white"
    return "red"


def _action_style(action: str) -> str:
    styles = {
        "STRONG_BUY": "bold green",
        "BUY": "green",
        "WEAK_BUY": "yellow",
        "SKIP": "dim white",
    }
    return styles.get(action, "white")


def _tier_color(tier_tags: list) -> str:
    tier_priority = [
        ("LEGENDARY", "bold magenta"),
        ("INSIDER_ALERT", "bold red"),
        ("SMART_MONEY", "bold yellow"),
        ("TIER_1_WHALE", "cyan"),
        ("TIER_2_SHARK", "white"),
    ]
    for tier, color in tier_priority:
        if tier in tier_tags:
            return color
    return "dim white"


class Dashboard:
    """
    Rich terminal dashboard that updates every 5 seconds.
    """

    def __init__(self, settings=None):
        self.settings = settings
        self.console = Console()
        self._live: Optional[Live] = None
        self._start_ts: float = now_ts()

        # State references (updated by bot)
        self.portfolio_state: Optional[PortfolioState] = None
        self.recent_signals: list[TradeSignal] = []
        self.recent_alerts: list[WhaleAlert] = []
        self.open_trades: list[dict] = []
        self.ws_connected: bool = False
        self.last_cycle_ts: float = now_ts()
        self.markets_tracked: int = 0
        self.api_calls_today: int = 0
        self.mode: str = "PAPER"
        self.paper_start_ts: float = now_ts()
        self.paper_end_ts: float = now_ts() + 48 * 3600
        self.db_size_mb: float = 0.0
        self.budget: float = 1000.0

    def start(self) -> None:
        """Start the live dashboard."""
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=0.2,  # 5 second refresh
            screen=False,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()

    def update(self) -> None:
        """Refresh the dashboard display."""
        if self._live:
            self._live.update(self._render())

    def _render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._render_header(), name="header", size=4),
            Layout(name="middle", ratio=3),
            Layout(name="bottom", ratio=2),
            Layout(self._render_status_bar(), name="statusbar", size=3),
        )
        layout["middle"].split_row(
            Layout(self._render_open_positions(), name="positions", ratio=3),
            Layout(self._render_stats_panel(), name="stats", ratio=1),
        )
        layout["bottom"].split_row(
            Layout(self._render_recent_signals(), name="signals", ratio=2),
            Layout(self._render_whale_alerts(), name="whales", ratio=1),
        )
        return layout

    def _render_header(self) -> Panel:
        runtime = format_duration(now_ts() - self._start_ts)
        portfolio_value = self.portfolio_state.total_portfolio_value if self.portfolio_state else self.budget
        roi = self.portfolio_state.roi_pct if self.portfolio_state else 0.0

        mode_color = "green" if self.mode == "LIVE" else "yellow"
        mode_str = f"[bold {mode_color}][{self.mode} MODE][/bold {mode_color}]"

        # Paper countdown
        extra = ""
        if self.mode == "PAPER":
            remaining = max(0.0, self.paper_end_ts - now_ts())
            countdown = format_duration(remaining)
            extra = f" | [cyan]48h countdown: {countdown}[/cyan]"

        pnl_str = _color_pct(roi)

        header_text = (
            f"[bold white]POLYMARKET ELITE BOT[/bold white]  {mode_str}  "
            f"Runtime: [white]{runtime}[/white]\n"
            f"Budget: [white]{fmt_usdc(self.budget)}[/white]  |  "
            f"Portfolio: [white]{fmt_usdc(portfolio_value)}[/white] ({pnl_str})"
            f"{extra}"
        )
        return Panel(header_text, style="bold blue", padding=(0, 1))

    def _render_open_positions(self) -> Panel:
        table = Table(
            title="[bold]Open Positions[/bold]",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
        )
        table.add_column("Market", style="white", max_width=25)
        table.add_column("Side", justify="center")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L%", justify="right")
        table.add_column("Hrs", justify="right")
        table.add_column("Source", justify="center")

        for trade in self.open_trades[-10:]:
            slug = str(trade.get("market_slug") or "")[:22]
            outcome = str(trade.get("outcome") or "?")
            entry = float(trade.get("fill_price") or trade.get("price") or 0)
            size = float(trade.get("size_usdc") or 0)
            shares = float(trade.get("shares") or 0)
            # Estimate current price
            current = float(trade.get("current_price") or entry)
            pnl = (current - entry) * shares
            pnl_pct = (current - entry) / max(entry, 0.001) * 100

            ts = float(trade.get("timestamp") or now_ts())
            hours = (now_ts() - ts) / 3600

            source = "WHALE" if trade.get("is_whale_copy") else "SIGNAL"
            source_style = "cyan" if source == "WHALE" else "white"

            side_style = "green" if outcome == "YES" else "red"
            table.add_row(
                slug,
                f"[{side_style}]{outcome}[/{side_style}]",
                f"{entry:.4f}",
                f"{current:.4f}",
                fmt_usdc(size),
                _color_pnl(pnl),
                _color_pct(pnl_pct),
                f"{hours:.1f}h",
                f"[{source_style}]{source}[/{source_style}]",
            )

        if not self.open_trades:
            table.add_row("[dim]No open positions[/dim]", "", "", "", "", "", "", "", "")

        return Panel(table, border_style="blue")

    def _render_recent_signals(self) -> Panel:
        table = Table(
            title="[bold]Recent Signals[/bold] (last 10)",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
        )
        table.add_column("Market", max_width=22)
        table.add_column("Dir", justify="center")
        table.add_column("Edge", justify="right")
        table.add_column("AI", justify="right")
        table.add_column("Whale", justify="right")
        table.add_column("Comp", justify="right")
        table.add_column("Action", justify="center")
        table.add_column("Age", justify="right")

        for sig in self.recent_signals[-10:]:
            direction_style = "green" if sig.final_direction == "YES" else (
                "red" if sig.final_direction == "NO" else "dim"
            )
            action_style = _action_style(sig.action)
            comp_color = _signal_color(sig.composite_score)
            age = format_duration(now_ts() - sig.computed_at)

            table.add_row(
                sig.market_slug[:22],
                f"[{direction_style}]{sig.final_direction}[/{direction_style}]",
                f"{sig.ai_edge:+.3f}",
                f"{sig.ai_score:.1f}",
                f"{sig.whale_score:.1f}",
                f"[{comp_color}]{sig.composite_score:.1f}[/{comp_color}]",
                f"[{action_style}]{sig.action}[/{action_style}]",
                age,
            )

        if not self.recent_signals:
            table.add_row("[dim]No signals yet[/dim]", "", "", "", "", "", "", "")

        return Panel(table, border_style="blue")

    def _render_whale_alerts(self) -> Panel:
        table = Table(
            title="[bold]Whale Alerts[/bold] (last 5)",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 0),
        )
        table.add_column("Time", max_width=8)
        table.add_column("Wallet", max_width=10)
        table.add_column("Market", max_width=15)
        table.add_column("Side", justify="center")
        table.add_column("$", justify="right")
        table.add_column("Lvl", justify="center")

        for alert in self.recent_alerts[-5:]:
            tier_tags = alert.wallet_tier if isinstance(alert.wallet_tier, list) else []
            tier_color = _tier_color(tier_tags)
            tier_display = tier_tags[0] if tier_tags else "UNKNOWN"
            tier_abbrev = tier_display[:7]
            wallet_short = alert.wallet_address[:8] + "..."
            level_color = "bold red" if alert.alert_level == "CRITICAL" else (
                "yellow" if alert.alert_level == "HIGH" else "white"
            )
            age = format_duration(now_ts() - alert.timestamp)
            side_style = "green" if alert.outcome == "YES" else "red"

            table.add_row(
                age,
                f"[{tier_color}]{wallet_short}[/{tier_color}]",
                alert.market_slug[:15],
                f"[{side_style}]{alert.outcome}[/{side_style}]",
                fmt_usdc(alert.size_usdc),
                f"[{level_color}]{alert.alert_level[:4]}[/{level_color}]",
            )

        if not self.recent_alerts:
            table.add_row("[dim]No alerts[/dim]", "", "", "", "", "")

        return Panel(table, border_style="magenta")

    def _render_stats_panel(self) -> Panel:
        ps = self.portfolio_state

        if ps:
            wr_text = f"{ps.roi_pct:.1f}%"  # we use roi_pct as proxy for win rate display
            total_pnl_text = _color_pnl(ps.total_pnl)
            drawdown_color = "red" if ps.current_drawdown > 0.05 else "white"
            drawdown_text = f"[{drawdown_color}]{ps.current_drawdown:.1%}[/{drawdown_color}]"
            positions_text = f"{ps.open_positions_count}/{self.settings.max_open_positions if self.settings else 10}"
            daily_pnl_text = _color_pnl(ps.daily_pnl)
            cash_text = fmt_usdc(ps.cash_balance)
        else:
            total_pnl_text = "$0.00"
            drawdown_text = "0.00%"
            positions_text = "0/10"
            daily_pnl_text = "$0.00"
            cash_text = fmt_usdc(self.budget)

        stats_text = (
            f"[bold]STATS[/bold]\n\n"
            f"[cyan]Total P&L:[/cyan]  {total_pnl_text}\n"
            f"[cyan]Daily P&L:[/cyan]  {daily_pnl_text}\n"
            f"[cyan]Cash:[/cyan]       {cash_text}\n"
            f"[cyan]Drawdown:[/cyan]   {drawdown_text}\n"
            f"[cyan]Positions:[/cyan]  {positions_text}\n"
        )

        # Trading allowed?
        if ps and (ps.daily_loss_limit_breached or ps.max_drawdown_breached):
            stats_text += "\n[bold red]⚠ TRADING HALTED[/bold red]"
        else:
            stats_text += "\n[bold green]✓ Trading Active[/bold green]"

        return Panel(stats_text, border_style="green", title="[bold]Status[/bold]")

    def _render_status_bar(self) -> Panel:
        ws_status = "[bold green]CONNECTED[/bold green]" if self.ws_connected else "[bold red]DISCONNECTED[/bold red]"
        last_cycle_age = format_duration(now_ts() - self.last_cycle_ts)

        mode_color = "green" if self.mode == "LIVE" else "yellow"
        mode_display = f"[{mode_color}]{self.mode}[/{mode_color}]"

        paper_info = ""
        if self.mode == "PAPER":
            remaining = max(0.0, self.paper_end_ts - now_ts())
            paper_info = f" ({format_duration(remaining)} remaining)"

        status = (
            f"WS: {ws_status}  |  "
            f"Last cycle: [white]{last_cycle_age} ago[/white]  |  "
            f"Markets tracked: [white]{self.markets_tracked}[/white]  |  "
            f"API calls today: [white]{self.api_calls_today}[/white]  |  "
            f"DB: [white]{self.db_size_mb:.1f}MB[/white]  |  "
            f"Mode: {mode_display}{paper_info}"
        )
        return Panel(status, style="dim", padding=(0, 1))

    def print_paper_report(
        self,
        stats: dict,
        go_criteria_results: dict,
        verdict: str,
    ) -> None:
        """Print the 48-hour paper trading report."""
        self.console.print("\n")
        self.console.rule("[bold cyan]48-HOUR PAPER TRADING REPORT[/bold cyan]")

        # Summary
        summary = Table(title="Performance Summary", box=box.ROUNDED)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", justify="right")

        summary.add_row("Total P&L", _color_pnl(stats.get("total_pnl", 0)))
        summary.add_row("Total ROI", _color_pct(stats.get("roi_pct", 0)))
        summary.add_row("Win Rate", f"{stats.get('win_rate', 0):.1%}")
        summary.add_row("Profit Factor", f"{stats.get('profit_factor', 0):.2f}")
        summary.add_row("Sharpe Ratio", f"{stats.get('sharpe_ratio', 0):.2f}")
        summary.add_row("Max Drawdown", f"{stats.get('max_drawdown', 0):.1%}")
        summary.add_row("Total Trades", str(stats.get("total_trades", 0)))
        summary.add_row("Whale Copy Trades", str(stats.get("whale_copy_count", 0)))
        summary.add_row("Arb Trades", str(stats.get("arb_count", 0)))
        self.console.print(summary)

        # Go/No-Go criteria
        self.console.print("\n")
        criteria_table = Table(title="GO/NO-GO Criteria", box=box.ROUNDED)
        criteria_table.add_column("Criterion", style="white")
        criteria_table.add_column("Required", justify="right")
        criteria_table.add_column("Actual", justify="right")
        criteria_table.add_column("Result", justify="center")

        for criterion, result_data in go_criteria_results.items():
            passed = result_data.get("passed", False)
            required = str(result_data.get("required", ""))
            actual = str(result_data.get("actual", ""))
            status = "[bold green]✓ PASS[/bold green]" if passed else "[bold red]✗ FAIL[/bold red]"
            criteria_table.add_row(criterion, required, actual, status)

        self.console.print(criteria_table)

        # Verdict
        self.console.print("\n")
        if verdict == "GO":
            self.console.print(
                "[bold green]═══════════════════════════════════════════════[/bold green]"
            )
            self.console.print(
                "[bold green]  PAPER TRADING COMPLETE — READY FOR LIVE  [/bold green]"
            )
            self.console.print(
                "[bold green]═══════════════════════════════════════════════[/bold green]"
            )
            self.console.print(
                "\nTo go live run: [bold white]python main.py --mode live --budget YOUR_AMOUNT[/bold white]"
            )
            self.console.print(
                '\nType [bold yellow]CONFIRM LIVE TRADING[/bold yellow] to activate now, or Ctrl+C to review'
            )
        else:
            self.console.print("[bold red]═══════════════════════════════════════[/bold red]")
            self.console.print("[bold red]  NO-GO: CONTINUING PAPER TRADING     [/bold red]")
            self.console.print("[bold red]═══════════════════════════════════════[/bold red]")
            self.console.print("\nFailed criteria:")
            for criterion, result_data in go_criteria_results.items():
                if not result_data.get("passed"):
                    self.console.print(
                        f"  [red]✗[/red] {criterion}: got {result_data.get('actual')}, "
                        f"needed {result_data.get('required')}"
                    )
            self.console.print(
                "\n[yellow]Continuing paper trading. Re-evaluating in 24 hours.[/yellow]"
            )

    def print_session_summary(self, portfolio_state: PortfolioState) -> None:
        """Print final session summary on shutdown."""
        self.console.print("\n")
        self.console.rule("[bold]Session Ended[/bold]")
        self.console.print(f"Mode: [bold]{portfolio_state.mode}[/bold]")
        self.console.print(f"Final Portfolio: [bold]{fmt_usdc(portfolio_state.total_portfolio_value)}[/bold]")
        self.console.print(f"Total P&L: {_color_pnl(portfolio_state.total_pnl)}")
        self.console.print(f"ROI: {_color_pct(portfolio_state.roi_pct)}")
        self.console.print(f"Runtime: {format_duration(now_ts() - self._start_ts)}")
        self.console.print("\n[bold green]All orders cancelled. State saved.[/bold green]")
        self.console.print("[dim]Session ended.[/dim]")

    def print_diagnostic_report(self, modules_analysis: dict) -> None:
        """Print diagnostic report after 7 days of no-go."""
        self.console.print("\n")
        self.console.rule("[bold red]7-DAY DIAGNOSTIC REPORT[/bold red]")
        self.console.print(
            "\nAfter 7 days of paper trading, the bot has not met GO criteria.\n"
        )
        self.console.print("[bold]Recommended adjustments:[/bold]\n")
        for module, suggestions in modules_analysis.items():
            self.console.print(f"[cyan]{module}:[/cyan]")
            for s in suggestions:
                self.console.print(f"  • {s}")
