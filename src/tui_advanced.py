"""Deep Research 高级终端 UI（基于 Textual）。

功能：
  - 左右分栏：左侧为可鼠标滚动的 Agent 事件日志，右侧为实时报告/状态预览
  - 顶部 Header：显示研究主题、运行状态、已用时间
  - 底部输入栏：处理 Agent 澄清提问
  - 支持 Ctrl+C / q 优雅退出

用法：
  python src/tui_advanced.py "研究主题"
  python src/tui_advanced.py          # 交互式输入主题
"""

import asyncio
import sys
import time
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, RichLog, Static, Markdown

from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ─────────────────────────────────────
# 自定义小部件
# ─────────────────────────────────────

class StatusBar(Static):
    """顶部状态条：显示研究主题、状态、耗时。"""

    topic: reactive[str] = reactive("（未设置）")
    status: reactive[str] = reactive("⏳ 准备中")
    elapsed: reactive[float] = reactive(0.0)

    def render(self) -> Text:
        mins, secs = divmod(int(self.elapsed), 60)
        time_str = f"{mins:02d}:{secs:02d}"
        txt = Text()
        txt.append("📌 ", style="bold")
        txt.append(self.topic, style="bold white")
        txt.append("  │  ", style="dim")
        txt.append(self.status, style="bold cyan")
        txt.append("  │  ", style="dim")
        txt.append(f"⏱ {time_str}", style="bold yellow")
        return txt


class PreviewPanel(Vertical):
    """右侧预览区：显示实时报告或研究阶段摘要。"""

    DEFAULT_CSS = """
    PreviewPanel {
        width: 1fr;
        height: 1fr;
        border-left: solid $primary;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("📋 [bold]实时预览[/]", id="preview-title")
        yield Markdown("*等待研究启动…*", id="preview-content")

    def update_content(self, markdown_text: str) -> None:
        """替换预览区的 Markdown 内容（原地刷新，不追加）。"""
        widget = self.query_one("#preview-content", Markdown)
        widget.update(markdown_text)

    def set_title(self, title: str) -> None:
        widget = self.query_one("#preview-title", Static)
        widget.update(title)


# ─────────────────────────────────────
# 主应用
# ─────────────────────────────────────

class DeepResearchApp(App):
    """Deep Research 高级 TUI。"""

    CSS = """
    Screen {
        layout: vertical;
    }

    #status-bar {
        dock: top;
        height: 3;
        padding: 0 2;
        background: $surface;
        border-bottom: solid $primary;
        content-align-vertical: middle;
    }

    #main-area {
        height: 1fr;
    }

    #event-log-container {
        width: 3fr;
        height: 1fr;
        border-right: solid $primary;
    }

    #event-log-title {
        dock: top;
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text;
    }

    #event-log {
        height: 1fr;
        padding: 0 1;
        scrollbar-size: 1 1;
    }

    PreviewPanel {
        width: 2fr;
    }

    #preview-title {
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text;
    }

    #preview-content {
        height: 1fr;
        padding: 0 1;
        overflow-y: auto;
    }

    #clarify-container {
        dock: bottom;
        height: auto;
        max-height: 5;
        padding: 0 1;
        background: $surface;
        border-top: solid $warning;
        display: none;
    }

    #clarify-label {
        height: 1;
        color: $warning;
    }

    #clarify-input {
        height: 3;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "退出", show=True),
        Binding("ctrl+c", "quit", "退出"),
    ]

    # 内部状态
    _topic: str = ""
    _start_time: float = 0
    _timer_handle = None
    _clarify_future: asyncio.Future | None = None
    _streaming_buffer: list[str] = []  # 缓冲 LLM token

    # ── 研究进度跟踪 ──
    _current_node: str = ""
    _sources: list[str] = []       # 搜索工具调用记录
    _research_notes: list[str] = []  # 压缩摘要记录
    _brief: str = ""

    def __init__(self, topic: str = ""):
        super().__init__()
        self._topic = topic

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield StatusBar(id="status-bar")
        with Horizontal(id="main-area"):
            with Vertical(id="event-log-container"):
                yield Static("🔍 [bold]Agent 事件日志[/]", id="event-log-title")
                yield RichLog(
                    id="event-log",
                    highlight=True,
                    markup=True,
                    wrap=True,
                    auto_scroll=True,
                    max_lines=5000,
                )
            yield PreviewPanel()
        with Vertical(id="clarify-container"):
            yield Static("🤔 Agent 需要你的回复：", id="clarify-label")
            yield Input(placeholder="输入你的回答后按 Enter…", id="clarify-input")
        yield Footer()

    def on_mount(self) -> None:
        """应用挂载后启动研究流程。"""
        self.title = "Deep Research Agent"
        self.sub_title = "高级终端界面"

        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.topic = self._topic or "（等待输入）"
        status_bar.status = "🚀 启动中…"

        if self._topic:
            self._start_research()
        else:
            # 无主题时，用底部输入框获取
            self._show_clarify("请输入你想研究的主题：", is_topic_input=True)

    # ─── 事件日志写入 ───

    def _log(self, *args, **kwargs) -> None:
        """向左侧日志区写入一行。"""
        log_widget = self.query_one("#event-log", RichLog)
        log_widget.write(*args, **kwargs)

    def _log_stream_flush(self) -> None:
        """将缓冲的 LLM token 一次性刷入日志。"""
        if self._streaming_buffer:
            text = "".join(self._streaming_buffer)
            self._streaming_buffer.clear()
            if text.strip():
                self._log(Text(text, style="dim italic"))

    # ─── 预览区更新 ───

    def _update_preview(self) -> None:
        """根据当前收集到的信息重建预览面板。"""
        preview = self.query_one(PreviewPanel)

        sections = []

        if self._brief:
            sections.append("## 📝 研究简报\n\n" + self._brief)

        if self._sources:
            src_lines = "\n".join(f"- {s}" for s in self._sources[-20:])  # 最多显示 20 条
            sections.append(f"## 🔎 搜索记录 ({len(self._sources)} 次)\n\n{src_lines}")

        if self._research_notes:
            notes = "\n\n---\n\n".join(self._research_notes[-5:])  # 最多显示 5 段
            sections.append(f"## 📚 研究笔记 ({len(self._research_notes)} 段)\n\n{notes}")

        if not sections:
            preview.update_content("*研究尚未开始，等待 Agent 启动…*")
        else:
            preview.update_content("\n\n---\n\n".join(sections))

    # ─── 计时器 ───

    def _start_timer(self) -> None:
        self._start_time = time.time()
        self._timer_handle = self.set_interval(1.0, self._tick_timer)

    def _tick_timer(self) -> None:
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.elapsed = time.time() - self._start_time

    def _stop_timer(self) -> None:
        if self._timer_handle:
            self._timer_handle.stop()

    # ─── Clarify 交互 ───

    def _show_clarify(self, question: str, is_topic_input: bool = False) -> None:
        """显示底部输入框。"""
        container = self.query_one("#clarify-container")
        label = self.query_one("#clarify-label", Static)
        input_widget = self.query_one("#clarify-input", Input)

        if is_topic_input:
            label.update("📌 请输入研究主题：")
            input_widget.placeholder = "例如：人工智能在医疗领域的最新进展"
        else:
            label.update(f"🤔 {question}")
            input_widget.placeholder = "输入你的回答后按 Enter…"

        input_widget.value = ""
        container.display = True
        input_widget.focus()

    def _hide_clarify(self) -> None:
        """隐藏底部输入框。"""
        container = self.query_one("#clarify-container")
        container.display = False

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """处理输入框提交。"""
        answer = event.value.strip()
        if not answer:
            return

        input_widget = self.query_one("#clarify-input", Input)
        input_widget.value = ""
        self._hide_clarify()

        # 如果是主题输入（首次）
        if not self._topic:
            self._topic = answer
            status_bar = self.query_one("#status-bar", StatusBar)
            status_bar.topic = self._topic
            self._start_research()
            return

        # 如果是 clarify 回答
        if self._clarify_future and not self._clarify_future.done():
            self._clarify_future.set_result(answer)
            self._log(Text(f"  ✅ 你的回答：{answer}", style="bold green"))

    # ─── 事件回调（runner 调用） ───

    async def _on_event(self, event: dict) -> None:
        """将标准化事件渲染到 TUI 各区域。"""
        t = event["type"]

        if t == "node_start":
            self._log_stream_flush()
            node = event.get("node", "")
            self._current_node = node
            self._log(Text(f"\n▶ {node}", style="bold cyan"))
            # 更新状态栏
            status_bar = self.query_one("#status-bar", StatusBar)
            status_bar.status = f"🔬 {node}"

        elif t == "llm_start":
            self._log_stream_flush()
            model = event.get("model", "")
            self._log(Text(f"  🤖 {model}", style="dim"))

        elif t == "llm_stream":
            self._streaming_buffer.append(event["token"])
            # 每累积一定量就刷一次（避免过于频繁）
            if len(self._streaming_buffer) >= 20:
                self._log_stream_flush()

        elif t == "llm_end":
            self._log_stream_flush()
            usage = event.get("usage", {})
            total = usage.get("total_tokens", 0)
            if total:
                self._log(Text(f"  💰 {total} tokens", style="dim"))

            # 尝试从内容中提取研究简报
            content = event.get("content", "")
            if self._current_node == "write_research_brief" and content:
                self._brief = content[:500]
                self._update_preview()

        elif t == "tool_start":
            self._log_stream_flush()
            tool = event.get("tool", "")
            args = event.get("args", "")[:80]
            self._log(Text(f"  🔧 {tool}", style="bold yellow"), Text(f" {args}", style="dim"))
            # 记录搜索操作
            if "search" in tool.lower() or "web" in tool.lower():
                self._sources.append(f"`{tool}`: {args[:60]}")
                self._update_preview()

        elif t == "tool_end":
            result = event.get("result", "")[:120]
            self._log(Text(f"     └─ {result}", style="dim"))

            # 如果是 ConductResearch 的结果，记录为研究笔记
            tool = event.get("tool", "")
            if "research" in tool.lower():
                self._research_notes.append(event.get("result", "")[:500])
                self._update_preview()

        elif t == "clarify":
            self._log_stream_flush()
            question = event["question"]
            self._log(Text(f"\n🤔 Agent 提问：{question}", style="bold yellow"))
            # 显示输入框并等待回复
            self._show_clarify(question)

        elif t == "report":
            self._log_stream_flush()
            self._log(Text("\n✅ 最终报告已生成！请查看右侧预览面板。", style="bold green"))
            # 更新预览区为最终报告
            preview = self.query_one(PreviewPanel)
            preview.set_title("📋 [bold green]最终研究报告[/]")
            preview.update_content(event["content"])
            # 更新状态
            status_bar = self.query_one("#status-bar", StatusBar)
            status_bar.status = "✅ 研究完成"
            self._stop_timer()

    async def _on_clarify(self, question: str) -> str:
        """等待用户在 TUI 底部输入框回答。"""
        loop = asyncio.get_event_loop()
        self._clarify_future = loop.create_future()
        self._show_clarify(question)
        answer = await self._clarify_future
        self._clarify_future = None
        return answer

    # ─── 启动研究 ───

    @work(exclusive=True, thread=False)
    async def _start_research(self) -> None:
        """在后台运行研究流水线。"""
        from runner import run_research

        self._start_timer()
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.status = "🔬 研究中…"

        self._log(Text(f"🚀 开始研究：{self._topic}\n", style="bold green"))

        try:
            run_dir = await run_research(
                self._topic,
                on_event=self._on_event,
                on_clarify=self._on_clarify,
            )
            self._log(Text(f"\n📁 结果已保存：{run_dir}", style="bold blue"))
        except asyncio.CancelledError:
            self._log(Text("\n⚠️ 研究已取消", style="bold red"))
            status_bar.status = "⚠️ 已取消"
        except Exception as e:
            self._log(Text(f"\n❌ 研究出错：{e}", style="bold red"))
            status_bar.status = "❌ 出错"
        finally:
            self._stop_timer()


# ─── 入口 ───

def main():
    topic = ""
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    app = DeepResearchApp(topic=topic)
    app.run()


if __name__ == "__main__":
    main()
