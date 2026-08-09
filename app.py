"""
Gradio front-end for the research agent (Phase 5).

Running locally
---------------
    source venv/bin/activate
    python app.py
"""

import gradio as gr
import spaces
from dotenv import load_dotenv

load_dotenv()

from graph.build_graph import build_graph   # noqa: E402
from eval.citation_checker import run_full_check  # noqa: E402

# ---------------------------------------------------------------------------
# Graph -- lazy-initialised on first request.
# ---------------------------------------------------------------------------
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ---------------------------------------------------------------------------
# Node display labels (no emojis).
# ---------------------------------------------------------------------------
_NODE_LABEL: dict[str, str] = {
    "planner":     "Planning sub-questions",
    "research":    "Searching sources",
    "synthesizer": "Drafting report",
    "critic":      "Critiquing draft",
    "reviser":     "Revising draft",
    "finalize":    "Finalising",
}


# ---------------------------------------------------------------------------
# Generator -- drives all three output components.
# ---------------------------------------------------------------------------
@spaces.GPU
def _dummy_gpu_function():
    # Hugging Face ZeroGPU spaces require at least one function to be decorated
    # with @spaces.GPU to start the container, even if the app never calls it.
    pass


def run_research(topic: str):
    topic = topic.strip()
    if not topic:
        yield _status_html([("Please enter a topic.", "", True, True)]), "", ""
        return

    graph = _get_graph()
    # Each entry: (label, detail, is_done, is_error)
    steps: list[tuple[str, str, bool, bool]] = []
    accumulated: dict = {}

    try:
        for chunk in graph.stream(
            {"topic": topic, "iteration": 0},
            stream_mode="updates",
        ):
            for node_name, updates in chunk.items():
                accumulated.update(updates)
                label  = _NODE_LABEL.get(node_name, node_name)
                detail = updates.get("status", "")

                if steps:
                    prev = steps[-1]
                    steps[-1] = (prev[0], prev[1], True, False)

                steps.append((label, detail, False, False))
                yield _status_html(steps), "", ""

    except Exception as exc:  # noqa: BLE001
        steps.append((f"Error: {exc}", "", True, True))
        yield _status_html(steps), "", f"**Error:** {exc}"
        return

    if steps:
        last = steps[-1]
        steps[-1] = (last[0], last[1], True, False)

    report  = accumulated.get("final_report", "No report generated.")
    cr_pass = bool(accumulated.get("critique_pass", False))
    iters   = accumulated.get("iteration", 0)
    issues  = accumulated.get("critique_issues") or []
    check   = run_full_check(report)

    yield _status_html(steps), report, _metrics_md(check, cr_pass, iters, issues)


# ---------------------------------------------------------------------------
# HTML builder -- CSS-drawn indicators, no emojis.
# ---------------------------------------------------------------------------
_CHECK_SVG = (
    '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" class="step-check">'
    '<polyline points="2.5,7 5.5,10 11.5,4" stroke="currentColor" '
    'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
    "</svg>"
)


def _status_html(steps: list[tuple[str, str, bool, bool]]) -> str:
    rows = ""
    for label, detail, done, error in steps:
        if error:
            cls       = "step-error"
            indicator = '<span class="step-dot step-dot-error"></span>'
        elif done:
            cls       = "step-done"
            indicator = _CHECK_SVG
        else:
            cls       = "step-active"
            indicator = '<span class="step-dot step-dot-active"></span>'

        detail_html = f'<span class="step-detail">{detail}</span>' if detail else ""
        spinner     = '<span class="spinner"></span>' if not done and not error else ""

        rows += (
            f'<div class="step {cls}">'
            f'  <span class="step-indicator">{indicator}</span>'
            f'  <span class="step-body">'
            f'    <span class="step-label">{label}</span>'
            f'    {detail_html}'
            f'  </span>'
            f'  {spinner}'
            f'</div>'
        )
    return f'<div class="pipeline-log">{rows}</div>' if rows else ""


# ---------------------------------------------------------------------------
# Metrics builder -- no emojis.
# ---------------------------------------------------------------------------
def _metrics_md(check: dict, cr_pass: bool, iters: int, issues: list[str]) -> str:
    result_label = "Passed" if cr_pass else "Did not pass (hit revision cap)"
    pct   = check["coverage_pct"]
    cited = check["cited_sentences"]
    total = check["total_sentences"]
    orp   = len(check["orphaned_citations"])
    unu   = len(check["unused_sources"])

    md = (
        "| Metric | Value |\n"
        "|:---|:---|\n"
        f"| Critique result | {result_label} |\n"
        f"| Revision iterations | {iters} |\n"
        f"| Citation coverage | {pct}% ({cited}/{total} sentences) |\n"
        f"| Orphaned citations | {orp} |\n"
        f"| Unused sources | {unu} |\n"
    )

    if issues:
        md += "\n**Critic issues:**\n"
        for iss in issues:
            md += f"- {iss}\n"

    if check["uncited_sentences"]:
        md += "\n**Uncited sentences (sample):**\n"
        for s in check["uncited_sentences"][:5]:
            md += f"- {s}\n"

    return md


# ---------------------------------------------------------------------------
# CSS -- professional dark theme, zero emojis.
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400&display=swap');

:root {
    --bg:          #0c0e12;
    --bg-raised:   #12151b;
    --bg-input:    #181c24;
    --accent:      #3b82f6;
    --accent-dim:  rgba(59, 130, 246, 0.12);
    --accent-ring: rgba(59, 130, 246, 0.28);
    --success:     #22c55e;
    --error-col:   #ef4444;
    --t1:          #e4e7ee;
    --t2:          #8892a4;
    --t3:          #4a5568;
    --border:      rgba(255,255,255,0.06);
    --border-mid:  rgba(255,255,255,0.10);
    --r:           6px;
    --r-lg:        10px;
}

body,
.gradio-container {
    background: var(--bg) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: var(--t1) !important;
    font-size: 14px;
}

/* Wordmark */
.ra-header {
    padding: 36px 0 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.ra-wordmark {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--t1);
    letter-spacing: -0.01em;
}
.ra-wordmark span { color: var(--accent); }
.ra-tagline {
    font-size: 0.78rem;
    color: var(--t3);
    margin-top: 4px;
}

/* Inputs */
.gradio-container textarea,
.gradio-container input[type="text"] {
    background:    var(--bg-input) !important;
    border:        1px solid var(--border-mid) !important;
    border-radius: var(--r) !important;
    color:         var(--t1) !important;
    font-family:   'Inter', sans-serif !important;
    font-size:     0.88rem !important;
    padding:       11px 13px !important;
    transition:    border-color 0.15s, box-shadow 0.15s;
}
.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus {
    border-color: var(--accent) !important;
    box-shadow:   0 0 0 3px var(--accent-ring) !important;
    outline:      none !important;
}
.gradio-container label {
    color:          var(--t2) !important;
    font-size:      0.72rem !important;
    font-weight:    500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* Run button */
#run-btn {
    background:     var(--accent) !important;
    border:         none !important;
    border-radius:  var(--r) !important;
    color:          #fff !important;
    font-family:    'Inter', sans-serif !important;
    font-size:      0.85rem !important;
    font-weight:    600 !important;
    padding:        11px 26px !important;
    letter-spacing: 0.01em !important;
    cursor:         pointer !important;
    transition:     background 0.15s, box-shadow 0.15s, transform 0.1s !important;
    white-space:    nowrap;
    min-height:     42px;
}
#run-btn:hover {
    background: #2563eb !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.35) !important;
    transform:  translateY(-1px) !important;
}
#run-btn:active { transform: translateY(0) !important; }

/* Section label */
.ra-section {
    font-size:      0.66rem;
    font-weight:    600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color:          var(--t3);
    margin:         26px 0 8px;
}

/* Pipeline log */
.pipeline-log {
    display:        flex;
    flex-direction: column;
    gap:            2px;
    background:     var(--bg-raised);
    border:         1px solid var(--border);
    border-radius:  var(--r-lg);
    padding:        6px;
    min-height:     48px;
}
.step {
    display:       flex;
    align-items:   center;
    gap:           10px;
    padding:       8px 10px;
    border-radius: 5px;
    font-size:     0.83rem;
    font-weight:   500;
    transition:    background 0.2s, color 0.2s;
}
.step-body {
    display:     flex;
    flex:        1;
    align-items: baseline;
    gap:         8px;
    min-width:   0;
    overflow:    hidden;
}
.step-label  { color: inherit; white-space: nowrap; }
.step-detail {
    color:         var(--t3);
    font-weight:   400;
    font-size:     0.78rem;
    overflow:      hidden;
    text-overflow: ellipsis;
    white-space:   nowrap;
}
.step-done   { color: var(--t3); }
.step-active { background: var(--accent-dim); color: var(--t1); }
.step-error  { color: var(--error-col); }

.step-indicator {
    display:     flex;
    align-items: center;
    flex-shrink: 0;
    width:       16px;
}
.step-check { color: var(--success); }
.step-dot {
    display:       block;
    width:         7px;
    height:        7px;
    border-radius: 50%;
}
.step-dot-active {
    background: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-dim);
}
.step-dot-error { background: var(--error-col); }

/* Spinner */
.spinner {
    width:            13px;
    height:           13px;
    border:           2px solid var(--border-mid);
    border-top-color: var(--accent);
    border-radius:    50%;
    animation:        spin 0.75s linear infinite;
    flex-shrink:      0;
    margin-left:      auto;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Tabs */
.gradio-container .tabs     { border-bottom: 1px solid var(--border) !important; }
.gradio-container .tab-nav button {
    color:         var(--t3) !important;
    font-family:   'Inter', sans-serif !important;
    font-size:     0.8rem !important;
    font-weight:   500 !important;
    padding:       9px 16px !important;
    background:    transparent !important;
    border:        none !important;
    border-bottom: 2px solid transparent !important;
    transition:    color 0.15s, border-color 0.15s !important;
}
.gradio-container .tab-nav button.selected {
    color:               var(--t1) !important;
    border-bottom-color: var(--accent) !important;
}

/* Markdown prose */
.gradio-container .prose,
.gradio-container [data-testid="markdown"] {
    color:       var(--t1) !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.75 !important;
    font-size:   0.88rem !important;
}
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3 {
    color:          var(--t1) !important;
    font-weight:    600 !important;
    letter-spacing: -0.01em !important;
}
.gradio-container .prose h2 {
    font-size:      1rem !important;
    border-bottom:  1px solid var(--border) !important;
    padding-bottom: 5px !important;
    margin-top:     26px !important;
}
.gradio-container .prose h3 { font-size: 0.9rem !important; }
.gradio-container .prose a  { color: var(--accent) !important; }
.gradio-container .prose code,
.gradio-container .prose pre {
    background:    var(--bg-input) !important;
    border:        1px solid var(--border) !important;
    border-radius: var(--r) !important;
    font-family:   'JetBrains Mono', monospace !important;
    font-size:     0.82em !important;
    color:         #a8b4cc !important;
}

/* Tables */
.gradio-container .prose table { border-collapse: collapse; width: 100%; font-size: 0.84rem; }
.gradio-container .prose thead tr { border-bottom: 1px solid var(--border-mid) !important; }
.gradio-container .prose th {
    color:          var(--t2) !important;
    font-weight:    500 !important;
    font-size:      0.7rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding:        7px 12px !important;
    background:     transparent !important;
}
.gradio-container .prose td {
    border-top: 1px solid var(--border) !important;
    padding:    7px 12px !important;
    color:      var(--t1) !important;
}
.gradio-container .prose tr:first-child td { border-top: none !important; }

/* Blockquote (cap-out warning) */
.gradio-container .prose blockquote {
    border-left:   3px solid var(--accent) !important;
    background:    var(--accent-dim) !important;
    padding:       10px 14px !important;
    color:         var(--t2) !important;
    border-radius: 0 var(--r) var(--r) 0 !important;
    margin:        14px 0 !important;
    font-size:     0.83rem !important;
}

/* Examples */
.gradio-container .examples-holder table td {
    background:    var(--bg-raised) !important;
    border:        1px solid var(--border) !important;
    border-radius: var(--r) !important;
    color:         var(--t2) !important;
    font-size:     0.78rem !important;
    padding:       5px 11px !important;
    cursor:        pointer;
    transition:    border-color 0.15s, color 0.15s;
}
.gradio-container .examples-holder table td:hover {
    border-color: var(--accent) !important;
    color:        var(--t1) !important;
}

/* Layout */
footer { display: none !important; }
.gradio-container > .main > .wrap { max-width: 900px; margin: 0 auto; }
"""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
EXAMPLE_TOPICS = [
    ["Effects of remote work on employee productivity"],
    ["How mRNA vaccines trigger an immune response"],
    ["The economics of vertical farming in urban environments"],
    ["Global efforts to regulate AI-generated deepfakes"],
    ["The causes of the 2008 global financial crisis"],
]

with gr.Blocks(
    title="Research Agent",
    css=CSS,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    )
) as demo:

    gr.HTML("""
    <div class="ra-header">
        <div class="ra-wordmark">Research<span>Agent</span></div>
        <div class="ra-tagline">
            Multi-step LLM pipeline — plans sub-questions, searches the web, drafts a cited report,
            critiques its own work, and revises until it passes or hits the revision cap.
        </div>
    </div>
    """)

    with gr.Row(equal_height=True):
        topic_input = gr.Textbox(
            placeholder="Enter a research topic",
            label="Topic",
            lines=1,
            scale=5,
        )
        run_btn = gr.Button("Run", variant="primary", scale=1, elem_id="run-btn")

    gr.Examples(
        examples=EXAMPLE_TOPICS,
        inputs=topic_input,
        label="Examples",
    )

    gr.HTML('<div class="ra-section">Pipeline</div>')
    status_out = gr.HTML(value="", label="")

    gr.HTML('<div class="ra-section">Output</div>')
    with gr.Tabs():
        with gr.Tab("Report"):
            report_out = gr.Markdown(
                value="Run the agent to see the report here.",
                elem_id="report-out",
            )
        with gr.Tab("Citation Metrics"):
            metrics_out = gr.Markdown(
                value="Metrics will appear after the run completes.",
                elem_id="metrics-out",
            )

    run_btn.click(
        fn=run_research,
        inputs=topic_input,
        outputs=[status_out, report_out, metrics_out],
    )
    topic_input.submit(
        fn=run_research,
        inputs=topic_input,
        outputs=[status_out, report_out, metrics_out],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
