"""
Run the generation agent over pinned_sample.tsv with headless Claude Code,
or with an open model behind an OpenAI-compatible server (vLLM).

    python3 run_llm.py run --model MODEL [--limit N] [--only DOC_ID ...]
    python3 run_llm.py probe --model MODEL "PROMPT" [--stream]
    python3 run_llm.py run --backend openai --base-url URL --model MODEL [--workers N]

`run` sends one request per row and appends one result row per document to
results/<model>.tsv; documents already there are skipped, so an interrupted
run resumes where it stopped. The full JSON reply of every call is kept in
results/raw/<model>/.

`probe` sends one free-form prompt in exactly the same environment, to check
that the environment is clean (README §5).

The environment, one fresh one per call:
- an empty temporary directory outside the repository as working directory,
  holding only a copy of wn_lookup.py: no CLAUDE.md, no project memory, and
  nothing carried over from the previous call;
- --safe-mode: no CLAUDE.md, skills, plugins, hooks, MCP servers, custom
  agents or output styles (login and billing work normally, so a
  subscription can be used, unlike --bare);
- --system-prompt: the three prompt documents replace the default prompt;
- --tools Bash with only `python3 wn_lookup.py ...` allowed, and
  --permission-mode dontAsk, so any other command is refused, not asked about;
- --strict-mcp-config, --disable-slash-commands, --no-session-persistence.

The openai backend (README §5, "Open models") replaces `claude -p` by a small
agent loop with the same system prompt, the same item prompt and one tool,
`bash`, which runs `python3 wn_lookup.py ...` and refuses everything else. The
reply is written in the same JSON shape as Claude Code's, so result_row() and
everything downstream read both.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE / "prompt_docs"
SYSTEM_FILES = ("sbn_manual.md", "repair_notation.md", "task.md")
SAMPLE = HERE / "pinned_sample.tsv"
RESULTS = HERE / "results"
TIMEOUT_S = 900
MAX_TURNS = 20
TOOL_OUTPUT_MAX = 30_000

# The openai backend's only tool. Named and shaped like Claude Code's Bash, so
# that task.md, which tells the model to run `python3 wn_lookup.py ...`, is
# used unchanged.
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": ("Run a shell command in the working directory. Only "
                        "`python3 wn_lookup.py senses <lemma> <pos>` and "
                        "`python3 wn_lookup.py check <lemma.pos.NN>` are "
                        "permitted; any other command is refused."),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string",
                                       "description": "The command to run"}},
            "required": ["command"],
        },
    },
}

# Muse Glimmer's chat template writes tool calls as ATEM blocks. Used only in
# --tool-mode text, when vLLM has no parser that turns them into tool_calls.
ATEM_INVOKE = re.compile(r'<atem:invoke name="([^"]+)">(.*?)</atem:invoke>', re.S)
ATEM_PARAM = re.compile(r'<atem:parameter name="([^"]+)">(.*?)</atem:parameter>', re.S)

DEVICE_TO_STRATEGY = (("d5_anchor_dummy", "anchor"), ("d4_movement", "hoist"))

FIELDS = [
    "doc_id", "repair_type", "interregnum", "nl_clean", "sbn_clean",
    "status", "reason", "site_index", "repair_synset", "reparandum_synset",
    "reparandum_value", "reparandum_surface", "nl_repair", "repair_onset", "sbn_repair",
    "devices", "strategy", "model", "parse", "num_turns", "input_tokens",
    "output_tokens", "cache_read_tokens", "duration_s", "raw_file",
]


def use_docs(path: Path) -> None:
    """Point the prompt at another set of prompt documents (`--docs`)."""
    global DOCS
    DOCS = path.resolve()


def prompt_hash() -> str:
    """sha256 of the system prompt plus the item template, so a result names
    the exact prompt it ran with."""
    h = hashlib.sha256(system_prompt().encode("utf-8"))
    h.update((DOCS / "item_template.md").read_bytes())
    return h.hexdigest()


def system_prompt() -> str:
    return "\n\n---\n\n".join((DOCS / f).read_text(encoding="utf-8")
                              for f in SYSTEM_FILES)


def item_prompt(row: dict) -> str:
    return (DOCS / "item_template.md").read_text(encoding="utf-8").format(**row)


def claude(prompt: str, model: str, stream: bool = False) -> tuple[str, float]:
    """One headless call in a fresh, empty working directory."""
    with tempfile.TemporaryDirectory(prefix="rsr_llm_") as work:
        shutil.copy(HERE / "wn_lookup.py", work)
        cmd = [
            "claude", "-p", prompt,
            "--model", model,
            "--safe-mode",
            # --safe-mode keeps permission settings. The user's settings allow
            # Read(//Users/<user>/**), under which `cat`/`ls` of any repo file
            # ran unasked in the first dry run. Load project settings only:
            # the empty work dir has none, so no allow rule survives.
            "--setting-sources", "project",
            "--system-prompt", system_prompt(),
            "--tools", "Bash",
            "--allowedTools", "Bash(python3 wn_lookup.py *)",
            "--permission-mode", "dontAsk",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
        ]
        cmd += (["--output-format", "stream-json", "--verbose"] if stream
                else ["--output-format", "json"])
        t0 = time.time()
        p = subprocess.run(cmd, cwd=work, capture_output=True, text=True,
                           timeout=TIMEOUT_S)
        if p.returncode != 0 and not p.stdout:
            raise RuntimeError(f"claude exited {p.returncode}: {p.stderr[-500:]}")
        return p.stdout, time.time() - t0


def run_wn_lookup(command: str, work: str) -> str:
    """The openai backend's stand-in for `Bash(python3 wn_lookup.py *)`.

    No shell is involved: `;`, `|` or `$(...)` reach wn_lookup.py as plain
    arguments, so nothing but the lookup can run.
    """
    try:
        argv = shlex.split(command)
    except ValueError as e:
        return f"Error: {e}"
    if argv[:2] not in (["python3", "wn_lookup.py"], ["python", "wn_lookup.py"]):
        return "Permission denied: only `python3 wn_lookup.py ...` may be run."
    p = subprocess.run([sys.executable, "wn_lookup.py", *argv[2:]], cwd=work,
                       capture_output=True, text=True, timeout=120)
    return (p.stdout + p.stderr)[:TOOL_OUTPUT_MAX] or "(no output)"


def text_tool_calls(content: str | None) -> list[dict]:
    calls = []
    for k, (name, body) in enumerate(ATEM_INVOKE.findall(content or "")):
        calls.append({"id": f"call_{k}", "type": "function",
                      "function": {"name": name,
                                   "arguments": json.dumps(dict(ATEM_PARAM.findall(body)))}})
    return calls


def post_json(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[-500:]}") from e


def openai_agent(prompt: str, args) -> tuple[str, float]:
    """One agent session against an OpenAI-compatible server.

    Returns a JSON string with the fields result_row() reads from Claude Code
    (`result`, `num_turns`, `usage`, `is_error`, `subtype`), plus the whole
    transcript without the system prompt.
    """
    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": prompt}]
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    turn_tokens: list[int] = []
    out = {"is_error": False, "subtype": "success", "result": "", "num_turns": 0}
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="rsr_llm_") as work:
        shutil.copy(HERE / "wn_lookup.py", work)
        for turn in range(1, MAX_TURNS + 1):
            payload = {
                "model": args.model, "messages": messages, "tools": [BASH_TOOL],
                "tool_choice": "auto" if args.tool_mode == "auto" else "none",
                "max_tokens": args.max_tokens, "temperature": args.temperature,
                "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed,
                "chat_template_kwargs": {"reasoning_strength": args.reasoning},
            }
            if args.tool_mode == "text":
                payload |= {"skip_special_tokens": False,
                            "stop": ["</atem:function_calls>"]}
            resp = post_json(args.base_url.rstrip("/") + "/chat/completions", payload)
            u = resp.get("usage") or {}
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            usage["input_tokens"] += u.get("prompt_tokens", 0) - cached
            usage["cache_read_input_tokens"] += cached
            usage["output_tokens"] += u.get("completion_tokens", 0)
            # per turn, to count how many items needed more than the first
            # round's 16384-token limit (README, Muse Glimmer round)
            turn_tokens.append(u.get("completion_tokens", 0))
            choice = resp["choices"][0]
            msg = choice["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            calls = msg.get("tool_calls") or (
                text_tool_calls(content) if args.tool_mode == "text" else [])
            out["num_turns"] = turn
            reply = {"role": "assistant", "content": content}
            if reasoning:
                reply["reasoning_content"] = reasoning
            if not calls:
                messages.append(reply)
                out["result"] = content
                if choice.get("finish_reason") == "length":
                    out.update(is_error=True, subtype="error_max_tokens")
                break
            messages.append(reply | {"tool_calls": calls})
            for c in calls:
                if c["function"]["name"] != "bash":
                    result = f"Error: no tool named {c['function']['name']!r}."
                else:
                    try:
                        cmd = json.loads(c["function"]["arguments"] or "{}").get("command", "")
                        result = run_wn_lookup(cmd, work)
                    except (json.JSONDecodeError, AttributeError) as e:
                        result = f"Error: bad tool arguments: {e}"
                    except subprocess.TimeoutExpired:
                        result = "Error: command timed out."
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "name": "bash", "content": result})
        else:
            out.update(is_error=True, subtype="error_max_turns")
    out["usage"] = usage
    out["turn_output_tokens"] = turn_tokens
    out["messages"] = messages[1:]
    return json.dumps(out, ensure_ascii=False), time.time() - t0


def call_model(prompt: str, args, stream: bool = False) -> tuple[str, float]:
    if args.backend == "claude":
        return claude(prompt, args.model, stream)
    return openai_agent(prompt, args)


def write_run_meta(raw_dir: Path, args) -> None:
    """Record the harness with the results (README §5: a result is about a
    model with a harness)."""
    meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "backend": args.backend,
            "model": args.model, "workers": args.workers,
            "sample": str(getattr(args, "sample", "")),
            "docs": str(DOCS), "prompt_sha256": prompt_hash()}
    if args.backend == "openai":
        meta |= {k: getattr(args, k) for k in (
            "base_url", "tool_mode", "reasoning", "temperature", "top_p",
            "top_k", "seed", "max_tokens")}
        meta["max_turns"] = MAX_TURNS
        try:
            meta["server_version"] = post_json(
                args.base_url.rstrip("/").removesuffix("/v1") + "/version")
        except (OSError, RuntimeError, json.JSONDecodeError):
            meta["server_version"] = None
    (raw_dir / f"run_meta_{meta['started'].replace(':', '')}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")


def last_json_block(text: str) -> dict | None:
    blocks = re.findall(r"```json\s*\n(.*?)```", text, re.S)
    if not blocks:
        return None
    try:
        return json.loads(blocks[-1])
    except json.JSONDecodeError:
        return None


def strategy_of(devices: list[str]) -> str:
    for d, s in DEVICE_TO_STRATEGY:
        if d in devices:
            return s
    return "inplace"


def result_row(row: dict, model: str, raw: str, secs: float, raw_file: Path) -> dict:
    out = {k: row.get(k, "") for k in ("doc_id", "repair_type", "nl_clean", "sbn_clean")}
    # Base repairs only: the model writes no interregnum (README §1). The
    # sample's assigned interregnum is not used in this round.
    out.update({"interregnum": "none", "model": model, "duration_s": f"{secs:.1f}",
                "raw_file": str(raw_file.relative_to(HERE))})
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError:
        out["parse"] = "bad_cli_output"
        return out
    usage = meta.get("usage") or {}
    out.update({"num_turns": meta.get("num_turns", ""),
                "input_tokens": usage.get("input_tokens", ""),
                "output_tokens": usage.get("output_tokens", ""),
                "cache_read_tokens": usage.get("cache_read_input_tokens", "")})
    if meta.get("is_error"):
        out["parse"] = f"cli_error: {meta.get('subtype', '')}"
        return out
    ans = last_json_block(meta.get("result", ""))
    if ans is None:
        out["parse"] = "no_json"
        return out
    devices = ans.get("devices") or []
    # The model gives the sentence in two parts, so the repair onset (the
    # interruption point) comes from the split, not from a character count by
    # the model. A stray comma at the join is kept as written, for
    # check_repair_part.py to report.
    left = str(ans.get("nl_left", "")).strip()
    right = str(ans.get("nl_right", "")).strip()
    nl_repair, onset = "", ""
    if left and right:
        nl_repair, onset = f"{left}, {right}", len(left) + 2
    out.update({
        "parse": "ok",
        "status": ans.get("status", ""),
        "reason": ans.get("reason", ""),
        "site_index": ans.get("site_index", ""),
        "repair_synset": ans.get("site_synset", ""),
        "reparandum_synset": ans.get("reparandum_synset", ""),
        "reparandum_value": ans.get("reparandum_value", ""),
        "reparandum_surface": ans.get("reparandum_surface", ""),
        "nl_repair": nl_repair,
        "repair_onset": onset,
        "sbn_repair": ans.get("sbn_repair", ""),
        "devices": "|".join(devices) if isinstance(devices, list) else str(devices),
        "strategy": strategy_of(devices) if ans.get("status") == "ok" else "",
    })
    return out


def cmd_run(args) -> None:
    use_docs(args.docs)
    rows = list(csv.DictReader(open(args.sample, encoding="utf-8"), delimiter="\t"))
    if args.only:
        rows = [r for r in rows if r["doc_id"] in set(args.only)]
    safe_model = re.sub(r"[^\w.-]", "_", args.model) + (f".{args.tag}" if args.tag else "")
    out_path = RESULTS / f"{safe_model}.tsv"
    raw_dir = RESULTS / "raw" / safe_model
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_run_meta(raw_dir, args)
    done = set()
    if out_path.exists():
        done = {r["doc_id"] for r in csv.DictReader(open(out_path, encoding="utf-8"),
                                                     delimiter="\t")}
    todo = [r for r in rows if r["doc_id"] not in done][: args.limit or None]
    print(f"{len(done)} done, running {len(todo)} with {args.model}")
    new_file = not out_path.exists()
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        if new_file:
            w.writeheader()
        # Rows are written as they finish, so with --workers > 1 the TSV order
        # differs from the sample; downstream code joins on doc_id.
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(call_model, item_prompt(r), args): r for r in todo}
            for k, fut in enumerate(as_completed(futures), 1):
                row = futures[fut]
                raw_file = raw_dir / (row["doc_id"].replace("/", "_") + ".json")
                try:
                    raw, secs = fut.result()
                except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
                    print(f"[{k}/{len(todo)}] {row['doc_id']} FAILED: {e}", flush=True)
                    continue
                raw_file.write_text(raw, encoding="utf-8")
                res = result_row(row, args.model, raw, secs, raw_file)
                w.writerow(res)
                f.flush()
                print(f"[{k}/{len(todo)}] {row['doc_id']} {row['repair_type']}: "
                      f"parse={res.get('parse')} status={res.get('status', '')} "
                      f"turns={res.get('num_turns', '')} {secs:.0f}s", flush=True)


def cmd_probe(args) -> None:
    raw, secs = call_model(args.prompt, args, stream=args.stream)
    print(raw)
    print(f"\n({secs:.0f}s)", file=sys.stderr)


def add_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="model id; for openai, the server's served model name")
    p.add_argument("--backend", choices=("claude", "openai"), default="claude")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--tool-mode", choices=("auto", "text"), default="auto",
                   help="auto: the server parses tool calls (needs "
                        "--tool-call-parser in vLLM); text: parse ATEM blocks here")
    p.add_argument("--reasoning", default="high",
                   help="Muse Glimmer reasoning strength: low/medium/high/xhigh")
    # Defaults: the Muse Glimmer model card's recommended sampling.
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--seed", type=int, default=13)
    # 16384 in the first Muse Glimmer round, where 2 of 70 items ran out
    p.add_argument("--max-tokens", type=int, default=32768,
                   help="per model turn, reasoning included")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel items; with vLLM, 8 lets the server batch")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    add_backend_args(r)
    r.add_argument("--sample", type=Path, default=SAMPLE,
                   help="sample TSV from select_docs.py (default: the first round's)")
    r.add_argument("--docs", type=Path, default=DOCS,
                   help="prompt documents (default prompt_docs/, as validated in round 2)")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--only", nargs="*")
    r.add_argument("--tag", default="",
                   help="suffix for the results file, e.g. smoke")
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("probe")
    add_backend_args(p)
    p.add_argument("prompt")
    p.add_argument("--stream", action="store_true")
    p.set_defaults(fn=cmd_probe)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
