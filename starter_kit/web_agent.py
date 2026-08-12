#!/usr/bin/env python3
"""LoomQ L2 interactive entry point — a dependency-free web console.

A single-file HTTP server (stdlib only) that wraps adapter.agent_chat() with a
lab-console style chat UI for users with zero quantum background. It also runs
any QASM the agent produces through our own L1 simulator and visualizes the
measurement counts, so "I don't read code" users can still see a result.

Run:
    python web_agent.py [--host 127.0.0.1] [--port 8787]

Routes:
    GET  /            chat page
    GET  /api/health  configuration status (never leaks the API key)
    POST /api/chat    {"prompt": str} -> {"reply": str}
    POST /api/run     {"qasm": str}   -> {"counts": {...}, "qubits": int}
"""

from __future__ import annotations

import argparse
import html
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

try:  # running inside the starter_kit package
    from .loomq_core import parse_qasm, sample_counts
    from . import adapter as adapter_mod
except ImportError:  # running as a top-level script
    from loomq_core import parse_qasm, sample_counts
    import adapter as adapter_mod

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LoomQ 量子对话台 · 实验室控制台</title>
<style>
  :root {
    --bg: #0a0e13;
    --panel: #10161d;
    --panel-2: #0d1219;
    --line: #1d2732;
    --ink: #d9e2ea;
    --ink-dim: #7c8a99;
    --accent: #41e0c0;
    --accent-2: #ffb454;
    --ok: #3ecf8e;
    --warn: #ffb454;
    --err: #ff6b6b;
    --code-bg: #070a0f;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    background:
      linear-gradient(rgba(65,224,192,.035) 1px, transparent 1px),
      linear-gradient(90deg, rgba(65,224,192,.035) 1px, transparent 1px),
      radial-gradient(1200px 600px at 70% -10%, rgba(65,224,192,.10), transparent 60%),
      var(--bg);
    background-size: 44px 44px, 44px 44px, auto, auto;
    color: var(--ink);
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    letter-spacing: .02em;
  }
  .console {
    max-width: 920px;
    margin: 0 auto;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    padding: 18px 20px 28px;
  }
  /* ---- instrument status bar ---- */
  .status {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 10px 14px;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: var(--panel-2);
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 12px;
    color: var(--ink-dim);
    flex-wrap: wrap;
  }
  .led {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--err);
    box-shadow: 0 0 8px var(--err);
    animation: pulse 1.6s infinite;
    flex: none;
  }
  .led.on { background: var(--ok); box-shadow: 0 0 8px var(--ok); animation: none; }
  @keyframes pulse { 50% { opacity: .35; } }
  .status .sep { opacity: .35; }
  .brand {
    font-family: Georgia, "Songti SC", serif;
    font-size: 15px; letter-spacing: .14em; color: var(--ink);
  }
  .brand b { color: var(--accent); font-weight: 600; }
  .meta { margin-left: auto; }

  /* ---- chat ---- */
  .chat { flex: 1; padding: 22px 2px; overflow-y: auto; }
  .msg { display: flex; gap: 12px; margin-bottom: 20px; animation: rise .28s ease both; }
  @keyframes rise { from { opacity: 0; transform: translateY(6px); } }
  .avatar {
    width: 32px; height: 32px; border-radius: 50%; flex: none;
    display: grid; place-items: center; font-size: 15px;
    border: 1px solid var(--line);
  }
  .msg.user { flex-direction: row-reverse; }
  .msg.user .avatar { background: var(--accent-2); color: #201500; }
  .msg.bot .avatar { background: var(--panel); color: var(--accent); }
  .bubble {
    max-width: 78%;
    padding: 11px 15px;
    border-radius: 10px;
    background: var(--panel);
    border: 1px solid var(--line);
    line-height: 1.65; font-size: 14.5px; white-space: pre-wrap; word-break: break-word;
  }
  .msg.user .bubble { background: #16222c; border-color: #24313c; }
  pre.qasm {
    background: var(--code-bg);
    border: 1px solid #15202a;
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 10px 12px;
    margin: 8px 0 4px;
    overflow-x: auto;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 12.5px; line-height: 1.55; color: #a9e6d8;
  }
  .backend-tag {
    display: inline-block;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 12.5px;
    color: var(--accent-2);
    border: 1px dashed rgba(255,180,84,.45);
    border-radius: 4px;
    padding: 2px 8px;
  }
  .hint { color: var(--ink-dim); font-size: 13px; }
  .err .bubble { border-color: rgba(255,107,107,.5); color: #ffc9c9; }
  .loading .bubble { color: var(--ink-dim); }
  .loading .dots::after { content: ""; animation: dots 1.2s steps(4) infinite; }
  @keyframes dots { 0% {content:""} 25% {content:"."} 50% {content:".."} 75% {content:"..."} }

  /* ---- result viz ---- */
  .viz { margin-top: 8px; }
  .viz h4 {
    margin: 6px 0 10px; font-size: 12px; letter-spacing: .18em;
    color: var(--ink-dim); font-weight: 500;
    border-bottom: 1px solid var(--line); padding-bottom: 6px;
  }
  .bar-row { display: grid; grid-template-columns: 74px 1fr 64px; gap: 10px; align-items: center; margin-bottom: 5px; }
  .bar-key { font-family: "Cascadia Mono", "Consolas", monospace; font-size: 12px; color: var(--ink-dim); text-align: right; }
  .bar-track { height: 14px; background: #0a0f15; border: 1px solid var(--line); border-radius: 3px; overflow: hidden; }
  .bar-fill { height: 100%; background: linear-gradient(90deg, #2ba88f, var(--accent)); width: 0; transition: width .5s ease; }
  .bar-count { font-family: "Cascadia Mono", "Consolas", monospace; font-size: 12px; color: var(--accent); }

  /* ---- chips ---- */
  .chips { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
  .chip {
    font-size: 13px; color: var(--ink-dim);
    border: 1px solid var(--line); border-radius: 999px;
    padding: 7px 14px; background: var(--panel-2); cursor: pointer;
    transition: all .18s ease;
  }
  .chip:hover { color: var(--accent); border-color: var(--accent); transform: translateY(-1px); }

  /* ---- input ---- */
  .inputbar { display: flex; gap: 10px; }
  .inputbar input {
    flex: 1;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 13px 16px;
    color: var(--ink);
    font-size: 14.5px;
    outline: none;
    transition: border-color .18s ease, box-shadow .18s ease;
  }
  .inputbar input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(65,224,192,.12); }
  .inputbar input::placeholder { color: #4c5a68; }
  .send {
    border: none; border-radius: 8px; padding: 0 22px;
    background: var(--accent); color: #04211b;
    font-size: 14px; font-weight: 700; letter-spacing: .06em;
    cursor: pointer; transition: filter .18s ease, transform .1s ease;
  }
  .send:hover { filter: brightness(1.12); }
  .send:active { transform: scale(.97); }
  .send:disabled { opacity: .45; cursor: not-allowed; }
  .foot { margin-top: 12px; text-align: center; font-size: 11.5px; color: #3e4c59; letter-spacing: .05em; }
</style>
</head>
<body>
<div class="console">
  <div class="status">
    <span class="led" id="led"></span>
    <span class="brand">LOOM<b>Q</b> · 量子对话台</span>
    <span class="sep">|</span>
    <span>MODEL <span id="model" class="hint">--</span></span>
    <span class="sep">|</span>
    <span>HOST <span id="host" class="hint">--</span></span>
    <span class="meta" id="status-text">正在检查模型服务…</span>
  </div>

  <div class="chat" id="chat">
    <div class="msg bot">
      <div class="avatar">⌁</div>
      <div class="bubble">你好，我是 LoomQ 量子助手——一个把"量子黑话"翻译成人话的翻译器。
你不需要懂量子计算，直接用中文说你想做什么，我来帮你写成量子程序、跑出结果。

<b>试试下面任意一个示例：</b>
        <div class="chips">
          <button class="chip" data-q="生成一个 3 比特的最大纠缠态（GHZ 态），并进行全测量">生成 3 比特 GHZ 态</button>
          <button class="chip" data-q="我想制备一个贝尔态，但这段代码报错了，帮我修好：H q[0]; CX q[0] q[1]">修复报错的贝尔态代码</button>
          <button class="chip" data-q="我需要运行一个 15 比特电路，且零排队等待，选哪个平台？">为 15 比特电路选后端</button>
        </div>
      </div>
    </div>
  </div>

  <div class="inputbar">
    <input id="in" type="text" placeholder="用大白话描述你的量子需求…" autocomplete="off">
    <button class="send" id="send">发送</button>
  </div>
  <div class="foot">LoomQ Quantum Accessibility Initiative · 全程本地运行，模型配置来自 LOOMQ_LLM_* 环境变量</div>
</div>

<script>
const chat = document.getElementById("chat");
const input = document.getElementById("in");
const sendBtn = document.getElementById("send");

function esc(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
function detectBackend(text) {
  const m = text.match(/[a-z0-9_]*(?:simulator|backend|device|spinq|originq|braket)[a-z0-9_]*/i);
  return m ? m[0] : null;
}
function renderQasm(code) {
  return `<pre class="qasm">${esc(code)}</pre>`;
}
function addMsg(role, bodyHtml, extra) {
  const div = document.createElement("div");
  div.className = "msg " + role + (extra && extra.cls ? " " + extra.cls : "");
  div.innerHTML = `<div class="avatar">${role === "user" ? "你" : "⌁"}</div>
    <div class="bubble">${bodyHtml}</div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}
function parseReply(text) {
  // escape, then re-inject qasm blocks as highlighted <pre>
  const parts = text.split(/```(?:qasm)?\\s*/i);
  let out = esc(parts[0]);
  for (let i = 1; i < parts.length; i++) {
    if (i % 2 === 1) { out += renderQasm(parts[i]); }
    else { out += esc(parts[i]); }
  }
  const tag = detectBackend(text);
  if (tag) out += `<div style="margin-top:8px">推荐后端：<span class="backend-tag">${esc(tag)}</span></div>`;
  return out;
}
function extractQasm(text) {
  const m = text.match(/```(?:qasm)?\\s*([\\s\\S]*?OPENQASM 2\\.0;[\\s\\S]*?)```/i);
  return m ? m[1] : null;
}
function addViz(qasm) {
  const box = document.createElement("div");
  box.className = "msg bot";
  box.innerHTML = `<div class="avatar">⌁</div><div class="bubble hint">正在用自研模拟器运行这段量子程序…</div>`;
  chat.appendChild(box);
  fetch("/api/run", { method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({qasm}) })
    .then(r => r.json())
    .then(data => {
      if (data.error) { box.innerHTML = `<div class="avatar">⌁</div><div class="bubble err">运行失败：${esc(data.error)}</div>`; return; }
      const total = Object.values(data.counts).reduce((a,b)=>a+b,0);
      const top = Object.entries(data.counts).sort((a,b)=>b[1]-a[1]).slice(0,8);
      let rows = "";
      for (const [k,v] of top) {
        const pct = (v/total*100).toFixed(1);
        rows += `<div class="bar-row">
          <div class="bar-key">${esc(k)}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${pct*2.6}%"></div></div>
          <div class="bar-count">${v} · ${pct}%</div></div>`;
      }
      box.innerHTML = `<div class="avatar">⌁</div><div class="bubble">
        <div class="viz"><h4>测量结果分布 · ${data.qubits} 比特 · ${total} 次采样</h4>${rows}</div></div>`;
    })
    .catch(e => { box.innerHTML = `<div class="avatar">⌁</div><div class="bubble err">可视化失败：${esc(String(e))}</div>`; });
}
function send(prompt) {
  if (!prompt.trim() || sendBtn.disabled) return;
  addMsg("user", esc(prompt));
  input.value = "";
  const wait = addMsg("bot", '<span class="dots">正在请模型推理并自检</span>', {cls:"loading"});
  sendBtn.disabled = true;
  fetch("/api/chat", { method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({prompt}) })
    .then(r => r.json())
    .then(data => {
      wait.remove();
      if (data.error) { addMsg("bot", `<span class="hint">${esc(data.error)}</span>`, {cls:"err"}); return; }
      addMsg("bot", parseReply(data.reply));
      const qasm = extractQasm(data.reply);
      if (qasm) addViz(qasm);
    })
    .catch(e => { wait.remove(); addMsg("bot", `<span class="hint">请求失败：${esc(String(e))}</span>`, {cls:"err"}); })
    .finally(() => { sendBtn.disabled = false; });
}
sendBtn.onclick = () => send(input.value);
input.addEventListener("keydown", e => { if (e.key === "Enter") send(input.value); });
document.querySelectorAll(".chip").forEach(c => c.onclick = () => send(c.dataset.q));

fetch("/api/health").then(r => r.json()).then(h => {
  document.getElementById("led").classList.toggle("on", h.ready);
  document.getElementById("model").textContent = h.model || "--";
  document.getElementById("host").textContent = h.host || "--";
  document.getElementById("status-text").textContent = h.ready
    ? "模型服务已就绪 · 可开始对话" : "未检测到 LOOMQ_LLM_* 配置，示例的后端推荐仍可用";
});
</script>
</body>
</html>
"""


def _extract_qasm(reply: str) -> Optional[str]:
    fences = re.findall(r"```(?:qasm)?\s*(.*?)```", reply, re.DOTALL)
    for candidate in fences:
        if re.search(r"OPENQASM\s+2\.0;", candidate):
            return candidate.strip()
    match = re.search(r"OPENQASM\s+2\.0;.*", reply, re.DOTALL)
    return match.group(0).strip() if match else None


def _health() -> Dict[str, Any]:
    import os
    base = os.environ.get("LOOMQ_LLM_BASE_URL", "")
    model = os.environ.get("LOOMQ_LLM_MODEL", "")
    ready = bool(base and os.environ.get("LOOMQ_LLM_API_KEY") and model)
    return {"ready": ready, "model": model, "host": _host_from(base)}


def _host_from(base: str) -> str:
    m = re.match(r"https?://([^/]+)", base)
    return m.group(1) if m else (base or "--")


class Handler(BaseHTTPRequestHandler):
    server_version = "LoomQConsole/1.0"

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: Dict[str, Any]) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/api/health":
            self._json(200, _health())
            return
        self._json(404, {"error": "not found"})

    def _read_json(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8")), None
        except Exception as exc:  # noqa: BLE001
            return None, f"bad request body: {exc}"

    def do_POST(self) -> None:
        if self.path == "/api/chat":
            data, err = self._read_json()
            if err:
                self._json(400, {"error": err})
                return
            prompt = (data or {}).get("prompt", "")
            try:
                reply = adapter_mod.agent_chat(str(prompt))
                self._json(200, {"reply": reply})
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                self._json(500, {"error": msg})
            return
        if self.path == "/api/run":
            data, err = self._read_json()
            if err:
                self._json(400, {"error": err})
                return
            qasm = _extract_qasm((data or {}).get("qasm", ""))
            if not qasm:
                self._json(400, {"error": "no OpenQASM 2.0 program found"})
                return
            try:
                circuit = parse_qasm(qasm)
                counts = sample_counts(circuit, shots=8192)
                self._json(200, {"counts": counts, "qubits": circuit.num_qubits})
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._json(404, {"error": "not found"})

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="LoomQ L2 interactive web console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"LoomQ 量子对话台已启动: {url}")
    print("提示: 网页/桌面均可访问；退出按 Ctrl+C。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
