"""End-to-end agent_chat test with a local mock OpenAI-compatible server."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, ".")

QASM_GHZ = """OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\nh q[0];\ncx q[0], q[1];\ncx q[1], q[2];\nmeasure q -> c;\n"""


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        user_msgs = [m for m in payload["messages"] if m["role"] == "user"]
        last_user = user_msgs[-1]["content"] if user_msgs else ""
        prompt = last_user
        # Simulate a model that emits a GHZ qasm for generation/repair tasks,
        # a canonical backend id for selection tasks, and a PASS verdict for
        # LLM-based semantic confirmation round-trips.
        if "模拟器测量分布" in prompt:
            content = "PASS"
        elif "backend" in prompt or "平台" in prompt or "排队" in prompt:
            content = "推荐使用 AWS Braket 本地模拟器，规范标识：braket_local_simulator"
        else:
            content = "```qasm\n" + QASM_GHZ + "\n```"
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    os.environ.update(
        {
            "LOOMQ_LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}",
            "LOOMQ_LLM_API_KEY": "local-test-key",
            "LOOMQ_LLM_MODEL": "deepseek-v4-flash",
            "LOOMQ_LLM_TIMEOUT_SECONDS": "30",
        }
    )
    try:
        import adapter

        print("=== 任务1: 意图生成 ===")
        r1 = adapter.agent_chat("生成一个 3 比特 GHZ 态并进行全测量")
        print("回复包含QASM:", "OPENQASM" in r1)

        print("=== 任务2: 代码纠错 ===")
        r2 = adapter.agent_chat("我想制备贝尔态但代码报错，帮我修好：H q[0]; CX q[0] q[1]")
        print("回复包含QASM:", "OPENQASM" in r2)

        print("=== 任务3: 智能选后端 ===")
        r3 = adapter.agent_chat("我需要运行一个 15 比特电路，且零排队等待，选哪个平台？")
        print("回复包含规范标识:", "braket_local_simulator" in r3)

        print("=== 回归1: 生成任务不被误判为选后端 ===")
        r4 = adapter.agent_chat("生成一个 3 比特的 GHZ 态并进行全测量")
        print("走生成分支(含QASM且非后端id):", "OPENQASM" in r4 and "braket_local_simulator" not in r4)

        print("=== 回归2: 未知目标态走 LLM 语义确认 ===")
        r5 = adapter.agent_chat("制备一个 3 比特 W 态")
        print("LLM确认后返回QASM:", "OPENQASM" in r5)

        print("=== 缺配置立即失败(不泄露Key) ===")
        import l2_agent
        saved = {k: os.environ.pop(k, None) for k in ("LOOMQ_LLM_BASE_URL", "LOOMQ_LLM_API_KEY", "LOOMQ_LLM_MODEL")}
        try:
            try:
                l2_agent.agent_chat("hello")
                print("FAIL: 应该抛错")
            except RuntimeError as e:
                print("OK 抛出 RuntimeError, 且不含Key:", "local-test-key" not in str(e))
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
