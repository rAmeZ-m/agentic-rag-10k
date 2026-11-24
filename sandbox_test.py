import subprocess

BASE = ["docker", "run", "--rm", "-i", "--network", "none", "--read-only",
        "--memory", "256m", "--cpus", "0.5", "--pids-limit", "64",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--tmpfs", "/tmp:size=16m", "rag-sandbox", "python", "-"]

CASES = {
    "1 normal pandas": 'import pandas as pd\nprint(pd.DataFrame({"a": [1, 2], "b": [3, 4]}).sum())',
    "2 network (should fail)": 'import urllib.request\ntry:\n    urllib.request.urlopen("http://example.com", timeout=5)\n    print("NETWORK WORKS (bad)")\nexcept Exception as e:\n    print("blocked:", type(e).__name__)',
    "3 read .env (should fail)": 'try:\n    print(open("/home/runner/.env").read())\nexcept Exception as e:\n    print("blocked:", type(e).__name__)',
    "4 write file (should fail)": 'try:\n    open("/home/runner/x.txt", "w").write("hi")\n    print("WRITE WORKS (bad)")\nexcept Exception as e:\n    print("blocked:", type(e).__name__)',
    "5 memory bomb (should be killed)": 'x = b"x" * (900 * 1024 * 1024)\nprint("allocated 900MB (bad)")',
}

print("SANDBOX TEST")
for name, code in CASES.items():
    try:
        r = subprocess.run(BASE, input=code, capture_output=True, text=True, timeout=90)
        out = (r.stdout + r.stderr).strip()
        print(f"\n--- {name} | exit code {r.returncode} ---\n{out[-300:]}")
    except Exception as e:
        print(f"\n--- {name} --- ERROR {type(e).__name__}: {e}")