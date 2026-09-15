"""Interactive byte-in / byte-out demo on the connectome, with live activation, served locally.

WHAT THIS IS NOT: a language model. Nothing here is trained. The input and output mappings below are
engineered choices, like those in the connectome game demos, not fly biology, and the readout is a FIXED
RANDOM projection. The emitted bytes are therefore arbitrary by construction. What is real is the middle:
the same steady-state solve, weights and signs as the rest of this repo, on the real MaleCNS wiring.

  input   last CONTEXT bytes drive sensory neurons: byte value v -> PORT sensory neurons drawn for v
          (seed 0, sampled without replacement per value), amplitude 0.6 ** age
  network h = ReLU(g W h + b + u), g = 0.9, b = 0.1, W[post, pre] = sign(pre) * w / in_tot_full[post]
  output  descending-neuron response (h - h_rest) -> 256 logits through a fixed random matrix (seed 1),
          sampled at the given temperature
  display per-neuron |response| as uint8, for the neurons that have a somaLocation

Serves scripts/viewer.html at http://127.0.0.1:PORT/ and answers:
  GET  /meta  model settings and neuron counts (JSON)
  GET  /pos   neuron positions, float32 xyz normalised to [-1, 1] (octet-stream)
  POST /step  {"text": str, "temperature": float} -> {"byte", "text", "act" (base64 uint8), "iters", "ms"}

Run from data/ after retention.py:  viz_serve.py [--rule=or1|baseline] [--port=8765] [--self-test]
--self-test: load, run two steps, print timings, and exit without serving.
"""
import base64, http.server, json, os, random, socketserver, sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp

arg = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{k}=")), d)
RULE, PORT = arg("rule", "or1"), int(arg("port", 8765))
SELF_TEST = "--self-test" in sys.argv
G, B, TOL, MAXIT = 0.9, 0.1, 1e-6, 2000
PORT_N, CONTEXT, DECAY = 64, 8, 0.6
HERE = os.path.dirname(os.path.abspath(__file__))
assert RULE in ("or1", "baseline")

print(f"loading {RULE} ...", flush=True)
ann = pd.read_parquet("annotations.parquet", columns=["bodyId", "superclass", "class", "somaLocation"])
ann = ann[ann.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
N = len(ann)
nt = pd.read_parquet("nt.parquet", columns=["body", "consensus_nt"]).set_index("body").consensus_nt
cons = ann.bodyId.map(nt).fillna("missing").astype(str).to_numpy()
SIGN = {"acetylcholine": 1, "dopamine": 1, "serotonin": 1, "octopamine": 1,
        "gaba": -1, "glutamate": -1, "histamine": -1}
sign = np.array([SIGN.get(c, 1) for c in cons], np.float32)

d = np.load("nn_edges.npz"); pre, post, w = d["pre"], d["post"], d["w"].astype(np.float64)
in_tot = np.bincount(post, weights=w, minlength=N)
out_tot = np.bincount(pre, weights=w, minlength=N)
m = w >= 3
if RULE == "or1": m &= (w / in_tot[post] >= 0.01) | (w / out_tot[pre] >= 0.01)
W = sp.csr_matrix(((G * sign[pre[m]] * w[m] / in_tot[post[m]]).astype(np.float32), (post[m], pre[m])),
                  shape=(N, N))
del pre, post, w, d

sc = ann.superclass.astype(str)
sens = np.flatnonzero(sc.str.contains("sensory").to_numpy())
dn = np.flatnonzero(sc.eq("descending_neuron").to_numpy())
loc = ann.somaLocation.fillna("").astype(str)
has = np.flatnonzero(loc.str.startswith("[").to_numpy())
xyz = np.array([np.fromstring(s.strip("[]"), sep=" ") for s in loc.iloc[has]], np.float32)
xyz = (xyz - xyz.mean(0)) / np.abs(xyz - xyz.mean(0)).max()

rng = np.random.default_rng(0)
ports = np.stack([rng.choice(sens, PORT_N, replace=False) for _ in range(256)])   # byte value -> input ports
W_out = np.random.default_rng(1).normal(0, 1, (len(dn), 256)).astype(np.float32) / np.sqrt(len(dn))


def solve(u):
    h = np.zeros(N, np.float32)
    for it in range(1, MAXIT + 1):
        hn = np.maximum(W @ h + B + u, 0.0)
        delta = np.abs(hn - h).max(); h = hn
        if delta < TOL: return h, it
    raise RuntimeError(f"no convergence in {MAXIT} iterations (last change {delta:.3g})")


h_rest, rest_iters = solve(np.zeros(N, np.float32))
print(f"{RULE}: {int(m.sum())} edges, rest state in {rest_iters} iterations, "
      f"{len(sens)} sensory, {len(dn)} descending, {len(has)} placed neurons", flush=True)


def step(text, temperature=1.0):
    t0 = time.time()
    u = np.zeros(N, np.float32)
    tail = text.encode("utf-8", "replace")[-CONTEXT:]
    for age, v in enumerate(reversed(tail)): u[ports[v]] += DECAY ** age
    h, iters = solve(u)
    r = h - h_rest
    logits = r[dn] @ W_out
    p = np.exp((logits - logits.max()) / max(temperature, 1e-3))
    byte = int(np.random.default_rng(random.randrange(1 << 30)).choice(256, p=p / p.sum()))
    a = np.abs(r[has])
    act = (255 * a / max(a.max(), 1e-12)).astype(np.uint8)
    return {"byte": byte, "act": base64.b64encode(act.tobytes()).decode(), "iters": iters,
            "ms": round(1000 * (time.time() - t0)), "responding": int((a > 1e-3 * a.max()).sum())}


META = {"rule": RULE, "gain": G, "b": B, "tol": TOL, "edges": int(m.sum()), "neurons": N,
        "placed": len(has), "sensory": len(sens), "descending": len(dn), "port_neurons": PORT_N,
        "context_bytes": CONTEXT, "trained": False,
        "note": "untrained; input and output mappings are engineered, not biological"}


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(open(f"{HERE}/viewer.html", "rb").read(), "text/html; charset=utf-8")
        elif self.path == "/meta":
            self._send(json.dumps(META).encode(), "application/json")
        elif self.path == "/pos":
            self._send(xyz.tobytes(), "application/octet-stream")
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        if self.path != "/step":
            return self._send(b"not found", "text/plain", 404)
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        self._send(json.dumps(step(req.get("text", ""), float(req.get("temperature", 1.0)))).encode(),
                   "application/json")

    def log_message(self, *a): pass


if SELF_TEST:
    text = "안녕"
    for i in range(2):
        r = step(text)
        text += bytes([r["byte"]]).decode("utf-8", "replace")
        print(f"step {i + 1}: byte {r['byte']:3d} | {r['iters']} iterations | {r['ms']} ms | "
              f"{r['responding']} neurons responding | activation {len(base64.b64decode(r['act']))} bytes")
    print("self-test ok"); sys.exit(0)

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as srv:
    print(f"serving http://127.0.0.1:{PORT}/  (ctrl-c to stop)", flush=True)
    srv.serve_forever()
