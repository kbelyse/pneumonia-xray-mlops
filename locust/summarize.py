"""Turns the Locust csv files into one table and one chart."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).parent / "results"

rows = []
for f in sorted(RESULTS.glob("*_containers_*u_stats.csv")):
    n = int(f.name.split("_")[0])
    users = int(f.name.split("_")[2].rstrip("u"))
    s = pd.read_csv(f)
    p = s[s.Name == "/predict"].iloc[0]
    rows.append({"users": users, "containers": n, "requests": int(p["Request Count"]), "failures": int(p["Failure Count"]),
                 "requests_per_s": round(p["Requests/s"], 1), "median_ms": p["50%"], "avg_ms": round(p["Average Response Time"]),
                 "p95_ms": p["95%"], "p99_ms": p["99%"], "max_ms": round(p["Max Response Time"])})
table = pd.DataFrame(rows).sort_values(["users", "containers"])
table.to_csv(RESULTS / "summary.csv", index=False)
print(table.to_markdown(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for users, g in table.groupby("users"):
    line, = axes[0].plot(g.containers, g.median_ms, "o-", label=f"{users} users, median")
    axes[0].plot(g.containers, g.p95_ms, "o--", color=line.get_color(), label=f"{users} users, 95th percentile")
    axes[1].plot(g.containers, g.requests_per_s, "o-", label=f"{users} users")
axes[0].set(xlabel="api containers", ylabel="response time (ms)", title="/predict response time", xticks=[1, 2, 4])
axes[1].set(xlabel="api containers", ylabel="requests per second", title="Throughput", xticks=[1, 2, 4])
axes[0].legend(fontsize=8)
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(RESULTS / "locust_summary.png", dpi=120)
