"""Summarize correctness cases: bit-identity, and classify every pause that did not complete."""
import glob, json, sys

rows = []
for p in sorted(glob.glob("results/*.json")):
    if "/fault_" in p or "bench_size" in p:
        continue
    r = json.load(open(p))
    if "error" in r:
        rows.append((r["tag"], "ERROR", r["error"]))
        continue
    total = r["n_ref_rows"]
    bad = []
    for e in r["events"]:
        if e.get("error"):
            bad.append("target finished before this pause point")
        elif not (e["pause"].get("state") == "checkpointed" and e["resume"].get("state") == "running"):
            kind = "after the job's last step" if e["at_line"] >= total else f"MID-RUN at line {e['at_line']}"
            bad.append(f"{kind}: pause={e['pause']} resume={e['resume']}")
    done = sum(1 for e in r["events"] if not e.get("error") and e["pause"].get("state") == "checkpointed"
               and e["resume"].get("state") == "running")
    rows.append((r["tag"], r["driver"]["gpu"].split(",")[1].strip(), r["ref_deterministic"],
                 r["rows_identical_to_ref"] and r["end_identical_to_ref"], f"{done}/{r['config']['pauses']}",
                 r["median_step_s_before_first_pause"], r["median_step_s_after_last_resume"], bad))
for x in rows:
    print(json.dumps(x))
