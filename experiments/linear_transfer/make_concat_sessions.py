"""Write a derived SmolTalk2 subset of long chat sessions made by concatenating multi-turn chats.

Each derived row joins 4-9 rows of multi_turn_reasoning_if_think (user/assistant turns with
reasoning), in seeded order, into one session, the way a long assistant thread spans several
topics. Rows already used by the given corpora are skipped, so no derived session reuses a
training, dev or test conversation. The output is a parquet file named like the pinned subsets
(`multi_turn_concat_sessions-00000-of-00001.parquet`) so build_corpus.py can draw from it, plus a
JSON sidecar recording the source rows of every derived row.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE = "multi_turn_reasoning_if_think"
DERIVED = "multi_turn_concat_sessions"


def used_rows(corpora):
    rows = set()
    for path in corpora:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                identifier = json.loads(line)["id"]
                subset, _, row = identifier.partition(":")
                if subset == SOURCE:
                    rows.add(int(row))
    return rows


def main(argv=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="conversations.jsonl whose source rows must not be reused (repeatable)")
    parser.add_argument("--sessions", type=int, default=600)
    parser.add_argument("--min-rows", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args(argv)
    if not 2 <= args.min_rows <= args.max_rows:
        raise SystemExit("need 2 <= --min-rows <= --max-rows")
    output = args.dataset_dir / f"{DERIVED}-00000-of-00001.parquet"
    sidecar = output.with_suffix(".sources.json")
    if output.exists() or sidecar.exists():
        raise SystemExit(f"output already exists: {output}")
    source_path = next(args.dataset_dir.glob(f"{SOURCE}-*.parquet"))
    table = pq.read_table(source_path, columns=["messages", "chat_template_kwargs", "source"])
    excluded = used_rows(args.exclude)
    rng = np.random.default_rng(args.seed)
    order = [int(i) for i in rng.permutation(table.num_rows) if int(i) not in excluded]
    sessions, sources, position = [], [], 0
    for _ in range(args.sessions):
        count = int(rng.integers(args.min_rows, args.max_rows + 1))
        if position + count > len(order):
            break
        picked = order[position:position + count]
        position += count
        rows = table.take(picked).to_pylist()
        messages = [message for row in rows for message in row["messages"]]
        if any(message["role"] == "system" for message in messages):
            continue
        sessions.append({"messages": messages, "chat_template_kwargs": rows[0]["chat_template_kwargs"],
                         "source": DERIVED})
        sources.append(picked)
    derived = pa.Table.from_pylist(sessions, schema=table.schema)
    pq.write_table(derived, output)
    sidecar.write_text(json.dumps({
        "derived_from": source_path.name,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "excluded_rows": len(excluded), "seed": args.seed, "min_rows": args.min_rows, "max_rows": args.max_rows,
        "rows": sources}) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sessions": len(sessions), "excluded_source_rows": len(excluded)}))


if __name__ == "__main__":
    main()
