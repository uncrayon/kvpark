"""Render corpus conversations into a model's capture records (collect_v2.py --docs, fit_hub.py records).

    python render_records.py --corpus CONVERSATIONS.jsonl --gguf MODEL.gguf --model-id NAME --split train \
        --output RECORDS.jsonl [--server-render] [--limit N] [--per-source K]

Each conversation is rendered as pair_eval renders it (pair_eval.render: the model's template from its GGUF,
chat.py; --server-render as llama-server renders a chat), with the given split. --per-source keeps at most K
conversations of each corpus "source" value and --limit at most N in all, in corpus order. Conversations a
template refuses are reported and left out.
"""
import argparse
import json
from pathlib import Path
import sys

import pair_eval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "linear_transfer"))
import chat  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--split", choices=sorted(chat.SPLITS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-render", action="store_true")
    parser.add_argument("--ids", help="comma list of conversation ids to keep (default: all)")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    wanted = set(args.ids.split(",")) if args.ids else None
    out, refused = [], []
    for line in args.corpus.read_text(encoding="utf-8").splitlines():
        conversation = json.loads(line)
        if wanted is not None and conversation.get("id") not in wanted:
            continue
        try:
            record = pair_eval.render(conversation, args.gguf, args.server_render)
        except ValueError as error:
            refused.append({"id": conversation.get("id"), "reason": str(error)[:160]})
            continue
        record.update(model=args.model_id, split=args.split, source=conversation.get("source", "corpus"),
                      conversation_id=conversation.get("id"))
        out.append(record)
        if args.limit and len(out) >= args.limit:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(json.dumps({"records": len(out), "refused": refused}))


if __name__ == "__main__":
    main()
