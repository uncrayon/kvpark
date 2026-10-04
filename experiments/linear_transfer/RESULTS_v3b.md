# Qwen → Muse v3b: long chats and agent traces

The v3b run finished on October 1, 2026. It asked the question v3 could not answer:
is the 8K+ failure in [RESULTS_v3.md](RESULTS_v3.md) about **length** or about **the
kind of content**? In v3, every 4K–8K document was chat and every 8K+ document was
long-document Q&A (LongAlign).

v3b repeats the same handoff (Design A: mapped old region [0, T0), a native tail of W
tokens, table fill, v3 mapper) on 24 held-out conversations of 8.3K–27.2K Muse tokens
whose content is chat or agent work, not long documents:

- **real multi-turn chats:** SmolTalk2 `multi_turn_reasoning_if_think`, the same source as
  the v3 4K–8K bucket (3 scored; long ones are rare in this subset);
- **concatenated sessions:** 4–9 multi-turn chats joined into one long thread, like an
  assistant session that spans several topics (`make_concat_sessions.py`);
- **agent traces:** SmolTalk2 `smolagents_toolcalling_traces_think`, tool calls and
  their outputs (web search results, fetched pages), then a final answer.

## Answer: content, not length

Pre-registered headline (W = 2048, v3 mapper, table fill), means weighted by scored
token as in v3:

| Content | Docs | Prefix (Muse tokens) | ΔNLL, headline | Information-free floor | Share of the gap recovered |
| --- | ---: | ---: | ---: | ---: | ---: |
| Real multi-turn chat | 3 | 10.1K–12.9K | **+0.029 [0.020, 0.044], meets target** | +0.141 | 79% |
| Concatenated sessions | 12 | 8.3K–27.2K | +0.064 [0.052, 0.078] | +0.106 | 40% |
| … 8K–16K | 5 | | +0.056 [0.042, 0.071] | +0.101 | 44% |
| … 16K–27K | 7 | | +0.069 [0.054, 0.088] | +0.110 | 37% |
| Agent traces (tool outputs) | 9 | 8.7K–27.1K | **+0.300 [0.175, 0.435], fails** | +0.804 | 63% |
| All 24 | 24 | | +0.093 [0.065, 0.134] | +0.217 | 57% |

(Floor: `mean_old_2048`, every handed-over cell the train content mean, same native tail.)

- **Chat survives at 8K+.** Real multi-turn chats at 10–13K behave like the 4–8K chats of
  v3 (+0.029 vs +0.021 there). The handoff cost does not grow with length: the slope of
  ΔNLL against prefix length over all 24 documents is +0.005 per 1K tokens
  [−0.010, +0.020].
- **Agent traces fail like long documents.** The mapped cache recovers 63% of what the
  old context contributes, but the old context matters a lot here (floor +0.80), so the
  remaining loss (+0.30) is six times the target. A 4,096-token tail does not rescue it
  (+0.256). In these traces 87–96% of the handed-over region is tool output (search
  results and web pages), so the content is document-like, the same failure mode as
  LongAlign Q&A in v3.
- **Concatenated sessions sit near the target but miss it.** They lose +0.064, and with
  W = 4096 +0.050 [0.040, 0.064]. Their old context carries little information for the
  final reply (floor only +0.106), so the mapped cache costs almost as much as it
  recovers: noisy mapped keys/values add a small cost of their own (~0.05) even where
  the conversation does not need them.

Combined with v3: the deciding variable is how much of the old region is prose
conversation versus document-like text the model must retrieve from precisely.

| Condition (all 24 documents) | ΔNLL | Top-1 agreement |
| --- | ---: | ---: |
| W = 4096, table fill, v3 mapper | +0.077 [0.053, 0.114] | 92.5% |
| **W = 2048, table fill, v3 mapper (headline)** | **+0.093 [0.066, 0.134]** | 91.0% |
| W = 2048, table fill, v2 mapper | +0.098 [0.067, 0.145] | 90.2% |
| W = 2048, information-free (table fill) | +0.227 | 86.1% |
| W = 2048, information-free (mean everywhere) | +0.217 | 87.1% |
| W = 0, table fill, v3 mapper | +0.281 [0.223, 0.346] | 80.9% |
| W = 0, information-free | +0.797 | 68.5% |

## Where agent traces lose it

A follow-up on the 9 agent traces (`score_extra.py`, same states, probe and settings;
its table-fill condition reproduces the run's numbers exactly) gave Muse's native values
to one class of handed-over cells at a time and kept the deployable handoff everywhere
else (W = 2048, v3 mapper):

| Native values for | Share of handed-over cells | ΔNLL |
| --- | ---: | ---: |
| nothing (the deployable handoff) | 0% | +0.300 [0.175, 0.435] |
| **fill cells: template, special, unmatched, straddling tokens** | **~9%** | **+0.103 [0.046, 0.161]** |
| system message and rendered tool schemas | ~3% (579 tokens) | +0.299 [0.170, 0.434] |
| tool outputs (content of every tool message) | ~90% | +0.081 [0.031, 0.158] |
| tool schemas and tool outputs | ~93% | +0.076 [0.029, 0.141] |

- **Two-thirds of the agent loss is in the fill.** Native values in just the ~9% of cells
  that the token table fills cut the loss from +0.300 to +0.103 (paired difference
  −0.196 [−0.284, −0.124]). That is about as much as making all tool output native.
  Agent traces have far more fill cells than chat: 4.8% template/special (tool-call and
  tool-response markup at every step) and 3.8% unmatched (tool-call JSON and tool output
  that the two chat templates render differently), against 1.6% and 1.9% in the
  concatenated chats. About 27% of the agent fill cells hold token ids the table has
  never seen.
- **The rest is genuine mapping error on tool output** (about +0.10, the remaining loss
  with a native fill), the same kind of loss as LongAlign documents, where a native
  fill does not help at all (run-v3: oracle fill ≈ table fill on every LongAlign
  document; they are ~98% aligned content).
- **The system prompt and tool schemas are not the problem** here (+0.299 vs +0.300).
  In these traces they are only 579 tokens; Hermes' own system prompt and tool schemas
  are much larger, so this does not settle the native-prefix question for Hermes.

So the cheapest lever for agent conversations is a better fill for template and
unmatched cells, not a better content mapper: for example, a context-aware table
(token id plus role or neighbouring token), aligning template tokens by message
structure and mapping them, and mapping unmatched tokens from the Qwen tokens that cover
the same bytes. Even a perfect fill leaves agent traces at about +0.10, twice the
target.

## Checks

- Restoring Muse's own native state at T0 on GPU gave KL exactly 0 on all 48 checks.
- Both information-free floors change every document's predictions (positive control).
- Scoring decoded 1.33M tokens on the GPU (221 tok/s); Qwen states saved at 192 tok/s,
  Muse states at 236 tok/s. Wall time 3.1 h with the live servers stopped.

## Limits

- Real multi-turn chats of 8K+ are rare in the held-out pool (5 found, 3 scored, the rest
  reserve), so that row rests on three conversations even though its CI is narrow.
- Concatenated sessions are synthetic: topic switches are abrupt and later topics rarely
  depend on earlier ones.
- Agent traces score short final answers (45–166 tokens, 678 in total), so they weigh
  less in the token-weighted "all" row than in a per-document mean (+0.153).
- Teacher-forced scoring; states restored by `kvpark-kv-probe`, not a live llama-server.
- The decomposition ran on agent traces only; whether a better fill also helps the
  concatenated sessions is untested (in v3's 4–8K chats it did not: −0.003).

## Reproduce

Run directory `evidence/linear-transfer/run-v3b/` (not shipped), corpus
`evidence/linear-transfer/corpus-long-chat/` (30 conversations: 13 concatenated sessions,
12 agent traces, 5 real multi-turn; 24 primary, 6 reserve). The derived concatenated
subset is `datasets/smoltalk2/multi_turn_concat_sessions-00000-of-00001.parquet` with a
`.sources.json` sidecar naming the source rows. The decomposition is in
`evidence/linear-transfer/run-v3b-agent-decomp/report.json`. Command (abridged; every setting is recorded
under `settings` in `run-v3b/long/report.json`):

```
run_long.py --output run-v3b --probe kv-probe-build-v1gpu/kvpark-kv-probe --device vulkan \
  --corpus corpus-long-chat/conversations.jsonl --prefix-buckets 8192,16384,29500 \
  --conditions sanity,table_fill,mean_old --extra-conditions mean_table --tails 0,2048,4096 \
  --v2-run run-v2 --token-table run-v2-token-table.npz --grow-train \
  --train-corpus corpus-v3-train/conversations.jsonl ...
```
