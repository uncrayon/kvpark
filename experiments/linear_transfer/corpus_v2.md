# Calibration corpus, version 2

Corpus v2 contains 125 chat conversations drawn from six Apache-2.0 subsets of
[SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2), pinned to
revision `fc6cc2103c066455aade5d7fbb346039ae36ca5e`. It totals 154,036 Muse tokens
and 157,168 Qwen tokens once each model's own chat template has rendered it. The
conversation text is not stored in this repository. `fetch_dataset.py` downloads
the six parquet files and checks their SHA-256 digests, and `build_corpus.py`
rebuilds the corpus deterministically:

```bash
python experiments/linear_transfer/fetch_dataset.py --output evidence/linear-transfer/datasets/smoltalk2
```

```bash
experiments/linear_transfer/.venv/bin/python experiments/linear_transfer/build_corpus.py \
  --dataset-dir evidence/linear-transfer/datasets/smoltalk2 \
  --probe evidence/linear-transfer/kv-probe-build-a2/kvpark-kv-probe \
  --qwen-model /path/to/Qwen3.8-27B-Q8_0.gguf --muse-model /path/to/Muse-Glimmer-30B-Q8_0.gguf \
  --output evidence/linear-transfer/corpus-v2
```

The builder loads only tokenizers (`kvpark-kv-probe --mode tokenize`, vocab only),
never model weights. It writes `conversations.jsonl` and `manifest.json`. The
manifest records the source file digests, template digests, settings, per-split
totals, and a count of every rejection reason.

## Why chat format

Version 1 used raw prose with no BOS or template, and Muse's first content token
turned into an attention sink. Deployment caches are chat transcripts. Muse
receives `<|begin_of_text|>` and its own `<|start|>…<|message|>` framing, while Qwen
uses `<|im_start|>` blocks with `<think>` sections. Version 2 renders each
conversation with the chat template stored in each GGUF, the same template
`llama-server --jinja` applies. Only message text shared by both renderings
(content, reasoning, string tool-call arguments) is aligned across models. Template
and special tokens stay native to each model.

## Contents

| Subset | Conversations (train/dev/test) | Muse tokens | Character |
| --- | ---: | ---: | --- |
| multi_turn_reasoning_if_think | 11 / 5 / 4 | 39,226 | long multi-turn chats with reasoning |
| smolagents_toolcalling_traces_think | 14 / 3 / 3 | 30,355 | agent traces: web search, page visits, tool outputs |
| smoltalk_everyday_convs_reasoning_Qwen3_32B_think | 13 / 3 / 3 | 21,874 | everyday conversations with reasoning |
| smoltalk_systemchats_Qwen3_32B_think | 17 / 3 / 4 | 21,729 | persona system prompts |
| hermes_function_calling_v1_no_think | 14 / 5 / 4 | 20,944 | Hermes function calling, several calls per turn |
| aya_dataset_Qwen3_32B_think | 12 / 4 / 3 | 19,908 | multilingual prompts with reasoning |

Across the 125 conversations there are 200 user turns, 242 assistant turns (160 with
reasoning), 45 tool calls and 45 tool results. 37 conversations define tools.
Documents are limited to 160–4,608 tokens per model. 19 documents with 50,078 Muse tokens
exceed Muse's 2,048-token sliding window. The split is by whole conversation and
stratified by subset: train 81 conversations (105,992 Muse tokens), dev 23 (23,384),
test 21 (24,660). Use dev for every selection (layers, k, lambda) and look at test
once.

## Conversion rules

- The system message is `custom_instructions` when present, otherwise a fixed
  generic prompt, with a tool-use variant when tools are defined.
- Assistant `<think>…</think>` becomes `reasoning_content`. `<tool_call>` payloads
  (JSON or Python literals) become structured `tool_calls` with object arguments.
  `<tool_response>` blocks, or a single `Observation:` text, become `tool` messages
  that answer the calls in order.
- A trailing unanswered `final_answer` call (smolagents) becomes the assistant's
  answer text. Any other unanswered trailing call is dropped.
- All text is whitespace-stripped, because Qwen's template trims it. Tool outputs over
  4,000 characters and reasoning over 6,000 characters are cut at a line boundary.
  Half of the conversations, chosen at random, are shortened at message boundaries
  to a random 600–1,800-token budget, which gives more and more varied documents.
  Every conversation ends with an assistant answer that has content, so any
  conversation can serve as a continuation test.
- The builder rejects rows it cannot parse, tool responses that do not match their
  calls, duplicate user-turn sequences, documents over 4,608 tokens for either model,
  and any document where a special-token string appears inside message text.

The conversations inherit SmolTalk2's generated content and its biases. They are a
calibration distribution for chat-formatted caches, not a benchmark or a sample of
kvpark users' conversations.
