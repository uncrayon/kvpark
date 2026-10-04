# Long-context evaluation corpus and v3 training corpus

Two corpora support the v3 handoff experiment (design v3, section T2). Both were
built on September 30, 2026 with `build_corpus.py` from pinned
[SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) files (revision
`fc6cc2103c066455aade5d7fbb346039ae36ca5e`). Each model's own GGUF chat template
renders the conversations, and `kvpark-kv-probe --mode tokenize` counts tokens with
the vocabulary only.

- `evidence/linear-transfer/corpus-long`: 45 test conversations, binned by Muse prefix
  length and spread evenly over each bin, for handoff scoring at long prefixes.
- `evidence/linear-transfer/corpus-v3-train`: 123 training conversations, about 150K
  Muse tokens, for refitting the mapper on more data.

Neither corpus shares a row id, a user-turn sequence or a document id with corpus v2 or
with the other corpus, and no 200-character stretch of user text appears in two of them.
`build_corpus.py` enforces the id and user-turn rules through `--exclude`, and all
three corpora were cross-checked afterwards. The conversation text is not stored in
this repository.

**Revision.** The first corpus-long build (now `evidence/linear-transfer/corpus-long-r1`,
conversations sha256 `20753bdd…`) was replaced the same day, after review, by the
current one (`0b68546e…`). Three things changed. Primaries now spread across the whole
range of each bucket; before, none of the top-bucket primaries had a prefix above 24K.
Chinese scored windows, which sat only in the top bucket, are gone. Every document
has a full 256-token scored window; before, five 4–8K primaries were short smolagents
answers. corpus-v3-train was built against r1 and is unchanged, because its GPU
collection had already started. The new corpus-long therefore excludes corpus-v3-train
rather than the other way round.

## Rebuild

The LongAlign shards come from `fetch_dataset.py --include-longalign`. Each command
below runs in about a minute and never loads model weights. The corpus-v3-train
command lists the exclusions it was built with. corpus-long-r1 was at
`evidence/linear-transfer/corpus-long` at the time, so a rebuild reproduces
`conversations.jsonl` byte for byte, and the manifest differs only in that path and in
the builder digest.

```bash
experiments/linear_transfer/.venv/bin/python experiments/linear_transfer/build_corpus.py \
  --dataset-dir evidence/linear-transfer/datasets/smoltalk2 \
  --probe evidence/linear-transfer/kv-probe-build-s1/kvpark-kv-probe \
  --qwen-model /path/to/Qwen3.8-27B-Q8_0.gguf --muse-model /path/to/Muse-Glimmer-30B-Q8_0.gguf \
  --output evidence/linear-transfer/corpus-v3-train --split train --seed 20260930 \
  --exclude evidence/linear-transfer/corpus-v2/conversations.jsonl \
  --exclude evidence/linear-transfer/corpus-long-r1/conversations.jsonl
```

```bash
experiments/linear_transfer/.venv/bin/python experiments/linear_transfer/build_corpus.py \
  --dataset-dir evidence/linear-transfer/datasets/smoltalk2 \
  --probe evidence/linear-transfer/kv-probe-build-s1/kvpark-kv-probe \
  --qwen-model /path/to/Qwen3.8-27B-Q8_0.gguf --muse-model /path/to/Muse-Glimmer-30B-Q8_0.gguf \
  --output evidence/linear-transfer/corpus-long \
  --subsets multi_turn_reasoning_if_think,smolagents_toolcalling_traces_think,LongAlign_64k_Qwen3_32B_yarn_131k_think \
  --prefix-buckets 4096,8192,16384,29500 --docs-per-bucket 10 --reserve-per-bucket 5 --strata-per-bucket 5 \
  --bucket-subsets '{"4096":"multi_turn_reasoning_if_think,smolagents_toolcalling_traces_think","8192":"LongAlign_64k_Qwen3_32B_yarn_131k_think","16384":"LongAlign_64k_Qwen3_32B_yarn_131k_think"}' \
  --split test --max-doc-tokens 32768 --tool-limit 0 --reasoning-limit 0 \
  --final-message-max-bytes 4096 --min-scored-tokens 256 --max-non-latin-share 0.1 --bytes-per-token 4.5 \
  --oversample 4 --seed 20260930 \
  --exclude evidence/linear-transfer/corpus-v2/conversations.jsonl \
  --exclude evidence/linear-transfer/corpus-v3-train/conversations.jsonl
```

A second run of the corpus-long command gave byte-identical `conversations.jsonl` and
`manifest.json`. Without the new flags, `build_corpus.py` still rebuilds corpus v2 byte
for byte: `conversations.jsonl` is identical, and the manifest differs only in the new
settings, an empty `excluded_from` and the builder digest.

## corpus-long

### Selection

- **Prefix P** is the number of Muse tokens that end at or before the first rendered
  segment of the final message. This is the `attention_eval` / `inject` convention, so
  the final message's template header (`<|start|>assistant to=self<|message|>`) counts
  as prefix. The manifest also records the Qwen prefix under the same rule.
- **Buckets** are [4,096, 8,192), [8,192, 16,384) and [16,384, 29,500). Each bucket
  holds 10 primary and 5 reserve conversations. `manifest.json` → `long_eval.documents[].role`
  marks each one. `conversations.jsonl` lists the conversations bucket by bucket,
  primaries first.
- **Strata.** Each bucket is split into 5 equal-width prefix strata
  (`--strata-per-bucket 5`): 819 tokens wide in the 4–8K bucket, 1,638 in the 8–16K
  bucket and 2,623 in the top bucket. Candidate gathering fills every stratum
  separately, so the common lengths cannot crowd out the long ones. Selection then
  rotates over the strata from low to high, which gives each stratum 2 primaries and 1
  reserve. `documents[].stratum` and `by_bucket[].by_stratum` record the result.
- **Sources.** The 4–8K bucket alternates, within each stratum, between
  `multi_turn_reasoning_if_think` and `smolagents_toolcalling_traces_think`. The 8K+
  buckets use only `LongAlign_64k_Qwen3_32B_yarn_131k_think`. A LongAlign row is a
  user turn (a long document plus a question) and an assistant turn (reasoning plus
  answer). The builder adds its default system prompt.
- **Whole conversations.** Nothing is shortened, and tool outputs and reasoning are
  not cut (`--tool-limit 0 --reasoning-limit 0`). The final message is a plain
  assistant answer with no tool calls. Documents have at most 32,768 tokens on both
  models (exact counts).
- **Full scored windows.** `--min-scored-tokens 256` requires the whole 256-token
  window inside the final message. This drops the short smolagents `final_answer`
  replies (31 rejected), so smolagents supplies only 1 of the 15 conversations in the
  4–8K bucket, and multi_turn supplies the rest.
- **Latin script only.** `--max-non-latin-share 0.1` drops a conversation when more
  than 10% of the letters in its user turns, or in its scored window, are outside the
  Latin script. It is checked on the final message's opening text before tokenizing
  and on the exact Muse scored window after. This rule rejected 87 LongAlign rows.
  In r1, the non-Latin scored windows were all Chinese: 4 top-bucket primaries and 5
  reserves. Two of those primaries were English documents with a Chinese question,
  and therefore Chinese reasoning. `documents[].script` records the non-Latin and CJK
  shares of the user text and of the scored window. The largest values in the corpus
  are 0.003 (user) and 0.035 (scored window).
- **Seeded candidate gathering.** Rows are drawn in seeded order. A byte estimate of P
  (`--bytes-per-token 4.5`) places each candidate in the nearest stratum, within ±25%
  of the subset's bucket range. 4.5 is the median measured on a trial build (LongAlign
  2.6–5.8, multi_turn 3.8–5.2 bytes per Muse token). The earlier 3.6 overestimated P
  and left the top strata empty. Exact tokenization then assigns the bucket and
  stratum.
- **Tokenization never aborts.** A document over the probe's 32,768-token context is
  named on the probe's stderr. It is dropped (8 were) and the batch is tokenized
  again. Documents over 6 bytes per allowed token are skipped before tokenizing.

### Final-message cut and its check

A final message longer than 4,096 bytes (reasoning plus answer) is cut at a whitespace
boundary. Reasoning is cut first, to no less than half the budget, and the answer keeps
its opening text. 36 of the 45 conversations were cut. The cut must leave the first
`check_tokens = max(P + 256, last of the first 256 scored positions + 1)` tokens
unchanged, plus a 64-byte margin. If the first cut falls inside that window, it is
widened. This never happened in the real build, but the fixture test exercises it.

Each cut conversation is then retokenized next to a tokenization of the uncut text. The
first `check_tokens` token ids and byte ends must match for Muse and for Qwen, where
Qwen uses its own P. All 36 matched. An independent re-check went back to the parquet
rows for all 45 conversations. It confirmed the prefix messages, the cuts, P, the
scored counts, the buckets, the strata and the script shares. It also tokenized every
uncut rendering in full: all fit in 32,768 tokens, and no token straddles the start of
the final message. The cut and uncut token streams agree for at least P + 256 + 106
Muse tokens, and at least P + 256 + 111 Qwen tokens with Qwen's own P.

### Contents

| Prefix bucket (Muse) | Conversations | Primary Muse P (sorted) | Qwen P median | Final-message Muse tokens, median (range) | Cut |
| --- | --- | --- | ---: | --- | ---: |
| 4,096–8,191 | 14 multi_turn + 1 smolagents | 4,287 4,909 5,026 5,290 5,904 6,050 7,025 7,035 7,433 7,933 | 6,312 | 818 (268–1,035) | 12 |
| 8,192–16,383 | 15 LongAlign | 9,179 9,748 10,474 11,110 12,647 12,798 13,301 13,808 14,946 15,099 | 13,098 | 830 (451–1,128) | 14 |
| 16,384–29,499 | 15 LongAlign | 16,642 18,105 19,288 21,416 22,053 24,212 25,832 25,916 27,654 28,449 | 25,431 | 855 (581–922) | 10 |

Totals are 656,667 Muse tokens and 695,389 Qwen tokens. The 30 primary conversations
hold 438,056 Muse and 462,696 Qwen tokens. The largest documents are 29,365 Muse tokens
and 30,600 Qwen tokens (Qwen prefix up to 29,574). All 45 conversations provide the
full 256-token scored window.

`manifest.json` → `long_eval` records, per document: bucket, stratum, role, token
counts, both prefixes, final-message and scored token counts, `check_tokens`, the cut
(old and new bytes per field), the verified token counts and the script shares. Per
bucket, it records the counts per stratum, the largest script shares, and
distributions of all prefixes, of the primary prefixes and of the scored windows.
`rejected` counts every rejection reason.

### Caveats and advice for scoring

- **Report P as a continuous covariate as well as by bucket.** The strata spread P
  evenly within each bucket. The buckets still differ in more than length, however:
  the 4–8K bucket is multi-turn chat, and the 8K+ buckets are single-question
  long-document reading (LongAlign), as the design specifies. The 8K→16K comparison
  is the clean length contrast, and the 4–8K bucket mixes in a change of source.
- **The script filter does not detect Latin-script languages.** All scored windows are
  English. Among the primaries, though, one top-bucket prefix is a Danish novel with
  an English question (LongAlign:5334), and one 8–16K prefix is Ruby source code
  (LongAlign:4555). A reserve in the top bucket is a genealogy index (LongAlign:749).
  These were found by an English stopword check, which the builder does not apply.
- **P is not uniform inside a stratum.** Within each stratum, the documents follow the
  source's own length distribution. A byte estimate places candidates in strata, and
  densities range from 2.6 to 5.8 bytes per token, so a very dense or very sparse
  document can be missed at the edges of the range.
- LongAlign documents are often PDF extractions with ragged whitespace. The reasoning
  and answers are Qwen3-32B generations. Scoring is teacher-forced on those
  references.

## corpus-v3-train

This corpus uses the v2 mix, the v2 settings and the v2 seeded quota procedure, with
seed 20260930. Every conversation is in the `train` split. Corpus v2 and
corpus-long-r1 are excluded, and the current corpus-long excludes this corpus.
Exclusion drops a row whose id matches, whose full user-turn sequence matches, or
whose first user turn matches when that turn has at least 64 characters. Short
greetings such as "Hi" recur across unrelated conversations, so they do not count.

| Subset | Conversations | Muse tokens | Qwen tokens |
| --- | ---: | ---: | ---: |
| multi_turn_reasoning_if_think | 19 | 39,730 | 40,899 |
| smolagents_toolcalling_traces_think | 18 | 32,296 | 32,061 |
| smoltalk_everyday_convs_reasoning_Qwen3_32B_think | 20 | 22,080 | 23,258 |
| smoltalk_systemchats_Qwen3_32B_think | 22 | 21,669 | 22,753 |
| hermes_function_calling_v1_no_think | 22 | 21,511 | 19,621 |
| aya_dataset_Qwen3_32B_think | 22 | 20,110 | 21,341 |
| **Total** | **123** | **157,396** | **159,933** |

21 documents, holding 54,540 Muse tokens, exceed Muse's 2,048-token sliding window.
Documents are 160–4,608 tokens per model, as in v2.
