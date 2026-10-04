# Synthetic calibration corpus, version 1

`corpus_v1.jsonl` contains 40 independently authored documents: **32 training
documents and 8 held-out evaluation documents**. Each JSONL record has `id`,
`split` (`train` or `eval`), and `text`. The model input is the `text` field only.
IDs and split labels are bookkeeping metadata, not prompt prefixes. Paragraph
breaks and Unicode characters are part of the original text and must be retained.

This is a small engineering calibration set for the first offline Qwen → Muse
value-projection probe. It is not a benchmark, a representative sample of user
conversations, or evidence that a transferred cache will support useful
generation. The documents are 193–220 whitespace-separated words each (8,463
total). Token counts depend on the tokenizer. Preflight against the actual Qwen
and Muse GGUFs found 226–275 tokens per document across the two tokenizers, below
both the 512-token prototype setting and the collector's 1,024-token default.
Overlong input is rejected rather than truncated.

All prose was newly authored for this experiment, including its fictional
people, organizations, operational rules, and records. No private conversations,
web excerpts, model downloads, or existing benchmark items were used. General
technical explanations are illustrative and are not quotations. This material
is provided under the repository's [MIT license](../../LICENSE).

## Split map

Each document has a distinct task or topic. The evaluation topics were set aside
at authoring time; they are not paraphrases or parameter substitutions of
training documents. Broad styles such as technical prose and fictional records
occur in both splits. The split therefore checks transfer across these selected
topics, not across wholly separate domains.

| Training ID | Topic and form |
| --- | --- |
| train01 | Connection-pool regression and controlled rollback investigation |
| train02 | Quoted CSV fields, multiline records, and parser state |
| train03 | Stable cursor pagination in a changing catalog |
| train04 | File-watch debounce and obsolete render results |
| train05 | UTF-8 offsets, token pieces, and normalization |
| train06 | Configuration precedence and value provenance |
| train07 | Ordered asynchronous results and bounded buffering |
| train08 | Conditional HTTP reads and optimistic writes |
| train09 | Buffer-pool ownership and lifetime |
| train10 | Fictional observatory equipment and exposure ledger |
| train11 | Fictional garden watering responsibilities and observations |
| train12 | Fictional nested packaging rules and item accounting |
| train13 | Fictional tram dispatch permissions and occupancy invariant |
| train14 | Fictional museum accession, components, and movement records |
| train15 | Fictional game quest prerequisites and item state |
| train16 | Simulated inspection tools and observation provenance |
| train17 | Calendar instants, time zones, and recurrence |
| train18 | Flow-rate units, timing, and measurement precision |
| train19 | Breadth-first room routing and path reconstruction |
| train20 | Distributed event causality and clock scope |
| train21 | Lexer states, token positions, and syntax boundaries |
| train22 | Container variance through permitted operations |
| train23 | Byte-array XOR parity and reconstruction verification |
| train24 | Audio resampling, filtering, and alignment |
| train25 | Typography, wrapping opportunities, and logical text |
| train26 | Kitchen scene in narrative prose |
| train27 | Personal letter about an invented workshop |
| train28 | Spanish bicycle maintenance log and diagnosis |
| train29 | Bilingual library directions and a temporary route limitation |
| train30 | Classroom cooling experiment and observational controls |
| train31 | Archival arrangement, descriptions, and evidence gaps |
| train32 | Studio dialogue about a painting's composition |

| Evaluation ID | Held-out topic and form |
| --- | --- |
| eval01 | DNS negative caching and resolver diagnostics |
| eval02 | Build dependency scheduling, cycles, and blocked targets |
| eval03 | Fictional balloon telemetry, local logs, and recovery records |
| eval04 | Simulated keyboard-navigation audit and scope of observations |
| eval05 | Spanish oral history, uncertain recollection, and publication choices |
| eval06 | Linear-light compositing and alpha conventions |
| eval07 | Three-against-four musical rhythm and a shared time base |
| eval08 | Fictional council grants, provisional motions, and final accounting |

## Use and limitations

Pass this file to `run_prototype.py` as shown in the [prototype instructions](README.md).
It collects each model once, then separates captures by document hash before
fitting. This saves repeated model loads while keeping training and evaluation
observations separate. Keep each selected `text` unchanged so that its UTF-8
SHA-256 remains stable across models.

The lower-level `collect.py` reads every document supplied and ignores `split`.
If using it directly, filter records into separate files or split the resulting
captures by whole-document hashes before fitting. Train the ridge mapping only
on training captures and calculate held-out reconstruction metrics only on
evaluation captures. Never fit on both splits together.

This split prevents identical-document leakage in the first probe. It does not
prove statistical independence: all documents were written in one authoring
session, and several share explanatory style. Adjacent token observations from
one document are also correlated. Report per-document and aggregate results;
token count alone is not the number of independent samples. Choosing layers or
hyperparameters after repeatedly inspecting these eight evaluation documents
would turn them into a development set and require a new held-out set.

The corpus is mostly English, with two Spanish documents, one bilingual document,
Unicode examples, short code-like expressions, and varied narrative structures.
It has no long programs, extended chat histories, real tool outputs, images,
audio, adversarial inputs, or task-scored answers. It is too short to assess
long-context behavior or the real Muse sliding-window boundary. Reconstruction
accuracy on its values does not establish attention quality, complete cache
transfer, or a prefill speedup. Those require later experiments using keys,
receiver inference, representative workloads, and independently scored tasks.

Corpus file SHA-256:
`55ce31ea1996dbe7499d6320f8e542af06254de59617b02ffac36d92bc46f0ea`.

## Tokenizer preflight

Vocabulary-only checks against the local Qwen3.8-27B and Muse Glimmer-30B GGUFs
confirmed exact text reconstruction and the following sizes for this corpus:

| Tokenizer | Total tokens | Shortest document | Longest document |
| --- | ---: | ---: | ---: |
| Qwen | 10,138 | 230 | 275 |
| Muse | 9,924 | 226 | 271 |

There are 9,860 shared token end offsets within matching documents, covering
99.355% of Muse's rows. This high alignment rate applies to these texts and
tokenizers; it does not establish that the models' value representations are
compatible. Every report document hash matches the corresponding corpus text.
These checks loaded vocabularies only and did not collect model activations.

Local evidence is stored in
`evidence/linear-transfer/corpus-v1-qwen-tokenization/report.json` and
`evidence/linear-transfer/corpus-v1-muse-tokenization/report.json` at the
repository root. Preserve those reports with the later capture artifacts so
that token accounting can be checked independently of the fit.
