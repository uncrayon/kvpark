# kvpark bridge (experimental): switch a conversation between models

`kvpark bridge` moves a live conversation's cache from one local model to another, so the
new model continues without re-reading the whole conversation. It is experimental and off
unless you configure it.

kvpark already parks and restores a model's **own** cache. The bridge adds **translation**:
the cache of model A is encoded into a shared, model-neutral latent (the *hub*) and decoded
into model B's cache format. Model B restores that cache for the older part of the
conversation and reads only the most recent ~2,000 tokens itself (the *native tail*).

Design and evidence: [experiments/universal_kv/ARCHITECTURE.md](../experiments/universal_kv/ARCHITECTURE.md).

## What it can and cannot do

- **Output is always well-formed.** llama-server reuses only cells whose tokens match the
  prompt, so the target conditions on exactly the right tokens. Translation only changes the
  *values* of the reused cells; imperfect values change what the model writes, never break it.
- **Quality depends on the conversation.** Plain chat translates well; long documents and
  agent work with large tool outputs lose noticeably more (measurements in
  [experiments/universal_kv/RESULTS.md](../experiments/universal_kv/RESULTS.md)). Use
  `kvpark bridge undo MODEL` to drop a translated cache; the next request then prefills natively.
- **Speed.** On an EVO-X2-class machine (Ryzen AI Max+ 395) with the engine daemon on the
  integrated GPU, a switch of an 11-14K-token conversation took 10-17 s between Qwen3.8-27B and
  Muse-Glimmer-30B and 8 s into Qwen3-4B, against 47-86 s and 15 s of native prefill (every
  ordered pair of the three, on a machine shared with other work). Most of it is the target
  re-reading its native tail (2,048 tokens: 8-11 s on Qwen3.8, 12-16 s on Muse, 4 s on
  Qwen3-4B). The translation itself takes 0.5-1.1 s on the GPU on a quiet machine (up to 3.7 s
  under load; a translation into the dense Qwen3-4B now takes 0.8-1.1 s) and 2-14 s on the CPU. Muse → the
  production Qwen3.8 build, the latest run: 0.9-3.2 s for the switch plus an 8 s tail, against
  41-59 s of native prefill. Staging the switch in Hermes loads the engine ahead (20 s the very
  first time, while the bridge caches the models' metadata). With the fifteen-model hub6 bundle
  and the Vulkan engine (October 9, under a thermal guard that paused the jobs): Kimi-Linear-48B and
  Gemma-4-26B → Qwen3.8 took 10-12.5 s against 48-52 s of native prefill (12-15K tokens), 3.3-6.8
  times faster where the native prefill was paused least, and switches into Gemma-4-26B 2.7-3.5 s
  against 6.6-6.8 s (8-10K tokens, no pause); a small, fast target such as Granite-4.0-H-Tiny prefills
  8-11K tokens in 4-5 s, so a switch into it (about 2 s) saves little. Measurements:
  [RESULTS.md](../experiments/universal_kv/RESULTS.md), sections 6, 8 and 10.
- **Models.** A model can take part only if the hub bundle has an adapter for it. The engine
  translates these llama.cpp architectures (`TRANSLATABLE` in `src/kvpark/bridge/capabilities.py`):
  - dense GQA `qwen3` and sliding-window + global `muse-glimmer` (measured live);
  - the hybrids with recurrent layers, whose recurrent state starts at the handover point from
    zeros or from the model's mean (see [Hybrid targets](#hybrid-targets-the-recurrent-state)) and
    is carried on by the native tail: `qwen35` (Qwen3.5/3.8 and Ornith, measured live),
    `qwen35moe`, `qwen3next`, `qwen4exp`, `minimax-01`, `jamba`, `falcon-h1`, `granitehybrid`,
    `nemotron_h`, `nemotron_h_moe`, `plamo2` (its cache; its PLaMo-2 vocabulary is not read yet, so
    switches with it are skipped as `tokenizer`), `lfm2`, `lfm2moe`;
  - the GQA family, one plain or a sliding-window + global cache with per-layer heads and
    RoPE: `llama`, `mistral3`, `qwen3moe`, `hy_v3` (Hy3), `phi3` (Phi-3/4; none of them LongRoPE),
    `gemma3`, `gemma4` (Gemma 4, with its shared KV layers and its MTP head), `gpt-oss`,
    `laguna`, `mimo2` (MiMo);
  - the MLA family, K-only caches: `deepseek2` (DeepSeek V2/V3, Kimi K2.x, GLM-4.7-Flash),
    `mistral4`, `hy_v4`, the DSA models with indexer keys `deepseek32`, `glm-dsa`, `dots3note`,
    and the K-only hybrids `kimi-linear`, `kimi-k3`, `bailingmoe3` and `glm5-next` (GLM-5.3-Flash,
    llama.cpp a7b94df2c only);
  - `deepseek4` (DeepSeek V4) as a target only: see [DeepSeek V4 targets](#deepseek-v4-targets).

  Real models have been measured for `qwen3`, `muse-glimmer` and `qwen35` (Qwen3-4B,
  Muse-Glimmer-30B, Qwen3.8-27B and Ornith-1.5-9B, fitted jointly), and onboarded into the hub and
  measured against Qwen3.8 in both directions for `qwen35moe` (Qwen3.6-35B-A3B), `lfm2` (LFM2-2.6B),
  `deepseek2` (DeepSeek-V2-Lite, GLM-4.7-Flash), `kimi-linear` (Kimi-Linear-48B-A3B), `bailingmoe3`
  (Ling-3.0-tiny), `falcon-h1` (Falcon-H1-3B), `granitehybrid` (Granite-4.0-H-Tiny), `nemotron_h_moe`
  (Nemotron-3.5-Lightning-30B-A3B), `jamba` (Jamba-Reasoning-3B) and `gemma4` (Gemma 4 26B-A4B and
  31B); the hub bundle with all of them but Jamba is `hub6-20261009-v2`
  ([RESULTS.md section 10](../experiments/universal_kv/RESULTS.md#10-hub6-eight-more-models-onboarded-october-8-9-2026);
  use it rather than `hub6-20261008-v2`, which carries the `reverse` encoders of Granite-4.0-H-Tiny and
  Nemotron-3.5 measured to do no good or harm). How much of the conversation a switch keeps varies by model: on the ordinary chats
  every measurement shares, a switch from Qwen3.8 removed 71-83% of what an information-free cache
  loses into GLM-4.7-Flash and Ling-3.0-tiny, 31-51% into LFM2, Qwen3.6, DeepSeek-V2-Lite and Gemma 4,
  16-21% into Nemotron-3.5 and Kimi-Linear, and nothing measurable into Granite-4.0-H-Tiny, Falcon-H1
  and Jamba. Out of a model, into Qwen3.8, it removed 25-42% (like the jointly fitted Qwen3-4B's 30%)
  wherever the model's cache predicted every hub dimension at least a little. Where the onboarding's
  encoder warnings fired (a hub dimension the model's cache predicts with R² below 0.05, and the
  variance cap binding on half of the dimensions or more: Granite-4.0-H-Tiny, Nemotron-3.5, Jamba),
  the bundle carries a plain `ridge` encoder for Granite and Nemotron. Out of Granite it is harmless and keeps about as little as an
  information-free cache (1%): the target gets little beyond its native tail, so when the earlier
  conversation matters a native prefill is the better choice. Out of Nemotron, as the servers render
  (what a live switch holds), even that encoder does worse than no translation, so register
  Nemotron-3.5 with `--no-source`: switches out of it are then skipped (`source-refused`) and the
  next model prefills natively, while switches into it still translate. Jamba is not in the bundle.
  Every other architecture is verified on tiny random-weight models only: its cache layout, the
  restore and decoding are checked, the translation quality is not. Pure
  recurrent models (Mamba, Mamba2, RWKV6/7, ARWKV7) keep no per-token cache and are always
  prefilled natively. MiniMax-M3 is not supported yet, nor are LongRoPE models (the
  long-context Phi-3/4 models, and Llama or Mistral 3 GGUFs converted with LongRoPE) and
  embedding models (such as gte-Qwen2). A switch the translator cannot handle is skipped, never
  forced: see [Skipped switches](#skipped-switches).
- **Vocabularies and templates.** The engine reads GPT-2 byte-level BPE vocabularies (Qwen,
  Muse, DeepSeek, GLM, Kimi, Laguna, ...), Gemma 4's and SentencePiece ones (Jamba); an engine
  older than its `vocabularies` feature reads GPT-2 BPE only, and the bridge then skips switches
  into or out of the others (`tokenizer`). It renders each conversation with the target's own chat
  template as llama-server does; all 19 local models render byte for byte like llama-server on
  a 96-case battery on both llama.cpp versions (through tokenizer twins: tiny models carrying the
  real tokenizer and template), and like their real servers on the earlier 56-case battery.
- **llama.cpp versions.** Servers on llama.cpp a7b94df2c (what `scripts/build_runtime.py` builds,
  in `runtime/` here) and on 85d5703a3 (the production Qwen3.8 build) both work, as sources and
  as targets, in any mix. The two save a few cache families differently (DeepSeek V2/V3 MTP
  layers, Qwen3.8-Flash-Next's indexer, Ling 3 with rope sections); the engine reads either and
  writes what the target's own save holds, and the bridge passes each server's `build_info` to
  it.
- **Servers.** Each model needs its own llama-server with `--slot-save-path` and one slot
  (`--parallel 1`). Upstream llama-server and kvpark's patched server (with its `.resume`
  sidecars) both work as targets, with or without a DFlash or MTP draft model; see
  [Draft models](#draft-models-dflash-mtp). The patched path is verified on CPU builds of both
  llama.cpp versions (`runtime/`, a7b94df2c, and the earlier 85d5703a3 build) with small test
  models and live on the production Qwen3.8 build (ROCm, DFlash2 draft, mmproj; and the same
  build with MTP), in and out of it.
  The target's resident conversation is replaced by the switch; park it first if you need it.
- **Images, audio and video.** A target that can read the media gets them as they are; the
  translated cache stops before the first message that holds them, and the target reads that
  message and everything after it itself. A target that cannot (an image moving to a text-only
  model) gets a text stand-in instead, such as `[Image description: ...]`, made by the source
  model or a describer and cached. See [Media across models](#media-across-models).

## Set up

1. Run each model's llama-server with a slot directory, for example:

   ```bash
   llama-server --model /path/to/Muse-Glimmer-30B-Q8_0.gguf --port 18083 --parallel 1 --slot-save-path ~/.local/share/kvpark/slots/muse --jinja
   ```

2. Register the models with the bridge (names must match the hub's model names, or pass
   `--hub-name`):

   ```bash
   kvpark bridge add muse --url http://127.0.0.1:18083 --gguf /path/to/Muse-Glimmer-30B-Q8_0.gguf --slot-dir ~/.local/share/kvpark/slots/muse
   ```

   Per-target settings, from the measurements:

   | Setting | Use it for | Why |
   | --- | --- | --- |
   | `--native-prefix` | Qwen-family targets (Qwen3, Qwen3.5/3.8, Ornith) | they need their own values for the first tokens of the prompt (the attention sink); Muse does better without; refused for DeepSeek V4 |
   | `--recurrent-mean NPZ` | hybrid targets (Qwen3.5/3.8, Ornith, ...) | the translated state's recurrent cell starts from the model's mean of native cells instead of zeros: see [Hybrid targets](#hybrid-targets-the-recurrent-state) |
   | `--recurrent-tools` | with `--recurrent-mean`, when the file's `tools` group was calibrated on your own agent prompts | conversations with tools then use that group instead of the pooled mean |
   | `--min-handover N` | small models (e.g. 4B) | their native prefill is already fast; translate only very long prompts |
   | `--tail N` | targets that need more native context | a longer native tail is slower but closer to native quality |
   | `--draft-window N` | targets whose server runs a DFlash draft with a sliding window other than 2048 | the native tail is kept at least `N - 1` tokens long, so the draft sees only native context |
   | `--no-source` | a model the hub translates out of worse than no translation (Nemotron-3.5 with the hub6 bundle) | switches out of it are skipped (`source-refused`) and the next model prefills natively; switches into it still translate |

   `kvpark bridge add` records the model's GGUF architecture (`arch` in `kvpark bridge status`);
   entries registered before that get it on their next switch.

   A model whose server runs with RoPE flags that change its cache (`--rope-scaling`,
   `--rope-scale`, `--rope-freq-base`, `--rope-freq-scale`, `--yarn-orig-ctx`,
   `--yarn-ext-factor`, `--yarn-attn-factor`, `--yarn-beta-slow`, `--yarn-beta-fast`) is
   registered with the same flags and values; every engine job carries them, so the engine
   reads and writes that model's cache with them. The state codec models these settings for
   the MLA family and the generic hybrids only; for today's translated architectures (Qwen3,
   Qwen3.5/3.8, Muse) it takes RoPE from the GGUF, so `kvpark bridge add` refuses the flags
   for them (a switch would save the source's cache and then fail in the engine).

   Likewise a model whose server runs with chat template flags (`--chat-template-kwargs`,
   `--reasoning on|off`, `--reasoning-effort`, `--reasoning-preserve` or
   `--no-reasoning-preserve`) is registered with the same flags and values: as a target, the
   engine then renders the conversation as that server does (see
   [Request template settings](#request-template-settings)). Without them a switch into such a
   server usually ends in `template-mismatch`. Flags that keep the template's own default (the
   production Qwen3.8's `--reasoning-effort xhigh --reasoning-preserve`) need not be given.

3. Point the bridge at the translation engine and a hub bundle (a directory written by
   `experiments/universal_kv/fit_hub.py`, ideally converted once to format 2 with
   `experiments/universal_kv/convert_hub.py HUB HUB-v2`, which loads a model pair in a
   fraction of a second instead of several seconds):

   ```bash
   kvpark bridge engine --python ~/venvs/rocm/bin/python --script experiments/universal_kv/engine.py --hub /path/to/hub-v2 --device auto
   ```

   The engine Python needs numpy with an optimized BLAS (OpenBLAS or MKL, as pip's numpy
   wheels bundle; a distribution numpy on the reference BLAS makes a switch take minutes) and,
   for the GPU, torch with ROCm, CUDA or Apple MPS, or the ggml Vulkan engine
   (`experiments/universal_kv/ggml`, no torch needed; build steps in its README). `--device gpu`
   uses torch (float16 weights with float32 accumulation), `--device vulkan` the ggml engine
   (the same precision), `--device cpu` numpy, and `--device auto` (the default) torch's GPU,
   else a Vulkan GPU through ggml, else the CPU. On the Radeon 8060S a warm 13K-token
   translation took 0.42-0.56 s with torch, 0.62-0.82 s with ggml-Vulkan and 3-4 s on the CPU;
   in a fresh engine process ggml was the fastest (1.1-1.3 s against torch's 2.3-2.6 s). Other options:
   `--mode daemon|oneshot`, `--idle-unload S`, `--idle-exit S`, `--threads N`; see
   [The engine](#the-engine). Settings not given keep their values, so
   `kvpark bridge engine --device cpu` later changes only the device.

4. Check:

   ```bash
   kvpark bridge status
   ```

   It shows the models, the hub and the engine daemon (`engine_daemon`: running or not,
   device, jobs). `engine_check` reports the engine Python's numpy BLAS and whether it sees
   a GPU (a running daemon answers; otherwise the engine Python is asked directly, which takes
   a few seconds when it imports torch). `warnings` flags an engine Python that cannot be
   asked (the check failed), a reference-BLAS numpy, a GPU asked for but missing, a GPU the
   daemon stopped using after an error, a daemon that cannot run here (no `engine_daemon.py`
   beside the engine script, or a bridge directory whose socket path would be too long:
   every switch then runs the slower one-shot engine), and a format 1 hub.

## Switch

Write the conversation as JSON in the OpenAI chat format (content parts, tool calls and tool
results as in a chat request), ending with the message the new model should answer:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."}, {"role": "user", "content": "now you"}],
 "tools": null}
```

```bash
kvpark bridge switch --from qwen38 --to muse --conversation conversation.json
```

The report shows the tokens restored (`restored_tokens`, = T0), how many prompt tokens the
target reused on the warm request (`reused_tokens`, expected T0), the target's draft model
(`draft`: kind, and for DFlash the window), whether the target writes `.resume` sidecars
(`resume_sidecar`), how the engine ran (`engine_mode`: `daemon` or `oneshot`, with
`daemon_error` when the daemon was unavailable; `engine.backend` and `engine.device`), the
native prompt start (`native_prefix_tokens`), what a hybrid target's recurrent cell started from
(`recurrent`), how the request's template settings were handled (`template_options`, see
[Request template settings](#request-template-settings)), the conversation's media and their
surrogates (`media`, `surrogates`, see [Media across models](#media-across-models)), the
engine's `features` when the switch needed to know them (`engine_features`), and the time of
each step. Send the conversation to the target's server as usual afterwards; it reuses the
cache. When the conversation holds media the target cannot read, send it as the switch
rendered it, with their text surrogates: `--request-out PATH` writes it. A switch the bridge
does not translate reports `"skipped"` with a reason code and a `"note"`; the target then
simply prefills the conversation.

## The engine

The translation runs in an engine daemon, `experiments/universal_kv/engine_daemon.py`,
started by the first switch (or by staging one in Hermes) with the configured engine Python.
It keeps the hub's model sides and the GPU (or numpy's BLAS threads) warm between switches,
so a switch pays only the arithmetic. For 10-13K translated tokens between Qwen3.8-27B and
Muse-Glimmer-30B that was 0.45-0.9 s (Qwen3.8 → Muse) and 0.55-2.4 s (Muse → Qwen3.8) on the
Radeon 8060S GPU, and 2.1-6.4 s on the CPU, against 13 s or more for a fresh engine process.
Dense models cost more: Qwen3-4B keeps 36 full KV layers (1.3-1.6 GB of cache at 11-13K
tokens), and switches into or out of it took 1.1-3.7 s on the GPU and 4.9-13.6 s on the CPU in
the first live runs, 1.4-2.0 s of it the assemble stage (RoPE at the target positions, the
cache cast and the copy into the output arrays). The GPU backend now rotates keys in float32
with fused multiply-adds (on the 8060S exactly the bits of its earlier float64 rotation; every
rotation is checked against the CPU reference on first use) and touches the fresh output pages
on four threads while the GPU decodes: assemble takes 0.13-0.15 s on the same jobs (0.23-0.36 s
before) and the engine step into Qwen3-4B 0.77-1.10 s live. With another process busy on the GPU every stage slows down (assemble 0.6-2.1 s
measured before this change). A DeepSeek V4 target always runs on the CPU.

- **Where.** One daemon per bridge directory (the directory of the bridge configuration),
  listening on `engine.sock` there (owner-only directory and socket; on Linux every client's
  user id is checked). The job travels over the socket, so the conversation is never written
  to disk. Its log, `engine.log` in the same directory (owner-only, rotated at 8 MB), holds
  timings and sizes only.
- **Status.** `kvpark bridge engine-status` prints the daemon's state: device, resident hub
  sides and their memory, jobs and failures, calibration of its memory guard, idle time,
  and the engine Python's BLAS. `kvpark bridge engine-stop` stops it; the next switch starts
  it again. Changing the engine with `kvpark bridge engine` stops a running daemon, so the
  next switch starts one with the new settings.
- **Idle.** After `--idle-unload` seconds without work (default 120) the daemon drops its hub
  sides and gives the GPU memory back; after `--idle-exit` seconds (default 600) it exits.
  Status requests do not count as work.
- **Memory.** Before each job a memory guard compares the job's estimated peak with the free
  memory (and the GPU's). It first drops hub sides the job does not need, then moves the job
  to the CPU, and finally refuses it: the switch is then skipped (`engine-memory`) and the
  target prefills natively. A refused job is never rerun outside the guard.
- **Fallback.** Where the daemon cannot run (Windows, a bridge directory whose socket path
  would exceed about 100 bytes, `--mode oneshot`, or an engine script without
  `engine_daemon.py` beside it) or when it cannot start or reports a fault of its own, the
  switch runs the engine as a one-shot process instead, within what is left of the engine
  timeout (`engine_mode` in the report; `daemon_error` says why). A job the daemon itself
  rejects fails the switch like a failed one-shot engine (a rendering it cannot reproduce is
  a skip instead, `template-mismatch`), and so does a job the daemon took and then overran
  the timeout on (the daemon is stopped) or died during: rerunning it would wait as long
  again, or repeat what stopped the daemon (running out of memory, say) without the memory
  guard. In Hermes a failed switch means a full prefill.
- **Features.** The daemon's ping and `kvpark bridge engine-status` list what the engine
  handles beyond older releases (`features`: `template-options`, `media-source`,
  `recurrent-mean`); the bridge decides those switches by them (see
  [Skipped switches](#skipped-switches)).
- **Devices.** `--device` (or the environment variable `KVPARK_ENGINE_DEVICE`) picks
  `gpu`, `vulkan`, `cpu` or `auto`. A GPU error that looks like a broken device moves the daemon to the
  CPU until it exits (`gpu_fault` in its status); the job itself is redone on the CPU.
- **The torch daemon keeps a core busy.** On ROCm (measured on the Radeon 8060S with torch 2.12),
  every torch process keeps one runtime thread at 100% of a core from its first GPU kernel on, also
  while it waits: a resident torch daemon does so until it exits (`--idle-exit`, 600 s after the
  last switch by default), including after its idle unload. On October 9 that thread alone held
  this machine's CPU at 93 °C at 48 W with the llama-servers idle. `--device auto` prefers torch;
  on an AMD APU whose cooling is tight, choose `--device vulkan` (the ggml engine has no such
  thread; a warm switch takes 0.6-0.8 s instead of 0.4-0.6 s) or a shorter `--idle-exit`.
- **Threads.** The CPU path uses one BLAS thread per physical core, at most 16
  (`--threads N`, or `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` or `MKL_NUM_THREADS` in the
  environment, overrides it; one set there is copied to the unset ones). The servers
  are idle while a switch translates; on the 16-core Ryzen AI Max+ 395, 16 threads translated
  13K-token switches 24-30% faster than 8 (2.1 s against 2.8 s Qwen3.8 → Muse, 2.8 s
  against 3.9 s Muse → Qwen3.8).
- **Prewarm.** Staging a switch (`/kvpark switch` in Hermes) starts the daemon in the
  background and loads the pair's hub sides. It also runs the switch's products once on
  zero rows of its shapes, because the first product of each shape on a GPU picks and loads
  its kernels. The first engine job after staging took 0.51-0.52 s (Qwen3.8 → Muse) and
  0.76 s (Muse → Qwen3.8), against 0.45-0.47 and 0.72-0.78 s warm; without the warm-up it
  took 0.69-0.71 and 0.95-0.96 s. (The last window of a job has a row count that depends on
  the conversation, so its shape can still be new.) The prewarmed sides stay loaded only for
  `--idle-unload` seconds (default 120) and the daemon for `--idle-exit` (600), while a
  stage lasts 15 minutes: a switch made later than that loads the pair itself (0.2-8 s in
  the live runs, plus a few seconds to start a daemon that has exited). Raise
  `--idle-unload` if you usually stage well before switching; the sides hold memory
  meanwhile (1.2 GB for Qwen3.8 and Muse on the CPU).

## Hybrid targets: the recurrent state

A hybrid model (Qwen3.5/3.8, Ornith, and the other architectures with gated DeltaNet, Mamba,
short-conv, lightning or KDA layers) keeps, next to its attention cache, a recurrent state that
summarizes the whole conversation and has no per-token form, so it cannot be translated. The
translated cache holds one recurrent cell at the handover point T0 - 1, and the target's native
tail carries it on from there. By default that cell is zero, as for a fresh conversation. A
mean of the model's own native cells does much better:

1. Build the mean once per model from training conversations in the format of
   `experiments/linear_transfer/chat.py` (their train split is used; no download; a GPU prefill
   with a Vulkan build of `kvpark-kv-probe`, about 18 minutes for Qwen3.8-27B):

   ```bash
   experiments/linear_transfer/.venv/bin/python experiments/universal_kv/r0_experiments.py calibrate --model /path/to/model.gguf --name qwen38 --corpus train.jsonl --probe /path/to/kvpark-kv-probe --output ~/kvpark-means/qwen38
   ```

   It saves the model's cells every 1,024 tokens and writes `recurrent-mean.npz` with the groups
   `all`, `chat` and `tools` (78 MB per group for Qwen3.8 in float16).
2. Register it: `kvpark bridge add qwen38 ... --recurrent-mean ~/kvpark-means/qwen38/recurrent-mean.npz`.
   The file is checked against the model's GGUF (architecture, layer count, and for `qwen35`,
   `qwen35moe` and `qwen3next` the recurrent layers and row sizes); models without a recurrent
   state refuse it.

Per switch the engine picks the group: `chat` for a conversation without tools; with tools the
pooled `all`, or `tools` when the model was registered with `--recurrent-tools` (the file's
`tools` group was calibrated on this deployment's own agent prompts and tools). A file without
the chosen group falls back to `all`; one that lacks that too, or was moved or damaged since it
was registered, gives zeros with a note, and the switch still happens. The report's `recurrent`
says what the cell started from (`strategy`, `group`, `note`). The engine daemon keeps a loaded
group until its idle unload (loading takes 0.2 s for Qwen3.8). An engine without the
`recurrent-mean` feature ignores the mean and writes zeros; the report then says so.

Measured on Qwen3.8-27B and Ornith-1.5-9B, 12 held-out conversations of 8.5-16.8K tokens
([RESULTS.md, section 7](../experiments/universal_kv/RESULTS.md)):

- With the target's own attention cache, so only the recurrent cell differs, zeros cost Qwen3.8
  +0.053 / +0.039 / +0.029 nats/token on chats at native tails of about 1.4K / 2.4K / 4.4K
  tokens, the matched mean +0.008 / +0.002 / +0.003; on Ornith zeros cost +0.065 / +0.035 /
  +0.021 and the mean -0.006 / -0.005 / -0.005, level with native. The mean's KL to the native
  prefill was lower than zero's on every document at every tail.
- After a real translation (Muse → Qwen3.8) the mean takes 0.03-0.04 nats/token off chats
  (+0.129 / +0.102 / +0.061 → +0.097 / +0.064 / +0.027).
- On agent traces the pooled mean did no better than zeros on Qwen3.8; a `tools` mean
  calibrated on the traces' own prompt and tools cut KL by more than half. Hermes' agent traffic
  has not been calibrated, so it gets the pooled mean.
- Live, Muse → the production Qwen3.8 registered with its mean: 6 of 6 switches reused exactly
  T0, with the `chat` group for chats and `all` for a tool-calling trace.

The native tail is not changed for you. With the mean, chats keep their quality on a shorter
tail (1,024 tokens with the mean beat about 4,400 with zeros), so `--tail 1024` suits a hybrid
target without a DFlash draft that serves chats. Conversations with tools keep it only with
`--recurrent-tools`: with the pooled mean, Qwen3.8's agent traces scored KL 0.069 at 1,024
against zeros' 0.057 at the default 2,048. A DFlash draft keeps its window rule either way, so
the production Qwen3.8 keeps 2,048.

## Skipped switches

The bridge skips a switch it cannot translate well or at all, and the target prefills the
conversation natively, which is always correct. A skipped switch leaves the source's cache as
it was (when the source described media for the target first, its slot was saved before that
side request and restored after). Most skips are decided before the target is touched either. Two come later:
`draft-unsupported`, and `short` because of the DFlash window. Both are decided after the
bridge has made the target's layout template. On the first switch into a target server, that
template replaces the target's resident conversation, as the switch itself would have. Three
come from the engine, after the source's slot was saved: `engine-memory`, `template-mismatch`,
and `no-indexer-keys` when the engine refuses what the GGUF header let through. When the engine
cannot render the request's template options and the bridge falls back to the default
rendering, that fallback is planned again after the save too, and can end in
`template-options` or `short`.

Skips that depend on the engine (`media-source`, `template-options`, `tokenizer`) are decided by
what the engine that runs the job advertises: the daemon's `features` (`kvpark bridge
engine-status`, `engine_daemon.py`'s ping), or the `FEATURES` an engine script declares when jobs
run one-shot. An engine from before a feature keeps the skip.

| `skipped` | When |
| --- | --- |
| `source-refused` | the source is registered as a target only (`kvpark bridge add --no-source`): its cache is never translated. Decided before any server is touched. For a model the hub translates out of worse than no translation: a campaign report's "Bridge settings" add the flag when a switch out of the model did worse than the floor |
| `short` | the target would translate fewer than `min_handover` tokens (default 1024) after its native tail, or before the first message with media; a native prefill is about as fast |
| `no-per-token-cache` | source or target is a pure recurrent model (Mamba, Mamba2, RWKV6/7, ARWKV7): its whole history is one fixed-size state, with no per-token cache to translate |
| `dsv4-source` | the source is DeepSeek-V4: it keeps per-token latents only for the last 128 tokens and pools the rest |
| `embedding-model` | source or target is an embedding model: its vocabulary appends EOS to every text it tokenizes (`tokenizer.ggml.add_eos_token`, e.g. gte-Qwen2), so its chat prompts cannot be spelled back. Decided from the GGUF before any server is touched |
| `rope-factors` | source or target is a LongRoPE model: a `longrope` RoPE scaling type, or `rope_factors_long`/`_short` and no `rope_freqs.weight` in a Phi-3/4, Llama, Mistral 3, Gemma 4 or granite GGUF. llama.cpp picks its RoPE frequency factors by the context size, which the engine does not model. Decided from the GGUF before any server is touched |
| `unsupported-arch` | the engine cannot translate this architecture's cache (it translates the architectures listed under [What it can and cannot do](#what-it-can-and-cannot-do)), or the GGUF cannot be read |
| `tokenizer` | the engine cannot read the source's or the target's vocabulary: an engine older than the `vocabularies` feature reads GPT-2 byte-level BPE only (not Gemma 4's or SentencePiece vocabularies); no engine reads other tokenizer models yet |
| `no-head-decoders` | the target is DeepSeek V4 and the hub lacks any of its three head decoders (`csa`, `hca`, `lid`; `fit_hub.py --heads`): its pooled caches are synthesized from per-token compressor inputs the hub decodes |
| `no-indexer-keys` | the target is a DSA model (`deepseek32`, `glm-dsa`, `dots3note`, `hy_v4` with an indexer; `qwen4exp`'s block indexer) whose prompt is longer than its indexer `top_k`, and the hub has no indexer ridge for it (`fit_hub.py --indexer-ridge`): its indexer would rank the translated cells by keys the engine cannot write. Decided from the GGUF header and the prompt length before the source is saved; the engine's own refusal gives the same skip |
| `target-no-vision`, `target-no-audio`, `target-no-video` | the conversation has that kind of media, the target's server has no encoder for it (no `--mmproj`), and no text surrogate replaces it: the media policy for it is `skip`, or nothing could make one (no cached one, and neither the source nor a describer has that encoder). See [Media across models](#media-across-models) |
| `dsv4-media` | the target is DeepSeek V4 and would read media itself (text surrogates are fine): media after a translated pooled cache has never been restored, so the target prefills it natively |
| `media-source` | only with an engine older than the `media-source` feature: the source has already processed the conversation's media (it has the encoder and the media sits in a message before its last answer). A current engine reads the source's cache up to its first medium instead |
| `text-parts` | a message gives its text as a list of parts and the target's template renders such lists differently from plain text |
| `draft-unsupported` | the target's server runs a draft model whose state the bridge cannot write (anything but DFlash/DSpark, MTP or none; e.g. a separate full-attention draft model, or an EAGLE3 or plain draft whose state looks like MTP's; for a DeepSeek V4 target any draft, DSpark and MTP included) |
| `target-no-media-marker` | the target's server does not report its media marker (`/props`), so media cannot be placed |
| `template-options` | the request carries `chat_template_kwargs` or `reasoning_effort` that change how the target renders it, and the engine cannot render with them (an engine older than the `template-options` feature, or a server whose own template settings the engine cannot reproduce, see below). The engine then translates the template's default rendering, which the target reuses only up to where the two renderings part; here fewer than `min_handover` tokens would be left (Qwen3.8 writes the reasoning effort into its system prompt). Options that change only the end of the prompt (a classic Qwen3 template's `enable_thinking: false`, which adds an empty think block after the generation prompt) do not shorten the translated part |
| `template-mismatch` | the engine's renderings of the conversation do not match the target server's tokens even without request options: the server renders with settings the engine cannot see (`--chat-template-kwargs`, a `--reasoning-effort` its template reads, `--reasoning off` or `--no-reasoning-preserve` the model was not registered with, see `kvpark bridge add`; a `--chat-template` other than the GGUF's) or a template the engine's Jinja renders differently (decided after the source's slot was saved) |
| `engine-memory` | the engine daemon's memory guard refused the translation: not enough free memory (decided after the source's slot was saved, which leaves the source's cache unchanged) |

The rules for architectures live in one table, `src/kvpark/bridge/capabilities.py`.

## Draft models (DFlash, MTP)

A target's server may draft with a speculative model: DFlash/DFlash2 (`-md` with a `dflash`
draft) or MTP (`--spec-type draft-mtp`, the model's own next-token head). The bridge detects
the setup from the target's own one-token template save: on kvpark's patched server from the
draft state in its `.resume`, on an upstream server from `/slots` (`speculative.types`).
`/props` cannot be used; it always reports `none`.

- **Upstream servers** neither save nor restore draft state: after a restore the draft is
  rebuilt from the native tail.
- **kvpark-patched servers** (`runtime/` here, on either llama.cpp version, and the production
  Qwen3.8 DFlash2 build) write `<slot>.resume` on every save and refuse a restore without it
  ("Missing slot resume file").
  With a draft model the file must hold the draft's state. For a translated cache the bridge
  writes `<slot>.resume` next to the translated slot (owner-only, deleted after the restore,
  like every bridge file) with a draft state synthesized in the layout of the template's own
  save, no speculative state and no checkpoints. The template's `.resume` is cached with the
  template; the native-prefix save is only read by the engine, never restored, so it needs none.
- **DFlash** keeps per-position draft keys and values computed from the target's own hidden
  states, every layer on a sliding window (2048 tokens for the local Qwen3.8 and Muse
  drafts). The bridge keeps the native tail at least `window - 1` tokens long, so at the first
  draft the draft's whole window holds only positions the target prefilled itself. Their
  hidden states still depend on the translated cache through the target's attention, so they
  are close to native, not identical. The restored draft state holds two zero cells at
  positions T0 - 2 and T0 - 1: they are never inside a later window, and they keep the
  draft's last position equal to the target's (the production build counts draft positions
  from the draft's own last position). Set `--draft-window N` when a draft uses another
  window. On the production Qwen3.8 build (its DFlash2 draft counts positions from its own
  last one) the 41 KB aligned dummy restored every time, the server logged no position
  warning or error, and over three conversations of 12-15K tokens DFlash2 accepted 37% of
  its drafted tokens in the 256 tokens after a translated restore against 41% after a native
  prefill (per conversation the difference went both ways; the answers themselves differ).
  The dummy itself costs nothing: with the target's own cache of the same tokens restored the
  same way, DFlash2 accepted 43.4% with the dummy and 43.7% with the server's real draft
  state.
- **MTP** keeps a full-attention cache for its head. The bridge restores it empty (zero cells
  would be attended forever and dilute it): the head then sees only the native tail and what
  follows, so expect lower acceptance than natively for the rest of the conversation. Output
  is unaffected. Measured on the production Qwen3.8 build with its MTP head (three
  conversations, 256 tokens each): 48% of drafted tokens accepted after a translated restore
  against 61% after a native prefill, and 14.7-16.3 against 15.4-20.5 generated tokens/s.
  The empty state accounts for about 5 points of that: the target's own cache restored the
  same way accepted 55% with the empty marker and 60% with the server's real MTP state,
  generating the same text; the rest goes with the translated cache and the text it leads to.
  An EAGLE3 or plain draft model whose state also has one layer is told apart by `/slots`
  (`draft-eagle3`, `draft-simple`) and skipped.
- **Gemma 4's MTP head** (`gemma4-assistant`) keeps no cache of its own: it reads the target's
  last sliding and full layers, so its draft state is an 8-byte header. A translated restore
  gets the same 8 bytes and the head reads the restored target cache. On the real Gemma 4 26B
  (CPU) a translated-shape slot continued token for token like the resident one with its MTP
  head. A server that names another speculative type while saving such a blob is skipped
  (`draft-unsupported`).
- **DeepSeek V4 targets** restore a translated slot only without a draft: its DSpark and MTP
  drafts keep their own window of target features, which has not been restored around a
  translated slot, so both are skipped (also on upstream servers, told by `/slots`).
- The bridge always erases the target slot before restoring; that also clears the draft's
  cache.

The template is cached per weights file, server build, slot count and speculative setup. A
template from before a server restart with other settings (flash attention, draft model)
makes the restore fail; the bridge then drops it, and the next switch makes a new one. A
cached template whose `.resume` cannot be read (a save cut short) fails the switch and is
dropped the same way.

## Media across models

**The short answer.** When a conversation holds an image, audio or a video and moves to a
model that cannot read that kind of media, the bridge puts a text stand-in (a *surrogate*) in
its place: `[Image description: ...]`, `[Audio transcript: ...]` or `[Video description: ...]`.
The source model writes it when it can read the media (Qwen3.8 describes its own images), else
a describer you name does; it is made once and cached. The target model then reads the
surrogate as ordinary text, the switch goes ahead, and the real request carries the same
surrogate, so the target's prompt is exactly what the switch translated for. A target that
*can* read the media gets the media themselves, as before. Before this, every such switch was
skipped, and the request itself failed on the target.

### What each target gets

| The conversation holds | Text-only target (Muse, Qwen3-4B) | Vision target without video input | Audio target without vision | Target with that input |
| --- | --- | --- | --- | --- |
| an image | `[Image description: ...]` | the image | `[Image description: ...]` | the image |
| audio | `[Audio transcript: ...]` (speech verbatim; "No speech." and a sentence about the sound otherwise) | the same transcript | the audio | the audio |
| a video | `[Video description: ...]`, and `[Video audio transcript: ...]` when its sound can be transcribed | its keyframes as images (with ffmpeg installed: one per 2 s, at most 8, 512 px wide) after `[Video keyframes at 1 s, 3 s, ...]`, plus the sound's transcript; without ffmpeg the description | the description and the transcript, as for a text-only target | the video |

"Vision target without video input" means a vision server built without `MTMD_VIDEO`. That
option is on by default wherever llama.cpp can run subprocesses (Linux, macOS, Windows; off on
iOS, Android and WebAssembly; `tools/mtmd/CMakeLists.txt:5`, pinned and upstream), so every
vision server on this machine also reports `video`: Qwen3.8 (its production build and
`runtime/build`) and Gemma 4 26B/31B on a default build read a video themselves, through the
last column, as frames only (llama.cpp never reads a video's sound track, see below). The
keyframes column applies to a vision server built without it. An audio-capable target without
vision (no local model is one) gets a video's sound track as a transcript too, not as audio.

Before this change (the bridge as first committed), every cell left of the last column was a
skipped switch (`target-no-vision`, `target-no-audio`, `target-no-video`), and llama-server
itself refused the real request: a server without the encoder answers a request with an image
with HTTP 500 `image input is not supported - hint: if this is unexpected, you may need to
provide the mmproj` (the same for audio and video). Unless the client converts the media
first, that conversation could not continue on the target at all.

A message whose media all became surrogates becomes one string, its text parts and surrogates
joined by newlines. A message that keeps some media (an image for an image+text target next
to audio it cannot read) stays a list, with each surrogate as a text part where its medium
was.

### How llama.cpp takes media

Both the current pin (a7b94df) and the previous one (85d5703a3) read three OpenAI content parts:
`image_url` (vision), `input_audio` (audio) and `input_video` (video; upstream also accepts
`video_url` as an alias, `tools/server/server-common.cpp:1182-1196` upstream). Each needs its
modality in the server's `/props` `modalities` (pinned `server-context.cpp:4693-4698`, upstream
4837-4842); without it the server throws (pinned `server-common.cpp:1216-1256`, upstream
1154-1196), which reaches the client as HTTP 500 (`tools/server/server.cpp` `ex_wrapper`); a part
type it does not know is HTTP 400. Video is not its own encoder: a server has `video` when it
has vision and was built with `MTMD_VIDEO` (`mtmd-helper.cpp` `mtmd_helper_support_video`; on
by default, see above; the server then needs ffmpeg in its PATH to decode a video). The
video helper runs ffmpeg for frames only, 4 per second by default (`--video-fps`), with a
`[0m5.00s]` time stamp every 5 s (`--video-timestamp-interval`), and never reads the sound track
(upstream `mtmd-helper.cpp:720-760, 836-872`). Audio needs an audio projector: Whisper-style
(Ultravox, Qwen2-Audio, Voxtral, Qwen3-ASR), Gemma 4 E2B/E4B, or Qwen2.5/Qwen3 Omni
(upstream `docs/multimodal.md`).

On this machine: Qwen3.8's projector is vision only (`qwen3vl_merger`; its builds also report
`video`, as frames), Gemma 4 26B and 31B's projectors are vision only (`gemma4v`, read from
their GGUF headers on the Windows SSD), and Muse and Qwen3-4B run without one. No local model
takes audio.

### Where the surrogates come from

For each medium the target cannot read, the first of these that has one wins:

1. **The conversation.** A text part right after the media part that already starts with the
   marker (`[Image description:` ...): the media part is dropped and that text used. (Hermes'
   own conversions carry no such marker, but they remove the media part as well, so there is
   nothing left to replace; see [The real request](#the-real-request).)
2. **The surrogate cache.** One made before for the same media bytes (their sha256), the same
   kind and the same prompt version, by any producer. Later switches and later turns get the
   same text, which keeps the target's prompt equal to its cache. The same medium twice in one
   conversation is made once (the status calls the second "repeated").
3. **The source model**, when its server has that input. One side request at temperature 0 with
   thinking off: the media and a fixed prompt ("describe this image for a reader who cannot
   see it and will be asked about it later; transcribe all visible text ...",
   `surrogates.PROMPTS`). The source's slot is saved first (the save the switch translates
   from) and restored from that file afterwards, so the conversation the source holds is
   unchanged; the switch reuses that save instead of saving again. Measured on the production
   Qwen3.8: a save after the restore is byte-identical to the save before the side request,
   draft state (`.resume`) included. If that save fails, the switch fails with the save's error
   (the switch needs the same save anyway) rather than handing the media to a describer.
4. **The describer** set for that kind of media (`kvpark bridge media --describer`): a
   registered model (its slot is saved and restored the same way), a local llama-server URL
   (its resident conversation on slot 0 is replaced by the side request, so use a server that
   does not hold one you need), or, for audio, a local OpenAI-compatible speech-to-text
   endpoint (`http://127.0.0.1:PORT/v1/audio/transcriptions`, for example Lemonade's Whisper on
   the NPU or a faster-whisper server). The bridge uploads the audio as a file with
   the `model` set by `--transcription-model` (default `whisper-1`), `response_format=json` and
   temperature 0, and takes the answer's `text` (empty: "No speech."). It takes inline audio
   only. This is tested against a stand-in server, not yet against a real one.
5. **Nothing**: the switch is skipped with `target-no-vision`, `target-no-audio` or
   `target-no-video`, and the request goes out unchanged.

A video is described by a producer with video input; failing that, a vision producer reads its
keyframes in one request (with ffmpeg); its sound track is transcribed by a producer with
audio input (with ffmpeg). When the video has sound that nothing could transcribe, its
surrogate says so (`[Video audio transcript: unavailable (the video has a sound track that
could not be transcribed)]`, `partial` in the report, "the sound of 1 not transcribed" in the
status), so the target knows it cannot hear it; a later switch or request with a producer for
audio completes that surrogate in the cache (the target then re-reads the conversation from
that message once). Without ffmpeg the bridge cannot tell whether a video has sound; like
llama.cpp's own video input, it then leaves the sound out. Media given by URL (`https://`,
`file://`) are passed on to the producer, which fetches them itself; the bridge never
downloads anything, and keys such media by their URL (a changed file behind the same URL
keeps its old surrogate until `--clear-cache`).

The cache is `surrogates/` next to the bridge configuration (private: the directory 0700, each
entry 0600), at most 64 MiB (`--cache-mib`), least recently used first out. It holds the
surrogate texts and, for vision targets, a video's keyframes (JPEG), keyed by the media's
digests; never the media themselves.

### Where the translated cache stops

The switch translates the conversation as the target will read it, surrogates included.

- **The source read the media itself** (Qwen3.8 → Muse with the image in an earlier message): the
  source's cache holds the image's own cells, which no other model can use. The translated
  cache stops before the message that held the image (`max_t0_message`), and the target
  prefills that message, with the surrogate, and everything after it.
- **The source read text** (Muse → Qwen3-4B, both text-only, the image described earlier): the
  source's cache holds the same surrogate text, so it is translated like any other text and
  nothing stops at it.
- **The target reads the media** (any source → Qwen3.8 with an image): unchanged from before, see
  [Images the target reads](#images-the-target-reads).
- An image only in the newest message costs nothing either way: the newest message is always
  prefilled natively.
- A warm request (`kvpark bridge switch` without `--no-warm`) is sent when the target is left
  with no media of its own.

### The real request

The target must receive the surrogates too, or its prompt differs from the translated cache
(and an image it cannot read fails the request). In Hermes the middleware does it for every
request to a registered bridge model whose server lacks a modality the request holds: each such
part becomes its cached surrogate (or one made now by the staged switch's source or a
describer). Hermes' own history keeps the media. Outside Hermes, `kvpark bridge switch
--request-out PATH` writes the conversation as the target must receive it (asking the source
too, when the switch was skipped before it made the surrogates).

`/kvpark switch status` shows what happened, for example `media: 1 image as text (made by qwen38
in 23.2 s)` after a switch, and for later requests `Last request with media for muse: media: 1
image as text (1 from the surrogate cache or the conversation)`, or why a request went out
unchanged.

How Hermes itself handles media matters here. For a model that is not marked
`supports_vision`, Hermes replaces images with its own description (`vision_analyze` through its
auxiliary vision route) before kvpark sees the request
(`agent/vision_message_prep.py` `_prepare_messages_for_non_vision_model`, applied in
`agent/chat_completion_helpers.py` before the request middleware runs); `local-muse` is such a
model in your configuration. Voice notes are transcribed by Whisper on the NPU before the
agent runs (`gateway/run_inbound.py`), and video and audio files reach the model as file paths
(the `video_analyze` tool reads videos on an auxiliary model). So in your Hermes setup a request
to Muse already carries text; the bridge then has nothing to replace, and the engine reads the
source's cache up to its first image (`source_media`) as described above. The bridge's
surrogates matter when the request still holds the media: a model marked `supports_vision`
whose server has no `--mmproj`, audio or video parts from other clients, or the `kvpark bridge
switch` command. One difference: Hermes describes at temperature 0.1 (`tools/vision_tools.py`
`_aux_call_kwargs`) and keeps its descriptions in memory only, so a description can change after
a restart, and the target then re-reads the conversation from that message; the bridge's
surrogates are made at temperature 0 and kept in its cache.

### Policy

```
kvpark bridge media                                   # show the settings, the cache and ffmpeg
kvpark bridge media --image caption|skip|default      # default: caption
kvpark bridge media --audio transcribe|skip|default   # default: transcribe
kvpark bridge media --video describe|skip|default     # default: describe
kvpark bridge media --model muse --image skip         # for one target only (default drops it)
kvpark bridge media --describer vision=qwen38         # who describes when the source cannot
kvpark bridge media --describer audio=http://127.0.0.1:8095 --describer video=
kvpark bridge media --describer audio=http://127.0.0.1:13305/v1/audio/transcriptions \
                    --transcription-model whisper-v3-turbo-FLM   # a local speech-to-text server
kvpark bridge media --cache-mib 128 --clear-cache
```

`skip` keeps a kind of media as it is: the switch is skipped as before, and the request goes out
with the media. `kvpark bridge status` shows the same settings under `media`.

### Costs and limits

- **Time.** A surrogate costs one side request on the producer, once per medium: on the
  production Qwen3.8 (DFlash2 draft, ROCm), describing the 480×320 chart (an 855-character
  description, 198 tokens; the server's own times) took 9.8 and 9.9 s in two runs on an
  otherwise idle GPU (23 tokens/s) and 21.4 and 21.9 s in two runs while other work shared the
  GPU (10.6 tokens/s), plus the save and restore of the source's slot around it (0.5-1.3 s each
  for 13.6K-16.2K-token slots of 1.05-1.22 GB). A Hermes switch
  runs it while the request waits; later switches and turns read it from the cache. The switch
  itself then costs what any switch costs (below).
- **What is lost.** The target reads a description, not the image: it knows what the
  description says, nothing more, and repeats its mistakes (in the one measured case the count,
  the colors and the tallest bar were right and one comparison was wrong). Audio becomes its
  words; tone of voice is lost.
- **Sound in video.** Transcribed only with ffmpeg and a producer with audio input. No local
  model has audio input; a speech-to-text describer (above) is the way to get it on this
  machine. A target that reads the video itself never hears its sound (llama.cpp reads frames
  only), and the bridge adds no transcript for it.
- **Determinism.** Surrogates are made at temperature 0 and cached, so the same media give the
  same text on every later turn. A new prompt version (`surrogates.PROMPT_VERSION`) makes new
  ones.
- **Privacy.** The cache holds text derived from your conversations (descriptions,
  transcripts) and keyframes of videos, privately; `kvpark bridge media --clear-cache` deletes
  them.

Measured live, Qwen3.8 (production build, vision) → Muse (Vulkan, text only), one model loaded at
a time (Qwen3.8 primed the conversation and described the image, then stopped; a stub handed its
saved slot to the switch), on a held-out chat of 8 messages with the generated bar chart (5
bars, the red one tallest) in message 3 and a question about it at the end (13,611 tokens for
Qwen3.8 with the image's cells, 12,958 for Muse with the description):

- **Today, without surrogates**, Muse refused the request: HTTP 500 `image input is not
  supported - hint: if this is unexpected, you may need to provide the mmproj`.
- **The surrogate.** Qwen3.8 described the chart in 21.4 s (the server's time, with the machine
  busy: 10.7 tokens/s against 23 on an idle GPU, where the same request takes 9.9 s): the five
  bars, their colors and the red one as the tallest, right; one comparison wrong (it called the
  purple bar taller than the orange one). With the save (1.1 s) and restore (1.0 s) around it,
  23.4 s. A save of Qwen3.8's slot after the
  restore was byte-identical to the save before the side request (1.05 GB, and its `.resume`).
  Qwen3.8 itself, seeing the image, answered "5 bars, red".
- **The switch** (the surrogate from the cache): 1.9 s in all, 1.0 s of it the engine on the
  GPU daemon, 0.03 s the restore (Qwen3.8's own slot save takes 1.2 s; here a stub handed over
  the saved file). T0 = 6,900 tokens, stopped before the image's message because Qwen3.8's
  cache holds the image's cells there.
- **The real request**, rewritten with the surrogate as the Hermes middleware rewrites it,
  reused exactly the 6,900 translated tokens and prefilled the other 6,058 in 41.1 s, against
  76.9 s for Muse's native prefill of the same request. Both answered "the chart has five bars.
  the tallest bar is red.", Muse's reasoning quoting the description.

Counting the prompt only (the description is needed either way): 23.4 + 1.9 + 41.1 = 66 s before
Muse's first token after the switch, against 23.4 + 76.9 = 100 s natively, in this one sample
on a busy machine (the description alone runs twice as fast on an idle GPU). An image later in
the conversation leaves more to translate.

Evidence: `evidence/universal-kv/live-media-surrogates-20261004/` (`surrogates.jsonl`, server
logs), `experiments/universal_kv/live_bridge.py surrogates`.

The run above made the surrogate outside the switch and gave the switch a stub of Qwen3.8's
saved slot. The Hermes path itself was then run live with both servers up (Qwen3.8 production
build as the source, Qwen3-4B on Vulkan as the text-only target, so one 27-30B model ran) and an
empty surrogate cache, through `hermes.Adapter.bridge_switch` (a staged switch, as
`/kvpark switch` stages it) and `bridge_media`, with the image in message 15 of a 26-message
chat (16,176 tokens on Qwen3.8, 8,688 on Qwen3-4B with the description):

- Qwen3-4B refused the raw request (HTTP 500, as Muse did).
- **The switch made the surrogate itself**: Qwen3.8's slot saved (1.3 s, the save the engine
  then translated), the side request, the slot erased and restored (1.3 s); 24.6 s in all, while
  another job was benchmarking Qwen3.8 on the same GPU (21.9 s for the side request at 10.6
  tokens/s; on an idle GPU the same request took 9.8 s in an earlier run, at 23.3 tokens/s, with
  0.5 s each for the save and restore of a 13.6K-token slot). The description was byte for byte
  the one of the earlier runs.
- The whole switch took 29.1 s (engine 4.1 s on the GPU daemon). T0 = 6,282 of 8,688, stopped
  before the image's message.
- `bridge_media` rewrote the request from the cache (2 ms), and the real request reused exactly
  T0 (`cache_n` 6,282), prefilling the other 2,406 tokens in 4.7 s against 12.4 s for a native
  prefill of 8,688. Both answered five bars, the red one tallest.
- Qwen3.8's slot saved after all this was byte-identical to its save before the switch (1.22 GB,
  and its `.resume`), and no bridge file was left in its slot directory.
- `/kvpark switch status` read: `Last switch qwen38 → qwen3-4b: restored 6,282 of 8,688 prompt
  tokens in 29.112 s (engine: daemon on the GPU, 4.054 s; media: 1 image as text (made by qwen38
  in 23.2 s)).` and `Last request with media for qwen3-4b: media: 1 image as text (1 from the
  surrogate cache or the conversation).`

An earlier conversation of that run was rendered by Qwen3-4B's template in 460 tokens (it drops
the earlier reasoning that makes it 13.6K on Qwen3.8), so its switch was skipped as `short`
after the surrogate was made (10.8 s, the source again unchanged), and the target prefilled the
rewritten request natively and answered correctly. Evidence:
`evidence/universal-kv/live-media-surrogates-hermes-20261004-runs/` (`1-reasoning-short`,
`2-concat`), `experiments/universal_kv/live_bridge.py surrogates-hermes`. These runs call the
adapter's methods, not a running Hermes.

### Images the target reads

A conversation with images (`image_url` parts), audio (`input_audio`) or video
(`input_video`) can be switched to a target that has that modality (`/props` modalities; a
server started with a matching `--mmproj`):

- The bridge renders each media part as the target's media marker exactly as llama-server
  renders the real request (text parts of a media message joined by newlines, none next to a
  marker), so the target's tokens up to the media match the real request's.
- The translated cache stops at the start of the first message with media (`max_t0_message`
  in the engine job); the target prefills that message, the media in it and everything after
  natively. With media only in the newest message nothing is lost; media in an early message
  shrinks the translated part to the text before it (and the switch may then be `short`).
- When the source's own server has encoded the media (a vision source that answered after the
  image), its cache holds the media's cells. The engine reads that cache up to its first medium
  (`source_media` in the engine report): the text before it is aligned and translated as usual,
  and the media cells and everything after them are left out (they lie after T0 anyway). T0
  also stops after the last message found in that text (`source_media.t0_cap`), which matters
  when the request no longer holds the source's media (a client that replaced an earlier image
  by text): the rest is prefilled natively instead of translated from nothing. An engine older
  than this skips such switches (`media-source`). Verified on tiny fixtures and on real
  llama-server slot saves holding an image (a tiny dense model with NORMAL image positions, and
  a tiny Qwen3.5 hybrid with Qwen3.8's projector and M-RoPE positions): the media slot gives
  the same translated cache, byte for byte, as a text-only save of its prefix. Live, Qwen3.8's
  slot with the chart's cells was read up to them in the Qwen3.8 → Muse switch above.
- The media's bytes never reach the bridge's files; the target reads them from the real
  request.
- No warm request is sent (it would read the marker as text); Hermes never sends one anyway.
- The native tail is lengthened by the marker's token count per media part, because the
  bridge counts the marker as text where the server puts the media's own cells.

Measured live from Muse (no vision) to the production Qwen3.8 (with its mmproj) on a 13.5K-token
conversation with a generated chart: with the image in the newest message the real chat
request reused all 11,403 translated tokens and prefilled the rest with the image in 9.4 s
(47 s natively), with the same answer as natively; with the image in an earlier message the
translated part stopped there (7,311 tokens), the request reused exactly those, and the answer
was right, although the model's reasoning first doubted that an image was attached. Over three
such conversations the translated prefix changed the outcome each time (that doubt, a
reasoning that did not reach its answer within 1,200 tokens, a formatting request no longer
followed), while the target's own cache restored the same way answered like the native
prefill: the restore places the image correctly, and what follows a translated prefix depends
on its quality.

## Request template settings

The real request's own template settings change what the target processes: llama-server passes
`chat_template_kwargs` and `reasoning_effort` to the chat template. Qwen3.8 writes the reasoning
effort into its system prompt (`xhigh`, the default, adds an instruction paragraph; `medium` adds
none; `low` another), `enable_thinking: false` drops it and closes the think block, and Muse
reads the effort as its "Reasoning strength". Such a request renders differently after its first
3 tokens, so a cache translated from the default rendering would be reused for 3 tokens.

The bridge renders the conversation with the request's settings (`/apply-template` with the
same fields) and, when that differs from the default rendering, hands the settings to the
engine, which renders the target record the same way: it builds the template context
llama-server builds for the request (`chat.server_context` in `experiments/linear_transfer`:
the kwargs as template variables, `reasoning_effort` as the `reasoning_effort` kwarg and
`reasoning_strength`, `"none"` as thinking off, a boolean `preserve_reasoning` as
`preserve_thinking` and its negations) and checks that the target's tokens spell its rendering
byte for byte. The server's own settings (`--reasoning-effort`, `--chat-template-kwargs`,
`--reasoning off`, the date a template prints) are not visible over HTTP, so the engine tries
the template's defaults, then llama-server's defaults (with the template flags the model was
registered with, `kvpark bridge add --reasoning-effort ...`), then those with thinking off,
and keeps the first rendering that matches (`rendering` in the engine report). Where none matches, it
says so (`template` error kind; `engine.py` exits with status 4) and the bridge translates the
default rendering instead, up to the tokens it shares with the request's (`template_options` in
the report: the options, `shared_tokens`, `engine` true when the engine rendered with them).
An engine older than the `template-options` feature gets the default rendering in the first
place. Hermes passes the request's settings; `kvpark bridge switch` reads them from the
conversation file.

What Hermes sends (read from `~/.hermes/config.yaml` and its custom-provider code): a top-level
`reasoning_effort: "medium"` for the main model (`agent.reasoning_effort`; `"none"` when
reasoning is turned off), plus `chat_template_kwargs: {"preserve_reasoning": true}` from a
provider entry's `extra_body`, and `chat_template_kwargs: {"enable_thinking": false}` for its
auxiliary compression calls.

Verified:

- Against llama-server's own `/apply-template` (the pinned CPU build, tiny models carrying the
  real Qwen3.8, Muse and Qwen3-4B templates; Qwen3.8 also with the production flags
  `--reasoning-effort xhigh --reasoning-preserve`): a chat and a tool-calling conversation, each
  with no options, `reasoning_effort` medium, low, high and none, `enable_thinking: false`,
  Hermes' pair, and `preserve_reasoning: false`. Every rendering matched byte for byte, with the
  template's defaults except Muse without a system message (its date line: llama-server's
  context). The same on the production Qwen3.8 build (b10815-85d5703a3), and for servers run
  with template flags the engine is told (Qwen3.8 with `--reasoning-effort low`,
  `--no-reasoning-preserve`, `--reasoning off` or `--chat-template-kwargs
  '{"enable_thinking": false}'`; Muse with `--reasoning-effort xhigh`), which match nothing
  without them. `experiments/universal_kv/test_engine.py`, `LlamaServerTemplateTest`.
- Live, Muse-Glimmer-30B → the production Qwen3.8 build (DFlash2 draft, mmproj, the
  production flags; context 65,536), two held-out chats of 12.1K and 13.6K tokens, the real
  chat request sent with the same settings after the switch: with no options, Hermes'
  `reasoning_effort: "medium"`, Hermes' pair (`medium` and `preserve_reasoning`) and
  `enable_thinking: false` (which renders Qwen3.8's prompt byte for byte as Hermes'
  `reasoning_effort: "none"` does), all 8 switches restored T0 and the request reused exactly T0 tokens
  (cache_n 11,579 / 11,541 / 11,541 / 11,543 and 10,028 / 9,990 / 9,990 / 9,992). Before this,
  the three with options were skipped (the renderings share 3 tokens). The switch took 1.4-4.1 s
  (engine 1.1-1.8 s), then the request prefilled its 2,048-token tail in 17.0-21.0 s, against
  80-102 s of native prefill on a machine shared with other GPU work. The Qwen3.8 log has no
  draft-position warning, position error or failed decode.

## DeepSeek V4 targets

`deepseek4` (DeepSeek V4, V4-Pro, V4-Flash) keeps per-token latents only for its last 128 tokens
and pools everything older into compressed rows (CSA with its lightning indexer, HCA). The
engine synthesizes those rows from per-token compressor inputs that the hub decodes with head
decoders fitted for the model (`experiments/universal_kv/fit_hub.py --heads NAME`):

- Target only: a DeepSeek V4 source is skipped (`dsv4-source`), since its cache cannot be read
  back token by token.
- The hub needs all three head kinds for the target (`csa`, `hca`, `lid`), else
  `no-head-decoders`.
- T0 is rounded down to whole 128-token blocks; `restored_tokens` and the reuse follow the
  engine's T0.
- The server must run without a draft model (`draft-unsupported`).
- Conversations with images or audio are prefilled natively (`dsv4-media`).
- No native prefix: `kvpark bridge add --native-prefix` is refused for it, and an older entry's
  flag is ignored (`native_prefix_tokens` 0).
- The translation runs on the CPU whatever `--device` says. Pooling took about 21 s for 32K
  tokens at V4-Flash's geometry, so a long switch takes tens of seconds.
- Verified on a tiny random-weight model only: a switch restores and reuses the translated cells,
  and a state synthesized from the model's own per-token inputs restores to the native logits
  within 1e-9. No real DeepSeek V4 model has been run (V4-Flash at IQ2 is about 91 GB), so the
  translation quality is unknown. Images with DeepSeek V4 are untested, and V4.1 is not in the
  pinned llama.cpp.

## In Hermes

Register the models and the engine as above (the model kvpark manages is the usual source).
Then, inside the conversation you want to move:

```
/kvpark switch muse
```

(`/kvpark switch muse from qwen38` names the model you are leaving.) Staging starts the engine
daemon in the background and loads the pair, so the switch itself runs warm. Then switch
Hermes to that model with `/model` and send your next message. kvpark's request middleware
sees the conversation's request go to the target's server, translates the cache, restores it
there and lets the request through; the target prefills only the native tail. The stage lasts
15 minutes and is used once. `/kvpark switch status` shows the last result (with how the
engine ran: daemon or one-shot, GPU or CPU, and its seconds, and whether the request's
template settings were rendered by the engine, and what became of the conversation's media),
including a skipped switch and its reason (for example a rendering the engine cannot
reproduce). Any failure (an engine error) falls back to a normal full prefill. The request
itself changes in one way only: media the target's server cannot read are replaced by their
text surrogates, in this and every later request to that model (see
[The real request](#the-real-request)).

## Add a model

A model joins the hub with its own K/V captured on the hub corpus (public SmolTalk2 chats):

1. Render the corpus with the model's template and capture K/V with `kvpark-kv-probe`
   (`experiments/linear_transfer/chat.py`, `collect_v2.py`; ~15 minutes on the GPU for a
   9B model).
2. Refit the hub with the new model added (`experiments/universal_kv/fit_hub.py`, about 2 hours
   on the CPU for four models), or fit only its adapters against an existing hub (`--base`,
   about 15 minutes). A DSA model also needs an indexer ridge (`--indexer-ridge NAME`) for
   prompts longer than its indexer's `top_k`, a DeepSeek V4 target its head decoders
   (`--heads NAME`). Convert the bundle once with `experiments/universal_kv/convert_hub.py`.
3. For a hybrid model, build its recurrent mean and register it with `--recurrent-mean` (see
   [Hybrid targets](#hybrid-targets-the-recurrent-state)).

The probe, state codec and template checks must support the model's architecture first, on the
llama.cpp version its server runs; see
[experiments/linear_transfer/README.md](../experiments/linear_transfer/README.md) for the state
layouts, the template parity check (`template_check.py`) and the vocabulary check, and [experiments/universal_kv/ARCHITECTURE.md](../experiments/universal_kv/ARCHITECTURE.md#what-the-translator-supports-today)
for what each architecture has been validated on.

## Limitations

- **Quality.** Chat translates well: on held-out chats the measured pairs stay within 0.05
  nats/token of a native prefill. Agent conversations with large tool outputs and long
  documents do not: agent traces lost about 0.3 nats/token into Muse, and after a switch into
  Qwen3.8 the model's next-token distribution stayed far from native (KL 0.4-0.5). For such
  conversations a native prefill (`kvpark bridge undo MODEL`, or simply not switching) is the
  safer choice.
- **Sources.** A small model's cache helps a larger model little: of the gap between an
  information-free cache and a native prefill, Qwen3-4B → Qwen3.8 closed 15% and Qwen3-4B → Muse
  nothing (Qwen3.8 → Muse: 56%). Prefer the larger model as the source.
- **Validated models.** Sixteen real models of thirteen llama.cpp architectures are measured end to end
  (four fitted jointly, twelve onboarded, fifteen of them in hub6; see Models above); the onboarded ones only
  against Qwen3.8 and themselves, apart from a few live hops, and on chats the methods were chosen on.
  Out of Granite-4.0-H-Tiny a switch keeps little more than an information-free cache, and out of
  Nemotron-3.5 less (register it with `--no-source`). Every other architecture is verified on random-weight
  models only.
- **The native tail** dominates a switch (8-16 s on the 27-30B models). A shorter tail needs
  better translation; on hybrids, the recurrent mean allows one for chats.
- **Images.** The translated part stops before the first image the target reads itself or the
  source's cache holds, so an image early in the conversation leaves little to translate, and
  in 3 of 3 measured conversations into Qwen3.8 the translated text in front of an image
  changed the answer. A target without vision reads a description of the image, which carries
  only what the description says.
- **Drafts.** MTP drafting accepts less after a switch (48% against 61%) for the rest of the
  conversation. Draft models other than DFlash/DSpark, MTP and Gemma 4's MTP head, and any draft
  on a DeepSeek V4 target, are skipped.
- **Server settings.** A target server run with chat template flags must be registered with the
  same flags, or switches into it are skipped (`template-mismatch`).
- **Platforms.** The engine daemon runs on Linux and macOS (macOS untested); Windows runs the
  slower one-shot engine. The GPU path is tested on ROCm (Radeon 8060S) only; CUDA and Apple MPS
  are untested.
- **The target's conversation** is replaced by the switch; park it first if you need it.
- **Hermes.** `/kvpark switch` is covered by integration tests with a stand-in engine, and its
  switch and media path (`Adapter.bridge_switch`, `bridge_media`) ran live on real servers
  outside Hermes; a live Hermes session has not been run yet.
