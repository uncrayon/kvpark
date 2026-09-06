# Attribution

kvpark originated as a local Qwen conversation slot archive integrated with
Hermes and was first released as sloth-memory. Its standalone proxy and CLI are MIT licensed.

The included patch modifies llama.cpp, Copyright (c) 2023–2026 The ggml authors,
under the MIT license. The upstream license is reproduced in
`patches/LLAMA_LICENSE`. The build downloads the pinned upstream source; model
weights remain separately supplied and licensed by their owners.

Related upstream work includes checkpoint persistence discussion and proposals:
https://github.com/ggml-org/llama.cpp/pull/20819

Qwen, Hermes, and llama.cpp are independent projects. This is a community project
and does not imply their endorsement.
