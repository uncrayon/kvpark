# Security and deployment scope

This alpha is for a trusted user on one Linux, macOS, or Windows machine. It binds to localhost,
rejects browser Origin requests, and optionally requires `KVPARK_API_KEY` for all
proxy endpoints. It is not a multi-user inference gateway. Do not expose its ports
through a tunnel or public reverse proxy as a shared service.

Archives contain model state and prompt tokens. They are sensitive data, not
anonymized caches. New directories/files use private permissions; existing
directories retain their existing permissions. Use a private local directory and
normal disk protection. `forget` removes files but does not promise secure erasure
from SSDs or backups.

Only restore archives created by your own compatible runtime. The native format
is not designed for untrusted file imports. Runtime identity checks detect local
replacement; they do not authenticate an attacker-controlled manifest or model.

Only one proxy and backend may own an archive directory. Advisory locks cannot
protect against unrelated processes that bypass the proxy or ignore its locks.
Backend administration remains a trusted local responsibility.

Report a vulnerability through GitHub's private vulnerability reporting if enabled.
If unavailable, open an issue requesting a private contact without exploit details.
