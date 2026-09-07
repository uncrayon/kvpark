# One-command installation

On macOS, Linux, or WSL:

```bash
curl -fsSL https://raw.githubusercontent.com/uncrayon/kvpark/main/install.sh | bash
```

The installer uses [uv](https://docs.astral.sh/uv/) to supply Python 3.12 when
needed. It downloads the published kvpark wheel and verifies GitHub's SHA-256
checksum before installing. You do not need Git, pip, or a Python virtual
environment for the package installation.

If Hermes is present, the installer selects its Python environment, verifies the
plugin entry point, and enables kvpark without granting tool overrides. It honors
`HERMES_HOME` and `HERMES_INSTALL_DIR`. After installation, a model chooser opens
in the same terminal, even with `curl | bash`. Choose a number, then confirm with
`yes`. Choose `0` to finish later. Restart Hermes and start a new conversation
after connecting. You can reopen the chooser with `/kvpark setup` in Hermes.

Without Hermes, it creates a private environment. The installer prints an
absolute command that works immediately; `kvpark` is available by name in a new
terminal. Run `kvpark setup` to choose a discovered model later. After setup,
`kvpark doctor` checks the selected connection; `kvpark start` restarts it after
a reboot. See [backend setup](backends.md) for agent integration details.

Scanning or declining setup does not change the current model route. Confirming
a choice starts kvpark and saves the working connection. The installer does not
download model weights or launch a model. Disk park/resume requires the managed patched
llama.cpp runtime; existing servers use routing mode.

## Options

Append options using `bash -s --`:

```bash
curl -fsSL https://raw.githubusercontent.com/uncrayon/kvpark/main/install.sh | bash -s -- --hermes
curl -fsSL https://raw.githubusercontent.com/uncrayon/kvpark/main/install.sh | bash -s -- --standalone
curl -fsSL https://raw.githubusercontent.com/uncrayon/kvpark/main/install.sh | bash -s -- --hermes-python "/path/to/hermes/venv/bin/python"
```

| Option | Effect |
| --- | --- |
| `--hermes` | Require Hermes; report a missing installation instead of choosing standalone |
| `--hermes-python PATH` | Select an exact Hermes interpreter for a custom install |
| `--standalone` | Create a private environment even if Hermes is installed |
| `--runtime cpu` | Also build the patched llama.cpp runtime |
| `--runtime metal/cuda/hip/vulkan` | Select one GPU build, using an already installed GPU toolchain |
| `--no-setup` | Install now and open the model chooser later |
| `--no-path` | Leave shell startup files unchanged; use the printed absolute command |
| `--install-dir PATH` | Choose the private environment and optional runtime location |
| `--bin-dir PATH` | Choose where to put the `kvpark` command |
| `--version 0.3.0a2` | Select an exact published release (currently the default) |
| `--package PATH` | Install a local wheel or checkout for development |

Run `bash install.sh --help` from a checkout to see the same options. From a local
checkout, `bash install.sh --standalone --package .` installs your working source.
The network command installs the published release. The wizard is available in
`0.3.0a2` and later.

To inspect the bootstrap before running it:

```bash
curl -fsSL https://raw.githubusercontent.com/uncrayon/kvpark/main/install.sh -o install.sh
bash install.sh --help
bash install.sh
```

The bootstrap downloads its Python helper from the same repository and uses
[uv's official installer](https://docs.astral.sh/uv/reference/installer/) if uv is
missing. Internet access to GitHub, Astral, and the Python package index is needed.

## Build the inference runtime too

```bash
curl -fsSL https://raw.githubusercontent.com/uncrayon/kvpark/main/install.sh | bash -s -- --runtime cpu
```

This requires Git, CMake, and a C++ compiler. Install those with your operating
system's package manager; on macOS install the Xcode Command Line Tools. GPU
builds also need their vendor toolchain. The installer exports the pinned llama.cpp
source, applies the persistence patch, builds it, and checks the executable.
It keeps the executable and its adjacent libraries together and prints the exact
server path and setup command. You still provide your own GGUF model.

The runtime is reused on repeat installs. See [runtime compatibility](compatibility.md)
for supported builds and snapshot compatibility.

## Locations and removal

Standalone environments live under `~/.local/share/kvpark/install` on Linux/WSL
(or `$XDG_DATA_HOME/kvpark/install`) and `~/Library/Application Support/kvpark/install`
on macOS. Hermes installations use Hermes' existing virtual environment. Optional
runtimes live in the installer directory's `runtimes` subdirectory.

The command is a wrapper at `~/.local/bin/kvpark` by default. Bash, Zsh, and Fish
startup configuration receives a PATH entry once; `--no-path` disables this.
uv-managed Python downloads have [uv's own storage locations](https://docs.astral.sh/uv/reference/storage/).
No system packages are replaced and no sudo command is executed.

Before removing anything, [disconnect services and restore your agent's backend](uninstall.md).
For a standalone installer environment, remove its `venv` directory after stopping
its services. For Hermes, uninstall the package from Hermes' own interpreter as
described in that guide. Remove the installer-created command and its PATH entry
if no longer needed. Standalone setup writes a connection record under the data
directory and keeps each selected connection in its own `connections` subdirectory.
`kvpark uninstall --confirm` clears the selected connection after stopping its
services. Optional runtime builds and archive snapshots are separate;
removing Python does not remove model weights or saved archives.

## Troubleshooting

- **Hermes not found:** pass `--hermes-python` with the Python inside the
  environment that runs Hermes. Activating kvpark's standalone environment does
  not make Hermes discover its plugin. Multiple Hermes environments require an
  explicit selection.
- **Download failed:** check your connection or proxy, then rerun the same
  command. The installer reports the failed URL; it does not bypass TLS checks.
- **Command not found:** use the absolute command printed at completion, or open
  a new terminal. The installer cannot change the parent shell's PATH.
- **Command belongs to another installation:** use `--bin-dir` to choose a
  different directory. The installer preserves an existing unmanaged command.
- **Moved or incomplete environment:** use a fresh `--install-dir`; Python
  environments are not portable between checkout locations or computers.
- **sloth-memory found:** follow the [rename migration](updates.md#migrate-from-sloth-memory)
  to preserve caches and routes. The installer does not enable a conflicting plugin.
- **A different kvpark version is already installed:** use `/kvpark update` in
  Hermes or the installed `kvpark update` command for coordinated updates.
- **Missing runtime compiler:** package installation needs no compiler. For
  `--runtime`, install the listed build prerequisites, then rerun the installer.

## Native Windows

The Bash installer supports Windows through WSL. For native Windows, use the
[manual standalone instructions](../README.md#manual-installation) or
[Hermes' Windows installation instructions](hermes.md#install-from-a-checkout-or-on-windows).
