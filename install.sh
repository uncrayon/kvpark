#!/usr/bin/env bash
# Install kvpark on macOS/Linux. Keep execution at the end for curl | bash.
set -euo pipefail

main() {
    local uv_bin helper work script_dir
    case "$(uname -s)" in
        Linux|Darwin) ;;
        *) printf '%s\n' 'Use this installer in Linux, macOS, or WSL. See docs/install.md for native Windows.' >&2; exit 1 ;;
    esac
    if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
        cat <<'HELP'
Install kvpark: bash install.sh [options]
  --hermes                 Install into detected Hermes Python and enable the plugin
  --hermes-python PATH     Select a custom Hermes Python environment
  --standalone             Install a private standalone environment
  --runtime cpu|metal|cuda|hip|vulkan  Also build the managed inference runtime
  --no-setup               Install now and choose a model later
  --no-path                Do not update shell startup files
  --install-dir PATH       Override the private installation directory
  --bin-dir PATH           Override the command directory (default: ~/.local/bin)
  --version VERSION        Select a published kvpark release
  --package PATH           Install a local wheel or source checkout (development)
Default: detect Hermes; otherwise install standalone. Then choose a model.
No sudo or activation needed.
HELP
        return
    fi
    work=$(mktemp -d "${TMPDIR:-/tmp}/kvpark-install.XXXXXXXX")
    trap "$(printf 'rm -rf -- %q' "$work")" EXIT
    uv_bin=${KVPARK_UV:-}
    if [[ -z "$uv_bin" ]]; then uv_bin=$(command -v uv || true); fi
    if [[ -z "$uv_bin" ]]; then
        printf '\nInstalling the Python setup tool...\n'
        curl --fail --show-error --silent --location --retry 3 --connect-timeout 15 \
            --proto '=https' --tlsv1.2 https://astral.sh/uv/install.sh -o "$work/uv-install.sh"
        UV_UNMANAGED_INSTALL="$work/uv" sh "$work/uv-install.sh"
        uv_bin="$work/uv/uv"
    fi
    helper=''
    if [[ -f "${BASH_SOURCE[0]}" ]]; then
        script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
        if [[ -f "$script_dir/scripts/install.py" ]]; then helper="$script_dir/scripts/install.py"; fi
    fi
    if [[ -z "$helper" ]]; then
        helper="$work/install.py"
        curl --fail --show-error --silent --location --retry 3 --connect-timeout 15 \
            --proto '=https' --tlsv1.2 \
            https://raw.githubusercontent.com/uncrayon/kvpark/main/scripts/install.py -o "$helper"
    fi
    # uv supplies Python when the machine has no compatible interpreter.
    "$uv_bin" run --no-project --no-config --python 3.12 -- python "$helper" --uv "$uv_bin" "$@"
    rm -rf -- "$work"
    trap - EXIT
}

main "$@"
