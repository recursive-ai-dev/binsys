#!/usr/bin/env bash
# binsys script helpers — colors, prompts, status, and error handling
# Source this from other scripts:  source "$(dirname "$0")/lib.sh"

set -euo pipefail

# ── colors ────────────────────────────────────────────────────────────────────
RST="\033[0m"; BLD="\033[1m"; DIM="\033[2m"
RED="\033[91m"; GRN="\033[92m"; YLW="\033[93m"; BLU="\033[94m"; CYN="\033[96m"

info()  { echo -e " ${BLU}→${RST} $*"; }
ok()    { echo -e " ${GRN}✓${RST} $*"; }
warn()  { echo -e " ${YLW}⚠${RST} $*"; }
err()   { echo -e " ${RED}✗${RST} $*" >&2; }
die()   { err "$@"; exit 1; }
header(){ echo -e "\n${BLD}${CYN}━━━ $* ━━━${RST}\n"; }
sub()   { echo -e " ${DIM}·${RST} $*"; }
done_msg()  { echo -e " ${GRN}✔${RST} Done.\n"; }

# ── prompts ───────────────────────────────────────────────────────────────────

prompt() {
    # Usage: val=$(prompt "Question" "default")
    local msg="$1" default="$2"
    local p
    if [[ -n "$default" ]]; then
        read -r -p "$(echo -e " ${BLU}?${RST} ${msg} [${default}]: ")" p
    else
        read -r -p "$(echo -e " ${BLU}?${RST} ${msg}: ")" p
    fi
    echo "${p:-$default}"
}

confirm() {
    # Usage: confirm "Question" "y/N"  → returns true/false
    local msg="$1" default="${2:-N}"
    local y n
    if [[ "$default" =~ ^[Yy] ]]; then y="Y"; n="n"; else y="y"; n="N"; fi
    local p
    read -r -p "$(echo -e " ${BLU}?${RST} ${msg} (${y}/${n}): ")" p
    p="${p:-$default}"
    [[ "$p" =~ ^[Yy] ]]
}

select_one() {
    # Usage: val=$(select_one "Choose" "opt1" "opt2" "opt3")
    # Displays a numbered menu, returns the selected value.
    local msg="$1"; shift
    local opts=("$@")
    echo -e " ${BLU}?${RST} ${msg}:"
    for i in "${!opts[@]}"; do
        echo "    $((i+1))) ${opts[$i]}"
    done
    local sel
    read -r -p "$(echo -e "   ${DIM}enter number [1-${#opts[@]}]:${RST} ")" sel
    sel="${sel:-1}"
    echo "${opts[$((sel-1))]}"
}

# ── status bar ────────────────────────────────────────────────────────────────

run_with_spinner() {
    local label="$1"; shift
    echo -ne " ${CYN}⠋${RST} ${label}..."
    if "$@" 2>/dev/null; then
        echo -e "\r ${GRN}✓${RST} ${label}  "
    else
        echo -e "\r ${RED}✗${RST} ${label}  "
        return 1
    fi
}

# ── binsys wrapper ─────────────────────────────────────────────────────────────

BINSYS="${BINSYS:-$(dirname "$0")/../binsys.py}"

binsys() {
    python3 "$BINSYS" "$@"
}

list_systems() {
    binsys list --json 2>/dev/null | python3 -c "
import json, sys
systems = json.load(sys.stdin)
for m in systems:
    badges = ''
    if m.get('mounted'): badges += 'M'
    if m.get('encrypted'): badges += 'E'
    if m.get('frugal'): badges += 'F'
    print(f\"  {m['name']:<22} [{m.get('type','?'):>8}]  {m.get('size','?'):<8}  {badges}\")
" 2>/dev/null || echo "  (no systems)"
}

require_binsys() {
    if [[ ! -f "$BINSYS" ]]; then
        die "binsys.py not found at $BINSYS"
    fi
}
