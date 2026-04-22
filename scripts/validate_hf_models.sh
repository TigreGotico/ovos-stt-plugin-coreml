#!/usr/bin/env bash
# Validate all published CoreML models on HuggingFace against the current plugin code.
#
# For each repo: download → spot-check with CoremlSTT → delete → report.
# Models are processed in parallel (configurable).
#
# Usage:
#   ./validate_hf_models.sh
#   ./validate_hf_models.sh --hf-org OpenVoiceOS --max-parallel 4
#   ./validate_hf_models.sh --pattern "parakeet-tdt-*" --python /path/to/python
#   ./validate_hf_models.sh --validate-dir /Volumes/hdd/tmp/validate
set -uo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
HF_ORG="OpenVoiceOS"
PYTHON="/Users/tigregotico/PycharmProjects/mobius/models/stt/parakeet-tdt-ctc-110m/coreml/.venv/bin/python3.10"
PLUGIN_DIR="/Users/tigregotico/PycharmProjects/ovos-stt-plugin-coreml"
TRACE_AUDIO="${PLUGIN_DIR}/scripts/yc_first_minute_16k_15s.wav"
VALIDATE_DIR="/tmp/ovos-coreml-validate"
MAX_PARALLEL=3
PATTERN="*"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --hf-org)       HF_ORG="$2";       shift 2 ;;
        --python)       PYTHON="$2";       shift 2 ;;
        --audio)        TRACE_AUDIO="$2";  shift 2 ;;
        --validate-dir) VALIDATE_DIR="$2"; shift 2 ;;
        --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
        --pattern)      PATTERN="$2";      shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "${VALIDATE_DIR}"

# ── All repos to validate ─────────────────────────────────────────────────────
REPOS=()
for slug in \
    parakeet-ctc-0.6b parakeet-ctc-1.1b parakeet-ctc-0.6b-vi \
    parakeet-tdt-ctc-110m parakeet-tdt-ctc-0.6b-ja \
    parakeet-tdt-0.6b-v2 parakeet-tdt-0.6b-v3 parakeet-tdt-1.1b \
    parakeet-rnnt-0.6b parakeet-rnnt-1.1b parakeet-rnnt-110m-da \
    parakeet-rnnt-120m-eou parakeet-rnnt-0.6b-multitalker \
    parakeet-tdt-0.6b-dutch parakeet-tdt-0.6b-estonian \
    parakeet-tdt-0.6b-polish parakeet-tdt-0.6b-portuguese \
    parakeet-tdt-0.6b-slovenian; do

    for suffix in "" "-fp16" "-int8" "-4bit" "-6bit"; do
        repo_name="${slug}-coreml${suffix}"
        # Apply pattern filter
        # shellcheck disable=SC2254
        case "$repo_name" in
            $PATTERN) REPOS+=("$repo_name") ;;
        esac
    done
done

# ── Helpers ───────────────────────────────────────────────────────────────────

_hf_repo_exists() {
    curl -sf -o /dev/null "https://huggingface.co/${HF_ORG}/$1/resolve/main/metadata.json" 2>/dev/null
}

_validate_one() {
    local repo_name="$1"
    local repo="${HF_ORG}/${repo_name}"
    local dir="${VALIDATE_DIR}/${repo_name}"
    local result_file="${VALIDATE_DIR}/${repo_name}.result"

    # Check exists on HF before downloading
    if ! _hf_repo_exists "${repo_name}"; then
        echo "SKIP  ${repo_name} (not on HF)" | tee "${result_file}"
        return 0
    fi

    # Download
    rm -rf "${dir}"
    mkdir -p "${dir}"
    if ! huggingface-cli download "${repo}" \
            --local-dir "${dir}" \
            --quiet \
            2>/dev/null; then
        echo "FAIL  ${repo_name}: download failed" | tee "${result_file}"
        rm -rf "${dir}"
        return 1
    fi

    # Spot check via plugin (cpu_only for speed)
    # Write Python script to a temp file to avoid heredoc issues in exported functions
    local py_script
    py_script=$(mktemp /tmp/coreml_validate_XXXXXX)
    cat > "${py_script}" << 'PYEOF'
import sys, logging, os
# Silence OVOS/coremltools/NeMo logging before any imports
logging.disable(logging.CRITICAL)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from ovos_stt_plugin_coreml import CoremlSTT
from ovos_plugin_manager.utils.audio import AudioFile

meta_path, audio_path = sys.argv[1], sys.argv[2]
stt = CoremlSTT(config={"metadata": meta_path, "compute_units": "cpu_only"})
with AudioFile(audio_path) as f:
    audio = f.read()
text = stt.execute(audio)
if text is None or text == "":
    sys.exit(1)
print(text[:120])
PYEOF

    local check_out
    check_out=$("${PYTHON}" "${py_script}" "${dir}/metadata.json" "${TRACE_AUDIO}" 2>/dev/null)
    local rc=$?
    rm -f "${py_script}"

    # Delete downloaded model
    rm -rf "${dir}"

    if [ $rc -eq 0 ]; then
        printf "OK    %-50s %s\n" "${repo_name}" "${check_out}" | tee "${result_file}"
    else
        printf "FAIL  %-50s (empty or error)\n" "${repo_name}" | tee "${result_file}"
        return 1
    fi
}

export -f _validate_one _hf_repo_exists
export HF_ORG PYTHON TRACE_AUDIO VALIDATE_DIR

# ── Run in parallel ───────────────────────────────────────────────────────────

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${VALIDATE_DIR}/summary_${TIMESTAMP}.txt"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  CoreML Plugin Validation — $(date '+%Y-%m-%d %H:%M')           "
echo "║  Org: ${HF_ORG}  |  Parallel: ${MAX_PARALLEL}  |  Models: ${#REPOS[@]}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Semaphore: at most MAX_PARALLEL jobs at once
active=0
pids=()

for repo_name in "${REPOS[@]}"; do
    _validate_one "${repo_name}" &
    pids+=($!)
    active=$((active + 1))

    if [ "$active" -ge "$MAX_PARALLEL" ]; then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
        active=$((active - 1))
    fi
done

# Wait for remaining
for pid in "${pids[@]}"; do
    wait "$pid"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

passed=0; failed=0; skipped=0
for repo_name in "${REPOS[@]}"; do
    rf="${VALIDATE_DIR}/${repo_name}.result"
    [ ! -f "$rf" ] && continue
    line=$(cat "$rf")
    case "$line" in
        OK*)   echo "$line"; passed=$((passed + 1)) ;;
        FAIL*) echo "$line"; failed=$((failed + 1)) ;;
        SKIP*) skipped=$((skipped + 1)) ;;
    esac
done

echo ""
echo "Results: ${passed} passed, ${failed} FAILED, ${skipped} skipped (not on HF)"
echo "$(date '+%Y-%m-%d %H:%M:%S') — ${passed} passed, ${failed} failed, ${skipped} skipped" \
    > "${SUMMARY_FILE}"

# Clean up result files
rm -f "${VALIDATE_DIR}"/*.result

[ "$failed" -gt 0 ] && exit 1 || exit 0
