#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARTIFACT_DIR="$REPO_ROOT/artifact"
OUTPUT_ZIP="${UPI_TRM_ARCHIVE_OUTPUT:-$ARTIFACT_DIR/upi_trm_repository.zip}"
OUTPUT_SHA="$OUTPUT_ZIP.sha256"
OUTPUT_DIR="$(dirname "$OUTPUT_ZIP")"
mkdir -p "$OUTPUT_DIR"
FILE_LIST="$(mktemp)"
TEXT_FILE_LIST="$(mktemp)"
TEMP_ROOT="$(mktemp -d)"
STAGING_ROOT="$TEMP_ROOT/repository"
TEMP_ZIP="$TEMP_ROOT/upi_trm_repository.zip"
HOME_PATH_PATTERN='/(home)/'
MAC_HOME_PATTERN='/User(s)/'
DATA_USER_PATH_PATTERN='/data/user(s)/'
PRIOR_TREE_PATTERN='UPI_TRM_N[A-Z]+'
PRIOR_VENUE_PATTERN='N(eur)IPS'
LOWER_PRIOR_VENUE_PATTERN='n(eur)ips'
SHORT_VENUE_PATTERN='NIP(S)'
LOWER_SHORT_VENUE_PATTERN='nip(s)'
LEGACY_USER_PATTERN='buik(sat)'

cleanup() {
    rm -f -- "$FILE_LIST" "$TEXT_FILE_LIST"
    rm -rf -- "$TEMP_ROOT"
}
trap cleanup EXIT

if [[ -n "${UPI_TRM_ARCHIVE_FILE_LIST:-}" ]]; then
    cp -- "$UPI_TRM_ARCHIVE_FILE_LIST" "$FILE_LIST"
else
    git ls-files --cached --others --exclude-standard | sort -u | while IFS= read -r path; do
        if [[ -e "$path" || -L "$path" ]]; then
            printf '%s\n' "$path"
        fi
    done > "$FILE_LIST"
fi

mkdir -p -- "$STAGING_ROOT"
while IFS= read -r relative_path; do
    [[ -n "$relative_path" ]] || continue
    [[ "$relative_path" != /* && "$relative_path" != *".."* ]] || {
        printf 'Unsafe archive path: %s\n' "$relative_path" >&2
        exit 1
    }
    source_path="$REPO_ROOT/$relative_path"
    [[ -e "$source_path" || -L "$source_path" ]] || {
        printf 'Missing archive input: %s\n' "$relative_path" >&2
        exit 1
    }
    mkdir -p -- "$STAGING_ROOT/$(dirname "$relative_path")"
    cp -a -- "$source_path" "$STAGING_ROOT/$relative_path"
done < "$FILE_LIST"

FORBIDDEN_PATTERN="$HOME_PATH_PATTERN|$MAC_HOME_PATTERN|$DATA_USER_PATH_PATTERN|$PRIOR_TREE_PATTERN|$PRIOR_VENUE_PATTERN|$LOWER_PRIOR_VENUE_PATTERN|$SHORT_VENUE_PATTERN|$LOWER_SHORT_VENUE_PATTERN|$LEGACY_USER_PATTERN"
PATH_FORBIDDEN_PATTERN="$PRIOR_TREE_PATTERN|$PRIOR_VENUE_PATTERN|$LOWER_PRIOR_VENUE_PATTERN|$SHORT_VENUE_PATTERN|$LOWER_SHORT_VENUE_PATTERN|$LEGACY_USER_PATTERN"

# Keep raw historical provenance in the repository while removing local user,
# machine, and prior-venue identifiers from the double-blind archive copy.
rg -l -0 --hidden --no-ignore "$FORBIDDEN_PATTERN" "$STAGING_ROOT" > "$TEXT_FILE_LIST" || true
if [[ -s "$TEXT_FILE_LIST" ]]; then
    xargs -0 perl -pi -e 's{/(?:home)/[A-Za-z0-9._-]+/trm_bellman}{<CODE_REPO_ROOT>}g; s{/(?:home)/[A-Za-z0-9._-]+/UPI_TRM/UPI_TRM_N[A-Z]+}{<HISTORICAL_PAPER_ROOT>}g; s{/(?:home)/[A-Za-z0-9._-]+/UPI_TRM/UPI_TRM_ICLR}{<PAPER_SOURCE_ROOT>}g; s{/(?:home)/[A-Za-z0-9._-]+/fbsource(?:/fbcode)?}{<BUILD_ROOT>}g; s{/data/(?:users)/[A-Za-z0-9._-]+/fbsource(?:/fbcode)?}{<BUILD_ROOT>}g; s{/(?:home)/[A-Za-z0-9._-]+}{<ANON_HOME>}g; s{/(?:Users)/[A-Za-z0-9._-]+}{<ANON_HOME>}g; s{/data/(?:users)/[A-Za-z0-9._-]+}{<ANON_DATA_ROOT>}g; s{UPI_TRM_N[A-Z]+}{UPI_TRM_LEGACY}g; s{N(?:eur)IPS}{prior venue}g; s{n(?:eur)ips}{prior_venue}g; s{NIP(?:S)}{LEGACY}g; s{nip(?:s)}{legacy}g; s{buik(?:sat)}{anon}ig' < "$TEXT_FILE_LIST"
fi

# Neutralize venue and user tokens in archived path names as well as contents.
while IFS= read -r -d '' staged_path; do
    [[ "$staged_path" != "$STAGING_ROOT" ]] || continue
    relative_path="${staged_path#"$STAGING_ROOT/"}"
    [[ "$relative_path" =~ $PATH_FORBIDDEN_PATTERN ]] || continue
    neutral_path="$(printf '%s' "$relative_path" | perl -pe 's{UPI_TRM_N[A-Z]+}{UPI_TRM_LEGACY}g; s{N(?:eur)IPS}{prior_venue}g; s{n(?:eur)ips}{prior_venue}g; s{NIP(?:S)}{LEGACY}g; s{nip(?:s)}{legacy}g; s{buik(?:sat)}{anon}ig')"
    [[ "$neutral_path" != "$relative_path" ]] || continue
    if [[ -d "$staged_path" && -d "$STAGING_ROOT/$neutral_path" ]]; then
        rmdir -- "$staged_path"
        continue
    fi
    [[ ! -e "$STAGING_ROOT/$neutral_path" && ! -L "$STAGING_ROOT/$neutral_path" ]] || {
        printf 'Archive path anonymization collision: %s\n' "$neutral_path" >&2
        exit 1
    }
    mkdir -p -- "$STAGING_ROOT/$(dirname "$neutral_path")"
    mv -- "$staged_path" "$STAGING_ROOT/$neutral_path"
done < <(find "$STAGING_ROOT" -depth -print0)

if find "$STAGING_ROOT" -print | grep -Eiq "$FORBIDDEN_PATTERN"; then
    printf 'Archive filename anonymization check failed.\n' >&2
    exit 1
fi

if rg -a -l --hidden --no-ignore "$FORBIDDEN_PATTERN" "$STAGING_ROOT"; then
    printf 'Archive content anonymization check failed.\n' >&2
    exit 1
fi

mkdir -p -- "$STAGING_ROOT/artifact"
(
    cd "$STAGING_ROOT"
    find . -type f ! -path './artifact/SHA256SUMS' -printf '%P\0' | LC_ALL=C sort -z | xargs -0 sha256sum > artifact/SHA256SUMS
    find . -type f -exec touch -t 198001010000.00 {} +
    find . \( -type f -o -type l \) -printf '%P\n' | LC_ALL=C sort | zip -X -q -y "$TEMP_ZIP" -@
)

mv -f -- "$TEMP_ZIP" "$OUTPUT_ZIP"
(
    cd "$OUTPUT_DIR"
    sha256sum "$(basename "$OUTPUT_ZIP")" > "$(basename "$OUTPUT_SHA")"
)

printf 'Wrote %s\n' "$OUTPUT_ZIP"
printf 'Wrote %s\n' "$OUTPUT_SHA"
