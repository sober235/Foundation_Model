#!/bin/bash
# [NEED_CARDS=k] wait_for_card.sh <candidates, e.g. 1,6,4> <minutes> <command...>
# Waits until k (default 1) of the candidate cards have each had no compute process and less than 1.5 GB in use for
# <minutes> consecutive minutes (another session's loop that relaunches within a minute or two keeps its card), then
# runs the command with CARD=<the first of them> and CARDS=<the k of them, comma-separated> in the environment, in the
# candidates' order of preference. Polls every 30 s; prints one line when it takes the cards.
set -u
CANDIDATES=$1; MINUTES=$2; shift 2
[ "$MINUTES" -ge 1 ] || { echo "minutes must be >= 1"; exit 1; }
NEED=$(( MINUTES * 2 ))
K=${NEED_CARDS:-1}
declare -A STREAK
for c in ${CANDIDATES//,/ }; do STREAK[$c]=0; done
busy_cards() {   # the indices of cards with a compute process
  local map
  map=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | while read -r u; do
    echo "$map" | awk -F', ' -v u="$u" '$2 == u {print $1}'
  done | sort -u
}
while true; do
  busy=" $(busy_cards | tr '\n' ' ') "
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
  for c in ${CANDIDATES//,/ }; do
    mem=$(echo "$used" | awk -F', ' -v c="$c" '$1 == c {print $2}')
    if [[ "$busy" != *" $c "* ]] && [ -n "$mem" ] && [ "$mem" -lt 1500 ]; then
      STREAK[$c]=$(( ${STREAK[$c]} + 1 ))
    else
      STREAK[$c]=0
    fi
  done
  ready=()
  for c in ${CANDIDATES//,/ }; do [ "${STREAK[$c]}" -ge "$NEED" ] && ready+=("$c"); done
  if [ "${#ready[@]}" -ge "$K" ]; then
    take=("${ready[@]:0:$K}")
    export CARD=${take[0]} CARDS=$(IFS=,; echo "${take[*]}")
    echo "[$(date '+%F %T')] cards $CARDS idle for $MINUTES min: taking them"
    exec "$@"
  fi
  sleep 30
done
