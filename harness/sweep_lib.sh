#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# sweep_lib.sh — Shared library for the v2 DFlash parameter sweep.
#
# Provides:
#   - launch_config()  : start vLLM container with one parameter combination
#   - wait_for_ready() : poll /v1/models until 200, with deadline + log-tail err detect
#   - run_throughput() : execute the v2 throughput harness against the running server
#   - run_prefill()    : execute the prefill harness (TTFT + tok/s @ 8k/16k/32k/64k/128k)
#   - stop_config()    : tear down vLLM container cleanly
#   - run_quality()    : execute HumanEval + MBPP @ c=8 against the running server
#   - eta_remaining()  : print elapsed/remaining/finish ETA
#
# Each config is identified by a LABEL like "A_b8k_c64_n8" and writes outputs to
# ${STUDY_ROOT}/configs/${STAGE}/${LABEL}/{server_args.txt,server.log,gates.json,
#                                          throughput.json,prefill.json,
#                                          humaneval_summary.json,mbpp_summary.json,
#                                          elapsed.txt}
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

STUDY_ROOT="${STUDY_ROOT:-/tmp/qwen-bench-2026-05-dflash-v2-sweep}"
PORT="${PORT:-11435}"
IMAGE="${IMAGE:-repne/vllm:v2}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
SERVED_NAME="${SERVED_NAME:-Qwen3.6-27B}"
DRAFTER="${DRAFTER:-z-lab/Qwen3.6-27B-DFlash}"
CONTAINER_NAME="vllm-sweep"
GPU_0_UUID="GPU-ba6334bc-6fec-5f2c-df75-a887bbca476e"
GPU_1_UUID="GPU-538bf008-7ff2-0d1d-69e9-20db81a00459"

BENCH_PY="/home/josh/qwen-vllm-test/llm-inference-bench/.venv/bin/python"
BENCH_SCRIPT="/home/josh/qwen-vllm-test/llm-inference-bench/llm_decode_bench.py"
STRESS_HARNESS="/home/josh/qwen-vllm-test/bench/stress-harness/stress_harness.py"
PROBLEMS_DIR="/home/josh/qwen-vllm-test/bench/stress-harness/problems"

HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token" 2>/dev/null || echo '')"

# Use linuxbrew python for harness (system Python is 3.14 which broke our bench .venv)
PY3=python3

# ──────────────────────────────────────────────────────────────────────────────
# launch_config LABEL MAX_BATCHED CAPTURE_SIZE NUM_SPEC OUT_DIR
# Spins up the vLLM container in detached mode. Returns 0 on detached-start.
# wait_for_ready must be called afterwards.
# ──────────────────────────────────────────────────────────────────────────────
launch_config() {
  local label="$1" max_batched="$2" capture="$3" num_spec="$4" out_dir="$5"
  mkdir -p "$out_dir"

  cat > "$out_dir/server_args.txt" <<EOF
LABEL=$label
IMAGE=$IMAGE
MODEL=$MODEL
DRAFTER=$DRAFTER
TP=2 (GPU0+GPU1 by UUID)
max-num-batched-tokens=$max_batched
max-cudagraph-capture-size=$capture
num_speculative_tokens=$num_spec
spec.method=dflash
spec.attention_backend=flashinfer
spec.use_local_argmax_reduction=true
attention-backend=flashinfer
gpu-memory-utilization=0.80
max-model-len=262144
max-num-seqs=128
launched_at=$(date -Iseconds)
EOF

  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

  docker run -d --name "$CONTAINER_NAME" \
    --device "nvidia.com/gpu=${GPU_0_UUID}" \
    --device "nvidia.com/gpu=${GPU_1_UUID}" \
    --ipc=host --shm-size=32g \
    --ulimit memlock=-1 --ulimit stack=67108864 --network host \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -v "${HOME}/.cache/vllm:/root/.cache/vllm" \
    -v "${HOME}/.cache/flashinfer:/root/.cache/flashinfer" \
    -v "${HOME}/.triton/cache:/root/.triton/cache" \
    -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -e "OMP_NUM_THREADS=8" \
    -e "VLLM_WORKER_MULTIPROC_METHOD=spawn" \
    -e "VLLM_ALLREDUCE_USE_SYMM_MEM=0" \
    "$IMAGE" \
      -O3 \
      --model "$MODEL" \
      --served-model-name "$SERVED_NAME" \
      --port "$PORT" \
      --tensor-parallel-size 2 \
      --gpu-memory-utilization 0.80 \
      --max-model-len 262144 \
      --max-num-seqs 128 \
      --max-num-batched-tokens "$max_batched" \
      --max-cudagraph-capture-size "$capture" \
      --language-model-only \
      --enable-auto-tool-choice \
      --reasoning-parser qwen3 \
      --tool-call-parser qwen3_coder \
      --enable-prefix-caching \
      --speculative-config.method dflash \
      --speculative-config.model "$DRAFTER" \
      --speculative-config.num_speculative_tokens "$num_spec" \
      --speculative-config.use_local_argmax_reduction true \
      --speculative-config.attention_backend flashinfer \
      --attention-backend flashinfer \
      --default-chat-template-kwargs.preserve_thinking true \
      >/dev/null
}

# ──────────────────────────────────────────────────────────────────────────────
# wait_for_ready OUT_DIR DEADLINE_SECONDS
# Tails container log for fatal errors, polls /v1/models. Returns:
#   0 = ready
#   1 = engine error (look at $OUT_DIR/server.log)
#   2 = timeout
# ──────────────────────────────────────────────────────────────────────────────
wait_for_ready() {
  local out_dir="$1" deadline="${2:-600}"
  local start=$(date +%s) now
  while true; do
    now=$(date +%s)
    if [ $((now - start)) -ge "$deadline" ]; then
      docker logs "$CONTAINER_NAME" > "$out_dir/server.log" 2>&1 || true
      echo "  [TIMEOUT] $((now-start))s elapsed, no /v1/models" >&2
      return 2
    fi
    if curl -s -m 3 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
      docker logs "$CONTAINER_NAME" > "$out_dir/server.log" 2>&1 || true
      echo "  [READY] in $((now-start))s" >&2
      return 0
    fi
    if docker logs --tail 30 "$CONTAINER_NAME" 2>&1 \
         | grep -qiE 'TypeError|ValueError|RuntimeError|Engine core init.*failed|CUDA error|out of memory|Traceback \(most recent'; then
      docker logs "$CONTAINER_NAME" > "$out_dir/server.log" 2>&1 || true
      echo "  [ENGINE-ERROR] check $out_dir/server.log" >&2
      return 1
    fi
    sleep 6
  done
}

# ──────────────────────────────────────────────────────────────────────────────
# settle SECONDS - post-ready settle per harness SOP
# ──────────────────────────────────────────────────────────────────────────────
settle() {
  local sec="${1:-60}"
  sleep "$sec"
}

# ──────────────────────────────────────────────────────────────────────────────
# run_throughput OUT_DIR
# Runs the v2 throughput matrix exactly as Repne specified:
#   --concurrency 1,2,4 --contexts 0,16k,32k,64k,128k --duration 60
#   --decode-warmup-seconds 20 --skip-prefill
# Writes:
#   $OUT_DIR/throughput.json
# ──────────────────────────────────────────────────────────────────────────────
run_throughput() {
  local out_dir="$1"
  "$BENCH_PY" "$BENCH_SCRIPT" \
    --host localhost --port "$PORT" --model "$SERVED_NAME" \
    --concurrency 1,2,4 \
    --contexts 0,16k,32k,64k,128k \
    --duration 60 \
    --decode-warmup-seconds 20 \
    --skip-prefill \
    --display-mode plain \
    --output "$out_dir/throughput.json" \
    --no-calibration-cache \
    < /dev/null >> "$out_dir/throughput.log" 2>&1
}

# ──────────────────────────────────────────────────────────────────────────────
# run_prefill OUT_DIR
# Runs the prefill (TTFT/tok-s) measurement per Repne's reference table:
#   contexts 8k,16k,32k,64k,128k, N=1 each
# Writes:
#   $OUT_DIR/prefill.json
# ──────────────────────────────────────────────────────────────────────────────
run_prefill() {
  local out_dir="$1"
  "$BENCH_PY" "$BENCH_SCRIPT" \
    --host localhost --port "$PORT" --model "$SERVED_NAME" \
    --concurrency 1 \
    --prefill-contexts 8k,16k,32k,64k,128k \
    --prefill-duration 10 \
    --standalone-prefill \
    --display-mode plain \
    --output "$out_dir/prefill.json" \
    --no-calibration-cache \
    < /dev/null >> "$out_dir/prefill.log" 2>&1
}

# ──────────────────────────────────────────────────────────────────────────────
# run_gates OUT_DIR
# Quick 4-gate sanity (Fibonacci 5x, tool, reasoning, multi-turn).
# Writes $OUT_DIR/gates.json. Returns 0 if 4/4 pass, 1 otherwise.
# ──────────────────────────────────────────────────────────────────────────────
run_gates() {
  local out_dir="$1"
  "$PY3" - "$out_dir" <<'PY' 2>&1 | tee "$out_dir/gates.log"
import json, os, sys, requests
OUT_DIR=sys.argv[1]
URL='http://localhost:11435/v1/chat/completions'; H={'Content-Type':'application/json'}
def ask(msgs, max_tokens=4096, tools=None):
    p={'model':'Qwen3.6-27B','messages':msgs,'temperature':0.0,'max_tokens':max_tokens}
    if tools: p['tools']=tools; p['tool_choice']='auto'
    r=requests.post(URL,headers=H,json=p,timeout=300).json()
    return r['choices'][0]['message']

results={}
# Gate 1 — Fibonacci 5x
try:
    ok=0
    for i in range(5):
        m=ask([{'role':'user','content':'Output the first 10 Fibonacci numbers as a comma-separated list (start: 1, 1).'}])
        c=(m.get('content') or '').strip()
        if '1, 1, 2, 3, 5, 8, 13, 21, 34, 55' in c: ok+=1
    results['fib_5x']=[ok,5]
    print(f"Gate 1 (Fibonacci 5x): {ok}/5")
except Exception as e:
    print(f"Gate 1: EXCEPTION {e}"); results['fib_5x']=[0,5]

# Gate 2 — Tool call
try:
    m=ask([{'role':'user','content':'What is the current weather in Tokyo? Use the tool.'}],
          tools=[{'type':'function','function':{'name':'get_weather','description':'Get weather','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}])
    tcs=m.get('tool_calls') or []
    ok=any(tc['function']['name']=='get_weather' and 'tokyo' in tc['function']['arguments'].lower() for tc in tcs)
    results['tool_call']=ok
    print(f"Gate 2 (Tool call): {'PASS' if ok else 'FAIL'}")
except Exception as e:
    print(f"Gate 2: EXCEPTION {e}"); results['tool_call']=False

# Gate 3 — Reasoning
try:
    m=ask([{'role':'user','content':'What is 47 times 83? Show the result as a number only on the last line.'}], 8192)
    c=(m.get('content') or '').strip()
    ok='3901' in c
    results['reasoning_47x83']=ok
    print(f"Gate 3 (47x83=3901): {'PASS' if ok else 'FAIL'}")
except Exception as e:
    print(f"Gate 3: EXCEPTION {e}"); results['reasoning_47x83']=False

# Gate 4 — Multi-turn
try:
    msgs=[{'role':'user','content':'Imagine the temperature in Tokyo is 28C. Just acknowledge.'}]
    t1=ask(msgs,2048); msgs.append({'role':'assistant','content':(t1.get('content') or '')})
    msgs.append({'role':'user','content':'Now imagine Berlin is at 18C. Just acknowledge.'})
    t2=ask(msgs,2048); msgs.append({'role':'assistant','content':(t2.get('content') or '')})
    msgs.append({'role':'user','content':'Which of the two cities I mentioned is warmer? Answer in one short sentence.'})
    t3=ask(msgs,4096); t3c=(t3.get('content') or '').strip()
    ok='tokyo' in t3c.lower() and 'warm' in t3c.lower()
    results['multi_turn']=ok
    print(f"Gate 4 (multi-turn): {'PASS' if ok else 'FAIL'}")
except Exception as e:
    print(f"Gate 4: EXCEPTION {e}"); results['multi_turn']=False

passed=sum([results['fib_5x'][0]==5, results['tool_call'], results['reasoning_47x83'], results['multi_turn']])
print(f"\nGates: {passed}/4")
with open(os.path.join(OUT_DIR,'gates.json'),'w') as f:
    json.dump({'results':results,'gates_passed':passed,'gates_total':4},f,indent=2)
sys.exit(0 if passed==4 else 1)
PY
}

# ──────────────────────────────────────────────────────────────────────────────
# run_quality OUT_DIR
# HumanEval (164) + MBPP-sanitized (257) @ c=8, max_tokens=4096, temp=0.0.
# ──────────────────────────────────────────────────────────────────────────────
run_quality() {
  local out_dir="$1"
  local label="$(basename "$out_dir")"
  "$PY3" "$STRESS_HARNESS" \
    --url "http://localhost:${PORT}/v1/chat/completions" \
    --model "$SERVED_NAME" \
    --config-label "$label" \
    --benchmark humaneval \
    --problems-file "$PROBLEMS_DIR/humaneval.jsonl" \
    --concurrency 8 --max-tokens 4096 \
    --output "$out_dir/humaneval.jsonl" \
    > "$out_dir/humaneval.log" 2>&1 || true
  "$PY3" "$STRESS_HARNESS" \
    --url "http://localhost:${PORT}/v1/chat/completions" \
    --model "$SERVED_NAME" \
    --config-label "$label" \
    --benchmark mbpp \
    --problems-file "$PROBLEMS_DIR/mbpp.jsonl" \
    --concurrency 8 --max-tokens 4096 \
    --output "$out_dir/mbpp.jsonl" \
    > "$out_dir/mbpp.log" 2>&1 || true
}

# ──────────────────────────────────────────────────────────────────────────────
# stop_config
# ──────────────────────────────────────────────────────────────────────────────
stop_config() {
  docker stop -t 30 "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

# ──────────────────────────────────────────────────────────────────────────────
# eta_remaining CURRENT TOTAL ELAPSED_S PHASE_LABEL
# ──────────────────────────────────────────────────────────────────────────────
eta_remaining() {
  local cur="$1" total="$2" elapsed="$3" label="$4"
  if [ "$cur" -eq 0 ]; then
    echo "  [ETA] $label $cur/$total starting"; return
  fi
  local per=$(( elapsed / cur ))
  local remain=$(( per * (total - cur) ))
  local rh=$(( remain / 3600 ))
  local rm=$(( (remain % 3600) / 60 ))
  local finish_ts=$(( $(date +%s) + remain ))
  local finish_hhmm=$(date -d "@$finish_ts" '+%H:%M')
  echo "  [ETA] $label $cur/$total — ${rh}h ${rm}m remaining → ~${finish_hhmm} MSK finish"
}
