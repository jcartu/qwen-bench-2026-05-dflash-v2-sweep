@Repne ran the 3×3 grid you asked for last night on 2× RTX PRO 6000 (TP=2) with `repne/vllm:v2`. Full writeup with heatmap + raw `throughput.json` per cell here:

https://github.com/jcartu/qwen-bench-2026-05-dflash-v2-sweep

**TL;DR for `--max-num-batched-tokens × --max-cudagraph-capture-size`:**

```
                   capture=64    capture=128   capture=256
batched=8192       187.4         184.6         183.5
batched=16384      188.7         186.1         188.2
batched=32768      187.5         188.4         190.1  ★
```

Mean aggregate decode tok/s across 15 (concurrency × context) cells per config, `num_spec=8` fixed, `--decode-warmup-seconds 20 --duration 60`, 60s post-ready settle, all 9 configs passed 4/4 server gates.

**Verdict: this axis is essentially flat — full spread is 3.6%.** Your published default `batched=32768 capture=256` wins by a margin small enough to be near-noise. So pick whatever the rest of your stack wants here.

---

**Bonus** — at the winner cell I also swept `--speculative-config.num_speculative_tokens ∈ {4, 8, 15, 16}`. This axis is the opposite of flat:

| n | tok/s | spec accept | vs n=8 |
|---|---|---|---|
| 4  | 72.1   | 0.5%  | **−62.1%** |
| **8**  | **190.0**  | **23.1%** | **★ winner** |
| 15 | 176.3  | 12.1% | −7.2% |
| 16 | 109.1  | 5.3%  | **−42.6%** |

There's a sharp cliff between `n=15` and `n=16` — looks like a fastpath/graph-capture boundary. Worth a peek. n=8 stays the right call.

Heatmap PNG (for the channel):
https://github.com/jcartu/qwen-bench-2026-05-dflash-v2-sweep/blob/main/docs/images/stage_a_heatmap.png

num_spec curve:
https://github.com/jcartu/qwen-bench-2026-05-dflash-v2-sweep/blob/main/docs/images/stage_b_curve.png

Total wall time 7h. Now indexed on https://github.com/jcartu/qwen-bench as study #3.
