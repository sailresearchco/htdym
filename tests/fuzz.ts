import { test } from 'vitest';

// Time-budgeted fuzz driver. Each fuzz test iterates its body until its
// FUZZ_MS budget expires (default 125ms, keeping the local suite near
// one second of total fuzzing, CI sets minutes). The seed is random per
// run so coverage accumulates across runs. Every test prints its seed
// and iteration count: replay a failure by pinning FUZZ_SEED to the
// printed value with at least the same budget.
const env = (globalThis as { process?: { env?: Record<string, string> } }).process?.env ?? {};
const FUZZ_MS = Number(env.FUZZ_MS ?? 125);
const SEED = Number(env.FUZZ_SEED ?? Math.floor(Math.random() * 0xffffffff));

// mulberry32
function rng(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let x = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

export function fuzzTest(name: string, body: (rand: () => number) => void): void {
  // each test draws from its own stream, derived from the name so one
  // FUZZ_SEED pins them all
  let tag = 0;
  for (const c of name) tag = (Math.imul(tag, 31) + c.charCodeAt(0)) | 0;
  test(
    name,
    () => {
      const rand = rng(SEED ^ tag);
      const deadline = Date.now() + FUZZ_MS;
      let iters = 0;
      try {
        do {
          body(rand);
          iters++;
        } while (Date.now() < deadline);
      } catch (e) {
        // a failing body never reaches the log line, so carry the seed
        // on the error itself
        throw new Error(`failed at iter ${iters} with FUZZ_SEED=${SEED}`, { cause: e });
      }
      console.log(`${name}: ${iters} iters, FUZZ_SEED=${SEED}`);
    },
    // the deadline is only checked between iterations, so one slow body
    // call may overshoot it, give the runner generous slack
    FUZZ_MS + 60_000,
  );
}
