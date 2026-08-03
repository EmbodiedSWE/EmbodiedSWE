"""Fill the sandbox pool ahead of runs: create N sandboxes and provision each with the full
environment (ENV_CACHE fast path when present), so agent runs claim a ready sandbox and start
in ~90 s instead of paying the 40-90 min acquisition themselves.

run_agent_sandbox.py has always pointed at this script in its "pool is empty" error; this file
makes that instruction true. GPU COUNT IS FIXED AT SANDBOX CREATION — a pool of 1-GPU
sandboxes can never serve a 2-GPU condition (discovered 2026-08-01: every parameter_search run
inherited 1 GPU from the pool, making the documented second-GPU search placement impossible).
Prewarm with --gpu 2 for conditions that grant parameter_search.

    python3 prewarm_sandbox_pool.py --count 8 --gpu 2
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_agent_sandbox as R  # noqa: E402


async def warm_one(i: int, image: str, gpu: int, sem: asyncio.Semaphore) -> str | None:
    async with sem:
        t0 = time.time()
        print(f"[{i}] creating sandbox (gpu_count={gpu})", flush=True)
        try:
            emc, sandbox, portal = await R.create_sandbox(image, gpu, budget_min=0)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}] create failed: {type(exc).__name__}: {exc}", flush=True)
            return None
        uid = sandbox.id
        print(f"[{i}] {uid} created in {time.time() - t0:.0f}s; provisioning", flush=True)
        try:
            await R.provision_sandbox(sandbox, portal, allow_acquire=True)
        except BaseException as exc:  # noqa: BLE001 -- SystemExit included
            print(f"[{i}] {uid} provision FAILED: {exc}; deleting the sandbox", flush=True)
            try:
                # delete_sandbox wants the SESSION OBJECT (it calls sandbox.close());
                # passing the uid string dies with "'str' object has no attribute 'close'"
                await emc.delete_sandbox(sandbox)
                print(f"[{i}] {uid} deleted", flush=True)
            except Exception as del_exc:  # noqa: BLE001
                print(f"[{i}] {uid} DELETE FAILED — the sandbox may still be holding "
                      f"{gpu} GPU(s), delete it by hand: {del_exc}", flush=True)
            R.pool_drop(uid)
            return None
        R.pool_set(uid, state="idle", pid=None, image=image, gpu=gpu,
                   created=time.time(), last_used=time.time())
        print(f"[{i}] {uid} READY (gpu={gpu}) in {(time.time() - t0) / 60:.0f} min", flush=True)
        return uid


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, required=True, help="sandboxes to add to the pool")
    ap.add_argument("--gpu", type=int, default=1,
                    help="gpu_count per sandbox — FIXED AT CREATION, so match the condition "
                         "the pool will serve (2 when parameter_search is granted)")
    ap.add_argument("--image", default=R.DEFAULT_IMAGE)
    ap.add_argument("--concurrency", type=int, default=2,
                    help="parallel provisions; more contends on the ENV_CACHE transfer")
    args = ap.parse_args()

    sem = asyncio.Semaphore(max(1, args.concurrency))
    results = await asyncio.gather(
        *[warm_one(i, args.image, args.gpu, sem) for i in range(args.count)])
    ready = [u for u in results if u]
    print(f"PREWARM_DONE: {len(ready)}/{args.count} ready: {ready}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
