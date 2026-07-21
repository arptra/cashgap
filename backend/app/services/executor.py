from concurrent.futures import ThreadPoolExecutor


# A small pool keeps the local workstation responsive during CPU-heavy training.
worker_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cashgap-worker")

