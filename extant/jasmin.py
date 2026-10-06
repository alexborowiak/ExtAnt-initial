"""The JASMIN Dask cluster: start or reuse one, and send this package to its workers."""

import tempfile
import time
import zipfile
from pathlib import Path

GATEWAY_URL = "https://dask-gateway.jasmin.ac.uk"
PACKAGE = Path(__file__).resolve().parent


def connect(n_workers=20, wait_for=6, account="extant", worker_memory=16, qos="standard"):
    """Reuse the running Dask Gateway cluster if there is one, otherwise start one, and scale it.

    Args:
        n_workers (int): Workers to scale to.
        wait_for (int): Workers to wait for before returning.
        account (str): JASMIN account the workers are charged to.
        worker_memory (int): GB per worker (one core, one thread each).
        qos (str): Slurm quality of service.

    Returns:
        tuple: (gateway, cluster, client).
    """
    import dask_gateway

    gateway = dask_gateway.Gateway(GATEWAY_URL, auth="jupyterhub")
    running = gateway.list_clusters()
    if running:
        cluster = gateway.connect(running[0].name)
    else:
        options = gateway.cluster_options()
        options.account = account
        options.worker_cores = 1
        options.worker_threads = 1
        options.scheduler_cores = 1
        options.worker_memory = worker_memory
        options.qos = qos
        cluster = gateway.new_cluster(options, shutdown_on_close=True)
    cluster.scale(n_workers)
    client = cluster.get_client()
    client.wait_for_workers(wait_for)
    return gateway, cluster, client


def stop_all(gateway):
    """Stop every cluster this user has running (e.g. ones left over from a crashed kernel)."""
    for running in gateway.list_clusters():
        gateway.stop_cluster(running.name)


def package_zip(directory):
    """Zip the package's .py files into ``directory``, with a unique name; return the zip's path."""
    archive = Path(directory) / f"extant_{time.time_ns()}.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for file in sorted(PACKAGE.rglob("*.py")):
            zf.write(file, file.relative_to(PACKAGE.parent))
    return archive


def upload_package(client):
    """Send this package to every worker (and to workers that join later). Re-run after editing it.

    Code that runs on the workers (the rolling quantiles, LOWESS, the
    bootstrap, the ERA5 monthly means) lives in the package, so the workers
    need a copy. Each upload has its own file name, and workers forget the
    old copy first, so an upload after an edit always takes effect.
    """
    def forget_package():
        #(c): Defined here, not at module level, so it is pickled by value: the first time, workers have no extant
        import sys
        for name in [name for name in sys.modules if name == "extant" or name.startswith("extant.")]:
            del sys.modules[name]

    client.run(forget_package)
    with tempfile.TemporaryDirectory() as tmp:
        client.upload_file(str(package_zip(tmp)))
