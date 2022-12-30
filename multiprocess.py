from os import environ, remove
from os.path import exists
from time import sleep
from pathlib import Path
from multiprocessing import Pool, Process, cpu_count, JoinableQueue
from multiprocessing.managers import BaseManager

from project_globals import ChartYear, Environment

process_running_semaphore = Path(environ['TEMP'] + '/process_running_semaphore.tmp')

def pm_init(): print(f'+   multiprocessing shared object manager', flush=True)
pool_notice = '(Pool)'
job_queue = JoinableQueue()
result_lookup = None
chart_yr = None
environs = None
jm = None

class JobManager:

    def __init__(self, q, lookup):
        remove(process_running_semaphore)
        print(f'+   job manager (Pool size = {cpu_count()})', flush=True)
        results = {}
        with Pool() as p:
            while True:
                while not q.empty():
                    job = q.get()
                    results[job] = p.apply_async(job.execute, callback=job.execute_callback, error_callback=job.error_callback)
                jobs = list(results.keys())
                for job in jobs:
                    if results[job].ready():
                        result = results[job].get()
                        lookup[result[0]] = result[1]
                        del results[job]
                        q.task_done()
                sleep(0.1)

    # def __del__(self):
    #     print(f'Deleting Job Manager', flush=True)

class WaitForProcess(Process):
    def start(self, **kwargs):
        open(process_running_semaphore, 'w').close()
        # noinspection PyArgumentList
        super().start(**kwargs)
        while exists(process_running_semaphore):
            sleep(0.1)

    # def __del__(self):
    #     print(f'Deleting Wait for Process', flush=True)

class SharedObjectManager(BaseManager): pass
SharedObjectManager.register('ENV', Environment)
SharedObjectManager.register('CY', ChartYear)
som = SharedObjectManager()
