import multiprocessing

from tqdm import tqdm
import psutil


class Decoder:

    def __init__(self, vocab, num_workers=4, fix_cpu_per_process=True):
        self.num_workers = num_workers
        self.vocab = vocab
        self.fix_cpu_per_process = fix_cpu_per_process

    def decode(self, probs):
        pass

    def decode_batch(self, prob_list):
        if psutil.LINUX:
            p = psutil.Process()
            available_cpus = p.cpu_affinity()
        else:
            available_cpus = [None]*1000

        num_procs = min(self.num_workers, len(available_cpus))
        decode_procs = []

        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()

        for i in range(num_procs):
            if self.fix_cpu_per_process:
                cpu_id = available_cpus[i]
            else:
                cpu_id = None

            decode_procs.append(
                DecoderProcess(self, cpu_id, tasks, results)
            )

        for p in decode_procs:
            p.start()

        for index, probs in enumerate(prob_list):
            tasks.put((index, probs))

        predictions = []

        for i in tqdm(range(len(prob_list))):
            predictions.append(results.get())

        # Send end
        for i in range(num_procs):
            tasks.put(None)

        predictions = sorted(predictions, key=lambda x: x[0])
        return [p[1] for p in predictions]


class DecoderProcess(multiprocessing.Process):

    def __init__(self, decoder, cpu_id, task_queue, result_queue):
        multiprocessing.Process.__init__(self)

        self.decoder = decoder
        self.cpu_id = cpu_id
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        if self.cpu_id is not None:
            p = psutil.Process()
            p.cpu_affinity(cpus=[self.cpu_id])

        for index, probs in iter(self.task_queue.get, None):
            prediction = self.decoder.decode(probs)
            self.result_queue.put((index, prediction))
