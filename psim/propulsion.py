import multiprocessing as mp
import sys

# Add parent directory to path to find engine_deck if necessary
sys.path.insert(1, '../')
from engine_deck import Turbofan_Deck

def initialize_deck(deck_name='PW2000_similar_deck.csv'):
        # Load deck logic inside the process to be safe across OSs
    try:
        # ############################################################################
        # Load Engine Deck
        # ############################################################################
        # Uses data and code from https://youtu.be/95Gy2wg3olE
        E1_deck = Turbofan_Deck('PW2000_similar_deck.csv')
        E2_deck = Turbofan_Deck('PW2000_similar_deck.csv')
        return (E1_deck, E2_deck)
    except Exception as e:
        print(f"[Engine Process] Failed to load deck: {e}")
        return

E1_deck, E2_deck = initialize_deck(deck_name='PW2000_similar_deck.csv')

# Instantiate the deck here (or inside the worker if you prefer isolation)
# Since the worker runs in a separate process, this global is safe for Linux (fork)
# For Windows (spawn), it's better to instantiate inside the worker or class.
# We will stick to the pattern used previously but encapsulated here.

def engine_worker(jobs_queue, results_queue):
    """
    Worker process for Engine Deck calculations.
    """
    print("[Engine Process] Worker started.")
    


    while True:
        try:
            job = jobs_queue.get()
            if job is None:
                print("[Engine Process] Shutdown signal.")
                break
            
            # Unpack: Alt (ft), Mach, TLA1, TLA2, Time
            job_alt, job_MN, job_E1_TLA, job_E2_TLA, job_on_ground, job_time = job
            E1_res = E1_deck.run_deck(job_alt, job_MN, job_E1_TLA, job_on_ground, job_time)
            E2_res = E2_deck.run_deck(job_alt, job_MN, job_E2_TLA, job_on_ground, job_time)
            results = (E1_res, E2_res)

            
            # Clear old results if queue is full (keep only freshest)
            if not results_queue.empty():
                try:
                    results_queue.get_nowait()
                except mp.queues.Empty:
                    pass
            results_queue.put(results)

        except Exception as e:
            print(f"[Engine Process] Error: {e}")
            break
    print("[Engine Process] Worker finished.")