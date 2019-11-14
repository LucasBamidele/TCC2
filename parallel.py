import multiprocessing as mp
print("processors: ", mp.cpu_count())

pool = mp.Pool(mp.cpu_count())