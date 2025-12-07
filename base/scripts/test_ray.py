import ray
import time

print("Scripts start...")
start = time.time()
ray.init(ignore_reinit_error=True)
print(f"Ray initialized in {time.time() - start:.2f}s")
ray.shutdown()
print("Ray shutdown")
ㅣ