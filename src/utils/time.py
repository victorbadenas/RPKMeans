import time

def timer(print_=False):
    def inner2(func):
        def inner(*args, **kwargs):
            st = time.time()
            ret = func(*args, **kwargs)
            if print_:
                print(f"{func.__name__} ran in {time.time()-st:.2f}s")
                return ret
            else:
                delta = time.time() - st
                return ret, delta
        return inner
    return inner2