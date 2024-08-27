import time
import logging
import math
import random

try:
    import gmpy2
    HAVE_GMP = True
except ImportError:
    HAVE_GMP = False

try:
    from Crypto.Util import number
    HAVE_CRYPTO = True
except ImportError:
    HAVE_CRYPTO = False

_USE_MOD_FROM_GMP_SIZE = (1 << (16))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def powmod(a, b, c):
    """
    Uses GMP, if available, to do a^b mod c where a, b, c
    are integers.

    :return int: (a ** b) % c
    """
    if a == 1:
        return 1
    if not HAVE_GMP or max(a, b, c) < _USE_MOD_FROM_GMP_SIZE:
        return pow(a, b, c)
    else:
        return int(gmpy2.powmod(a, b, c))

def timer(func):
    def wrapper(*args,**kwargs):
        print(f"Start Executing {func.__name__}()")
        t1 = time.time()
        result = func(*args,**kwargs)
        t2 = time.time()
        print(f'Time : {str(t2-t1)}')
        return result
    return wrapper

def row_to_dict(row):
    return {
            "date" : row['Date'],
            "open" : row["Open"],
            "high" : row["High"],
            "low" : row["Open"],
            "close" : row["Close"],
            "volume" : row["Volume"],
            }


def prime(start, end):
    for i in range(start, end, 2):
        if is_prime(i):
            return i

def is_prime(n, k=25):
    """ Test if a number is prime
        Args:
            n -- int -- the number to test
            k -- int -- the number of tests to do
        return True if n is prime
    """
    # Test if n is not even.
    # But care, 2 is prime !
    if n == 2 or n == 3:
        return True
    if n <= 1 or n % 2 == 0:
        return False
    # find r and s
    s = 0
    r = n - 1
    while r & 1 == 0:
        s += 1
        r //= 2
    # do k tests
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, r, n)
        if x != 1 and x != n - 1:
            j = 1
            while j < s and x != n - 1:
                x = pow(x, 2, n)
                if x == 1:
                    return False
                j += 1
            if x != n - 1:
                return False
    return True


def prime_generator(length):
    min = (1 << (length-1))
    max = (1 << (length)) - 1
    random_start = random.randint(min,max)
    if random_start % 2 == 0 :
      random_start = random_start + 1
    return prime(random_start,max)



def multiplicative(number):
        gs = [i for i in range(1,10000) if math.gcd(number,i) == 1]
        return gs
