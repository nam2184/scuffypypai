import numpy as np
import random
import math
import libnum
from .utils import *
from .encoding import EncodedNumber
from tqdm import tqdm
import multiprocessing
import time
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])

class PaillierKeyGen:
    """ Object to Generate Public/Private Key pair """
    def __init__(self, key_length = 256):
        self.key_length = key_length
        length = self.key_length//2
        p = prime_generator(length)
        q = p
        while p == q:
           q = prime_generator(length)
        self.n = p*q
        self.glambda = math.lcm(p-1,q-1)
        self.g = random.choice(multiplicative(self.n))
        l = (pow(self.g, self.glambda, self.n*self.n)-1)//self.n
        self.gMu = libnum.invmod(l,self.n)

    def publickey(self):
        """ Returns PublicKeyO object, can be used to encrypt data """
        return PublicKey(self.n, self.g)

    def privatekey(self, public_key):
        """ Returns PrivateKey object, can be used to encrypt data """
        return PrivateKey(self.glambda,self.gMu, self.n, public_key)

class PublicKey:
    def __init__(self, n, g):
        self.n = n
        self.g = g
        self.n_squared = n*n
        self.max_int = n//3 -1
        self.r = random.randint(0, self.n_squared)

    def raw_encrypt(self, m, r):
          k1 = pow(self.g, m, self.n_squared)
          k2 = pow(r, self.n, self.n_squared)
          cipher = (k1* k2) % (self.n_squared)
          return cipher

    def encrypt(self, encoding, precision=None):
      if isinstance(encoding, EncodedNumber):
          cipher = self.raw_encrypt(encoding.encoding, self.r)
      else:
          encoding = EncodedNumber.encode(self, encoding, precision)
          cipher = self.raw_encrypt(encoding.encoding, self.r)
      return EncryptedNumber(self, cipher, encoding.exponent)

    def raw_add(self, cipher1, cipher2):
        return (cipher1 * cipher2) % (self.n_squared)

    def raw_mul(self, cipher, plaintext):
        if not isinstance(cipher, int):
            raise TypeError('Expected ciphertext to be int, not %s' %
                type(cipher))

        if plaintext < 0 or plaintext >= self.n:
            raise ValueError('Scalar out of bounds: %i' % plaintext)

        if self.n - self.max_int <= plaintext:
            # Very large plaintext, play a sneaky trick using inverses
            neg_c = libnum.invmod(cipher, self.n_squared)
            neg_scalar = self.n - plaintext
            return powmod(neg_c, neg_scalar, self.n_squared)
        else:
            return powmod(cipher, plaintext, self.n_squared)

class PrivateKey:
    def __init__(self,glambda, gMu, n, public_key):
        self.glambda = glambda
        self.gMu= gMu
        self.public_key = public_key
        self.n = n

    def repr(self):
        pub_repr = repr(self.public_key)
        return f"[{self.__class.__name}{pub_repr}]"

    def raw_decrypt(self, cipher):
        l = (pow(cipher, self.glambda, self.n*self.n)-1) // self.n
        mess= (l * self.gMu) % self.n
        return mess

    def decrypt(self, cipher):
        """Return the decrypted & decoded plaintext of *encrypted_number*. """

        encoded = self.decrypt_encoded(cipher)
        return encoded.decode()

    def decrypt_encoded(self, cipher, Encoding=None):
        if not isinstance(cipher, EncryptedNumber):
            raise TypeError('Expected encrypted_number to be an EncryptedNumber'
                            ' not: %s' % type(cipher))

        if self.public_key.n != cipher.public_key.n and self.public_key.g != cipher.public_key.g:
            raise ValueError('encrypted_number was encrypted against a '
                             'different key!')

        if Encoding is None:
            Encoding = EncodedNumber

        encoded = self.raw_decrypt(cipher.ciphertext())
        return Encoding(self.public_key, encoded,
                             cipher.exponent)


class EncryptedNumber(object):
    """Represents the Paillier encryption of a float or int."""

    def __init__(self, public_key, ciphertext, exponent=0):
        self.public_key = public_key
        self.__ciphertext = ciphertext
        self.exponent = exponent
        if isinstance(self.ciphertext, EncryptedNumber):
            raise TypeError('ciphertext should be an integer')
        if not isinstance(self.public_key, PublicKey):
            raise TypeError('public_key should be a PublicKey')

    def __add__(self, other):
        """Add an int, float, `EncryptedNumber` or `EncodedNumber`."""
        if isinstance(other, EncryptedNumber):
            return self._add_encrypted(other)
        elif isinstance(other, EncodedNumber):
            return self._add_encoded(other)
        else:
            return self._add_scalar(other)

    def __radd__(self, other):
        """Called when Python evaluates `34 + <EncryptedNumber>`
        Required for builtin `sum` to work.
        """
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, EncryptedNumber):
          raise ValueError("Not possible to multiply ciphertexts")
        if isinstance(other, EncodedNumber):
            encoding = other
        else:
            encoding = EncodedNumber.encode(self.public_key, other)
        product = self.public_key.raw_mul(self.ciphertext(), encoding.encoding)
        exponent = self.exponent + encoding.exponent

        return EncryptedNumber(self.public_key, product, exponent)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __truediv__(self, scalar):
        return self.__mul__(1 / scalar)

    def ciphertext(self):
        return self.__ciphertext

    def decrease_exponent_to(self, new_exp):
        if new_exp > self.exponent:
            raise ValueError('New exponent %i should be more negative than '
                             'old exponent %i' % (new_exp, self.exponent))
        multiplied = self * pow(EncodedNumber.BASE, self.exponent - new_exp)
        multiplied.exponent = new_exp
        return multiplied

    def _add_scalar(self, scalar):
        """Returns E(a + b), given self=E(a) and b.

        Args:
          scalar: an int or float b, to be added to `self`.
        """

        encoded = EncodedNumber.encode(self.public_key, scalar,
                                       max_exponent=self.exponent)

        return self._add_encoded(encoded)

    def _add_encoded(self, encoded):
        if self.public_key != encoded.public_key:
            raise ValueError("Attempted to add numbers encoded against "
                             "different public keys!")

        # In order to add two numbers, their exponents must match.
        a, b = self, encoded
        if a.exponent > b.exponent:
            a = self.decrease_exponent_to(b.exponent)
        elif a.exponent < b.exponent:
            b = b.decrease_exponent_to(a.exponent)

        encrypted_scalar = a.public_key.raw_encrypt(b.encoding, 1)

        sum_ciphertext = a.public_key.raw_add(a.ciphertext(), encrypted_scalar)
        return EncryptedNumber(a.public_key, sum_ciphertext, a.exponent)

    def _add_encrypted(self, other):
        """Returns E(a + b) given E(a) and E(b)."""

        if self.public_key != other.public_key:
            raise ValueError("Attempted to add numbers encrypted against "
                             "different public keys!")

        # In order to add two numbers, their exponents must match.
        a, b = self, other
        if a.exponent > b.exponent:
            a = self.decrease_exponent_to(b.exponent)
        elif a.exponent < b.exponent:
            b = b.decrease_exponent_to(a.exponent)

        sum_ciphertext = a.public_key.raw_add(a.ciphertext(), b.ciphertext())
        return EncryptedNumber(a.public_key, sum_ciphertext, a.exponent)

def encrypt_vector(features, min, max, public_key):
    encrypted_lst= []
    encrypted_sublst = []
    c = 0
    with tqdm(total=max-min, desc=f"Feature ", unit=" samples") as pbar:
      for i in range (min, max):
          for j in features[i]:
              encrypted_sublst.append(public_key.encrypt(j))
          c+=1
          encrypted_lst.append(encrypted_sublst)
          encrypted_sublst = []
          pbar.update(1)
    return encrypted_lst

def encrypt_vector_no_bar(features, min, max, public_key):
    encrypted_lst= []
    encrypted_sublst = []
    c = 0
    t1 = time.time()
    for i in range (min, max):
          for j in features[i]:
              encrypted_sublst.append(public_key.encrypt(j))
          c+=1
          if c % 500 == 0:
            logger.info(f"Encrypted {c} features")
          encrypted_lst.append(encrypted_sublst)
          encrypted_sublst = []
    t2 = time.time()
    logger.info(f'Time : {str(t2-t1)}')
    return encrypted_lst

def decrypt_vector(private_key, x):
    return np.array([private_key.decrypt(i) for i in x])

def multi_encrypt(features, min_val, max_val, public_key, max_processes = 3):
    batch_size = (max_val - min_val) // 5
    args_list = []
    start_num = min_val
    while start_num <= max_val:
        end_num = start_num + batch_size
        if start_num == max_val:
            break
        if end_num >= max_val :
            end_num = max_val
        logger.info(f"Start: {start_num}, End: {end_num}, Total Args: {len(args_list)}")
        args_list.append((features, start_num, end_num, public_key))
        start_num += batch_size

    with multiprocessing.Pool(max_processes) as pool:
        results = pool.starmap(encrypt_vector_no_bar, args_list)

    concatenated_list = []
    for result in results:
        concatenated_list.extend(result)

    return concatenated_list

