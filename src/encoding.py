'''
/***************************************************************************************
*    Title: Encoding
*    Author: data61
*    Availability: https://github.com/data61/python-paillier?tab=readme-ov-file
*
***************************************************************************************/
'''

import fractions
import math
import sys


class EncodedNumber(object):
    BASE = 16
    """Base to use when exponentiating. Larger `BASE` means
    that :attr:`exponent` leaks less information. If you vary this,
    you'll have to manually inform anyone decoding your numbers.
    """
    LOG2_BASE = math.log(BASE, 2)
    FLOAT_MANTISSA_BITS = sys.float_info.mant_dig

    def __init__(self, public_key, encoding, exponent):
        self.public_key = public_key
        self.encoding = encoding
        self.exponent = exponent

    @classmethod
    def encode(cls, public_key, scalar, precision=None, max_exponent=None):
        """Return an encoding of an int or float."""

        # Calculate the maximum exponent for desired precision
        if precision is None:
            if isinstance(scalar, int):
                prec_exponent = 0
            elif isinstance(scalar, float):
                # Encode with *at least* as much precision as the python float
                bin_flt_exponent = math.frexp(scalar)[1] #Base 2 exponent

                # The least significant bit has value 2 ** bin_lsb_exponent
                bin_lsb_exponent = bin_flt_exponent - cls.FLOAT_MANTISSA_BITS

                # What's the corresponding base BASE exponent? Round that down.
                prec_exponent = math.floor(bin_lsb_exponent / cls.LOG2_BASE)
            else:
                raise TypeError("Don't know the precision of type %s."
                                % type(scalar))
        else:
            prec_exponent = math.floor(math.log(precision, cls.BASE))

        # Exponents are negative for numbers < 1.
        # If we're going to store numbers with a more negative exponent than demanded by the precision, then we may as well bump up the actual precision.
        if max_exponent is None:
            exponent = prec_exponent
        else:
            exponent = min(max_exponent, prec_exponent)

        # Use rationals instead of floats to avoid overflow.
        int_rep = round(fractions.Fraction(scalar)
                        * fractions.Fraction(cls.BASE) ** -exponent)

        if abs(int_rep) > public_key.max_int:
            raise ValueError('Integer needs to be within +/- %d but got %d'
                             % (public_key.max_int, int_rep))

        # Wrap negative numbers by adding n
        return cls(public_key, int_rep % public_key.n, exponent)

    def decode(self):
        """Decode plaintext and return the result. """
        if self.encoding >= self.public_key.n:
            # Should be mod n
            raise ValueError('Attempted to decode corrupted number')
        elif self.encoding <= self.public_key.max_int:
            # Positive
            mantissa = self.encoding
        elif self.encoding >= self.public_key.n - self.public_key.max_int:
            # Negative
            mantissa = self.encoding - self.public_key.n
        else:
            raise OverflowError('Overflow detected in decrypted number')

        if self.exponent >= 0:
            # Integer multiplication. This is exact.
            return mantissa * self.BASE ** self.exponent
        else:
            # BASE ** -e is an integer, so below is a division of ints.
            # Not coercing mantissa to float prevents some overflows.
            try:
                return mantissa / self.BASE ** -self.exponent
            except OverflowError as e:
                raise OverflowError(
                    'decoded result too large for a float') from e

    def decrease_exponent_to(self, new_exp):
        """Return an `EncodedNumber` with same value but lower exponent."""

        if new_exp > self.exponent:
            raise ValueError('New exponent %i should be more negative than'
                             'old exponent %i' % (new_exp, self.exponent))
        factor = pow(self.BASE, self.exponent - new_exp)
        new_enc = self.encoding * factor % self.public_key.n
        return self.__class__(self.public_key, new_enc, new_exp)
