"""The exact Denef-Loeser reduction against direct rational evaluation of the sum."""
from fractions import Fraction
from math import prod
import random
import unittest

from engine.topological import _reduce


def value_at(text, point):
    """Evaluate the reduction's polynomial text in s at a rational point."""
    return eval(text.replace("^", "**"), {"__builtins__": {}}, {"s": point})


def numerical_data(seed, components, strata):
    """Numerical data with repeated candidate poles and strata whose sum is 1 at s = 0."""
    choice = random.Random(seed)
    numbers = {f"E{i}": (choice.randint(1, 12), choice.randint(1, 9)) for i in range(components)}
    names = list(numbers)
    rows, total = [], 0
    for _ in range(strata):
        chosen = choice.sample(names, choice.randint(1, 5))
        multiple = choice.choice((-3, -2, -1, 1, 2, 3))
        # The value of this term at s = 0 is the integer `multiple`.
        rows.append({"components": chosen, "euler_characteristic": multiple * prod(numbers[c][1] for c in chosen)})
        total += multiple
    numbers["B"] = (1, 1)
    rows.append({"components": ["B"], "euler_characteristic": 1 - total})
    return numbers, rows


class TopologicalReduction(unittest.TestCase):
    def test_sum_matches_direct_evaluation_in_any_order_of_the_strata(self):
        numbers, strata = numerical_data(7, 40, 1500)
        reduced = _reduce(numbers, strata)
        self.assertIsNotNone(reduced)
        _, numerator, denominator, poles = reduced
        for point in (Fraction(0), Fraction(1), Fraction(2, 3), Fraction(7, 5), Fraction(11)):
            direct = sum(Fraction(row["euler_characteristic"]) /
                         prod(numbers[c][0] * point + numbers[c][1] for c in row["components"]) for row in strata)
            self.assertEqual(value_at(numerator, point) / value_at(denominator, point), direct)
        self.assertEqual(value_at(numerator, Fraction(0)), value_at(denominator, Fraction(0)))
        candidates = {Fraction(-nu, order) for order, nu in numbers.values()}
        self.assertLessEqual({Fraction(pole["value"]) for pole in poles}, candidates)
        # The kernel adds the strata in its own order. The answer is the same
        # reduced fraction however they arrive.
        shuffled = strata[:]
        random.Random(11).shuffle(shuffled)
        for rows in (strata[::-1], shuffled):
            self.assertEqual(_reduce(numbers, rows), reduced)

    def test_strata_of_zero_euler_characteristic_are_validated_and_add_nothing(self):
        numbers, strata = numerical_data(3, 12, 60)
        reduced = _reduce(numbers, strata)
        silent = {"components": list(numbers)[:3], "euler_characteristic": 0}
        self.assertEqual(_reduce(numbers, [silent, *strata, silent]), reduced)
        from engine.polynomial import InputError
        with self.assertRaises(InputError):
            _reduce(numbers, [*strata, {"components": ["E0", "E0"], "euler_characteristic": 0}])


if __name__ == "__main__":
    unittest.main()
