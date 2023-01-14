from util import compact_intervals, get_interval_compliment
import numpy as np
import logging


def test_compact_intervals():
    tests = [([], 10, []),
             ([(1, 4)], 10, [(1, 4)]),
             ([(1, 4), (7, 9)], 10, [(1, 4), (7, 9)]),
             ([(1, 4), (3, 5), (7, 9)], 10, [(1, 5), (7, 9)]),
             ([(1, 4), (5, 8), (7, 9)], 10, [(1, 4), (5, 9)]),
             ([(1, 4), (3, 8), (7, 9)], 10, [(1, 9)]),

             ([(0, 3), (2, 4)], 10, [(0, 4)]),
             ([(-1, 4)], 10, [(0, 4)]),
             ([(7, 9), (8, 10)], 10, [(7, 10)]),
             ([(8, 10)], 10, [(8, 10)]),

             ([(0, 10)], 10, [(0, 10)]),
             ([(0, 11), (1, 2)], 10, [(0, 10)]),
             ]

    for test_ind, (intervals_in, max_val, intervals_out) in enumerate(tests):
        output = compact_intervals(intervals_in, max_val)
        assert np.array_equal(output, np.array(intervals_out)), "Failed test %i, got %s instead of %s." % (
            test_ind, output, intervals_out)


def test_get_interval_compliment():
    tests = [([], 10, [(0, 10)]),
             ([(1, 4), (7, 8)], 10, [(0, 1), (4, 7), (8, 10)])]

    for test_ind, (intervals_in, max_val, intervals_out) in enumerate(tests):
        output = get_interval_compliment(intervals_in, max_val)
        assert np.array_equal(output, np.array(intervals_out)), "Failed test %i, got %s instead of %s." % (
            test_ind, output, intervals_out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_compact_intervals()
    test_get_interval_compliment()
    print("All tests pass.")
