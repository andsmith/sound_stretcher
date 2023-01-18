import logging
import os


def compact_intervals(ints, max_val):
    """
    Reduce improper interval list, out of bounds (clip) or overlapping (merge)

    :param ints: list of (start, stop) intervals, starts in order
    :param max_val:  final "stop" must be <= this
    :return:  interval list, all in [0, max_val], all non-overlapping
    """

    if len(ints) == 0:
        return []

    # overlaps
    new_ints = []
    cur_int = ints[0]
    for interval in ints[1:]:
        if cur_int[1] >= interval[0]:
            cur_int = cur_int[0], max(cur_int[1], interval[1])
        else:
            new_ints.append(cur_int)
            cur_int = interval
    new_ints.append(cur_int)

    # start
    if new_ints[0][0] < 0:
        new_ints[0] = (0, new_ints[0][1])
    # end
    if new_ints[-1][1] > max_val:
        new_ints[-1] = (new_ints[-1][0], max_val)

    return new_ints


def get_interval_compliment(intervals, max_val):
    """
    Get minimal intervals whose union with input is whole range
    (For ints)
    :param intervals:  list of pairs (low, high) of intervals, in order,
    :return:  list of anti_interval pairs, so union of interval list with anti_interval list is (0, max_val)
    """
    if len(intervals) == 0:
        anti_intervals = [(0, max_val)]
    else:
        anti_intervals = []
        if intervals[0][0] > 0:
            anti_intervals = [(0, intervals[0][0])]
        for seg_i, segment in enumerate(intervals):
            if seg_i < len(intervals) - 1:
                #  anti-interval starts at end of this interval, ends at beginning of next
                anti_intervals.append((segment[1], intervals[seg_i + 1][0]))

            else:
                anti_intervals.append((segment[1], max_val))
    return anti_intervals


def make_unique_filename(unversioned):
    if not os.path.exists(unversioned):
        return unversioned
    version = 0
    file_part, ext = os.path.splitext(unversioned)
    filename = "%s_%i%s" % (file_part, version, ext)
    while os.path.exists(filename):
        version += 1
        filename = "%s_%i%s" % (file_part, version, ext)

    return filename
