import numpy as np

from studies.wcte_delayed_bursts.burst_tools import find_bursts, t0_group_centers


def test_t0_group_centers_merge_close_times_and_keep_bunches_separate():
    assert t0_group_centers([100, 105, 111, 430, 437]) == [105.0, 433.5]


def test_find_bursts_returns_thresholded_time_ordered_windows():
    times = np.r_[np.full(12, 100.0), np.full(15, 400.0)]
    charges = np.ones(len(times))
    pmts = np.arange(len(times))

    bursts = find_bursts(times, charges, pmts, min_pmts=10, width_ns=50,
                         min_separation_ns=100)

    assert [burst.center_ns for burst in bursts] == [125.0, 425.0]
    assert [burst.n_hits for burst in bursts] == [12, 15]
    assert [burst.n_pmts for burst in bursts] == [12, 15]


def test_find_bursts_rejects_mismatched_hit_arrays():
    try:
        find_bursts([100, 101], [1], [0, 1])
    except ValueError as error:
        assert "equal lengths" in str(error)
    else:
        raise AssertionError("Expected a ValueError for mismatched arrays")
