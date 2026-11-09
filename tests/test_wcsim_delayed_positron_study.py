from pathlib import Path

import numpy as np

from studies.framework.core import SelectionBook
from studies.framework.datatools_wcsim_root import event_from_wcsim_file
from studies.wcsim_delayed_positrons.wcsim_delayed_positron_study import (
    event_hit_times,
    find_delayed_clusters,
)


def test_event_hit_times_adds_the_matching_trigger_offset_once():
    event = {
        "digi_hit_time": np.array([2.0, 5.0]),
        "digi_hit_trigger": np.array([0, 1]),
        "trigger_time": np.array([100.0, 200.0]),
    }
    times, mode = event_hit_times(event)
    np.testing.assert_array_equal(times, [102.0, 205.0])
    assert mode == "trigger_plus"


def test_datatools_root_event_adapter_preserves_times_and_truth_fields():
    class FakeDataToolsReader:
        def get_event(self, entry):
            self.entry = entry

        def get_event_info(self):
            return {"pid": 211, "energy": 800.0}

        def get_digitized_hits(self):
            return {"time": np.array([2.0]), "pmt": np.array([7]),
                    "charge": np.array([1.5]), "trigger": np.array([0])}

        def get_triggers(self):
            return {"time": np.array([100.0]), "type": np.array([0])}

    event = event_from_wcsim_file(FakeDataToolsReader(), 4, Path("sample.root"))
    assert event["event_id"] == 4
    assert event["pid"] == 211
    assert event["energy"] == 800.0
    np.testing.assert_array_equal(event["digi_hit_time"], [2.0])
    # The source preserves DataTools values; trigger-time addition belongs to
    # the single downstream event_hit_times conversion.
    times, mode = event_hit_times(event)
    np.testing.assert_array_equal(times, [102.0])
    assert mode == "trigger_plus"


def test_rank_one_is_most_distinct_pmts_not_second_in_time():
    # Ten PMTs form the prompt and next-in-time candidate; twelve distinct
    # PMTs form a later, higher-ranked candidate.
    prompt_t = np.arange(10, dtype=float)
    next_t = 250.0 + np.arange(10, dtype=float)
    later_t = 500.0 + np.arange(12, dtype=float)
    times = np.concatenate((prompt_t, next_t, later_t))
    pmts = np.concatenate((np.arange(10), np.arange(10), np.arange(12)))
    charges = np.ones(times.size)
    prompt, candidates = find_delayed_clusters(
        times, pmts, charges, width_ns=50, min_pmts=10,
        prompt_min_pmts=10, search_start_ns=200, search_end_ns=1000,
    )
    assert prompt is not None
    assert candidates
    assert candidates[0].n_pmts == 12
    assert candidates[0].center_ns > 450


def test_selection_book_masks_are_parented_without_overwriting():
    book = SelectionBook(3)
    parent = book.add("prompt", [True, False, True])
    child = book.add("candidate", [False, True, True], parent="prompt")
    np.testing.assert_array_equal(parent, [True, False, True])
    np.testing.assert_array_equal(child, [False, False, True])
    assert book.counts() == {"all": 3, "prompt": 2, "candidate": 1}
