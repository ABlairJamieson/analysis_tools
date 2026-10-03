from scripts.beamline_timing import correct_tdc_hits


def test_reference_correction_preserves_repeated_hits_and_bank_offsets():
    hits = [(31, 100.), (46, 200.), (0, 110.), (0, 410.), (32, 235.), (32, 535.)]
    corrected, refs = correct_tdc_hits(hits)
    assert refs == (100., 200.)
    assert [hit.corrected_ns for hit in corrected] == [None, None, 10., 310., 35., 335.]
    assert corrected[3].corrected_ns - corrected[2].corrected_ns == 300.


def test_missing_reference_never_silently_uses_raw_time():
    corrected, refs = correct_tdc_hits([(31, 100.), (0, 110.), (32, 235.)])
    assert refs == (100., None)
    assert [hit.corrected_ns for hit in corrected] == [None, 10., None]
