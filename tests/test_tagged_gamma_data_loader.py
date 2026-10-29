import awkward as ak
import uproot

from analysis_tools.data_loader import DataLoader


def make_tagged_gamma_root(path):
    # Reference-subtracted T0 average is ~10.15 ns; T2 is 13.5 ns.
    channels = [31, 46, 0, 1, 2, 3, 8, 32]
    good = [100.0, 200.0, 110.0, 110.1, 110.2, 110.3,
            113.5, 210.0]
    no_t2 = [100.0, 200.0, 110.0, 110.1, 110.2, 110.3,
             0.0, 210.0]
    with uproot.recreate(path) as root:
        tree = root.mktree("WCTEReadoutWindows", {
            "beamline_pmt_tdc_ids": "var * int32",
            "beamline_pmt_tdc_times": "var * float64",
            "beamline_pmt_qdc_ids": "var * int32",
            "beamline_pmt_qdc_charges": "var * float64",
        })
        tree.extend({
            "beamline_pmt_tdc_ids": ak.Array([channels, channels, channels]),
            "beamline_pmt_tdc_times": ak.Array([good, good, no_t2]),
            "beamline_pmt_qdc_ids": ak.Array([[99], [99], [99]]),
            "beamline_pmt_qdc_charges": ak.Array([[300.0], [400.0], [300.0]]),
        })


def test_tagged_gamma_selection_uses_configured_tof_and_optional_hc2_veto(tmp_path):
    path = tmp_path / "tagged_gamma.root"
    make_tagged_gamma_root(path)
    requested = ["beamline_pmt_tdc_ids"]
    with DataLoader(path, requested) as loader:
        loader.apply_tagged_gamma_selection(
            require_clean_mpmt=False,
            t0_t2_tof_range_ns=(3.0, 4.0),
            tdc_channel_offsets_ns={channel: 0.0 for channel in (0, 1, 2, 3, 8)},
            hc2_qdc_channels=(99,),
            hc2_qdc_max=350.0,
        )
        batches = list(loader.iterate(step_size="10 MB"))
        assert sum(len(batch) for batch in batches) == 1
        assert requested == ["beamline_pmt_tdc_ids"]
        assert "beamline_pmt_tdc_times" in loader.branches_to_load


def test_tagged_gamma_hd_charge_cut_is_not_enabled_by_default(tmp_path):
    path = tmp_path / "tagged_gamma.root"
    make_tagged_gamma_root(path)
    with DataLoader(path) as loader:
        loader.apply_tagged_gamma_selection(require_clean_mpmt=False)
        batches = list(loader.iterate(step_size="10 MB"))
        # The first two events have all T0/T2/HD timing channels. The QDC
        # values are deliberately not used unless a charge cut is requested.
        assert sum(len(batch) for batch in batches) == 2

