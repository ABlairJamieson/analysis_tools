import uproot
import awkward as ak
import math

class DataLoader:
    """
    A class for loading data the processed WCTE data and apply data quality cuts
    """

    def __init__(self, file_name, branches_to_load=None):
        #if branches to load is none then load all branches are loaded
        self.file_name = file_name
        # Keep caller-owned lists untouched; DataLoader manages its own branch set.
        self.branches_to_load = None if branches_to_load is None else list(branches_to_load)
        
        try:
            self.file = uproot.open(self.file_name)
        except Exception as e:
            raise RuntimeError(f"Failed to open ROOT file: {self.file_name}\n{e}")
        
        available_branches = set(self.file["WCTEReadoutWindows"].keys())
        self.available_branches = available_branches

        #must load the mask branches if they are not already loaded for data quality cuts
        if self.branches_to_load is not None:            
            if "window_data_quality_mask" not in self.branches_to_load and "window_data_quality_mask" in available_branches:
                self.branches_to_load.append("window_data_quality_mask")
            if "hit_pmt_readout_mask" not in self.branches_to_load and "hit_pmt_readout_mask" in available_branches:
                self.branches_to_load.append("hit_pmt_readout_mask")
            if "vme_digi_issues_bitmask" not in self.branches_to_load and "vme_digi_issues_bitmask" in available_branches:
                self.branches_to_load.append("vme_digi_issues_bitmask")
            if "vme_evt_quality_bitmask" not in self.branches_to_load and "vme_evt_quality_bitmask" in available_branches:
                self.branches_to_load.append("vme_evt_quality_bitmask")
            if "T5_HasValidHit" not in self.branches_to_load and "T5_HasValidHit" in available_branches:
                self.branches_to_load.append("T5_HasValidHit")
            if "T5_HasMultipleScintillatorsHit" not in self.branches_to_load and "T5_HasMultipleScintillatorsHit" in available_branches:
                self.branches_to_load.append("T5_HasMultipleScintillatorsHit")
            if "T5_HasInTimeWindow" not in self.branches_to_load and "T5_HasInTimeWindow" in available_branches:
                self.branches_to_load.append("T5_HasInTimeWindow")

        self.mPMT_data_quality_cuts = False
        self.vme_event_quality_cuts = False
        self.t5_event_quality_cuts = False
        self.tagged_gamma_selection = False
        self.tagged_gamma_settings = None


    def iterate(self, verbose=False,**kwargs):
        """Iterate over the tree in batches using uproot.iterate"""
        defaults = {
        "step_size": "100 MB",
        "library": "ak",
        }
        defaults.update(kwargs)
        yield from (
            self._apply_all_data_quality_cuts(batch,verbose)
            for batch in self.file["WCTEReadoutWindows"].iterate(
                expressions=self.branches_to_load,
                **defaults,
            )
        )
    
    def _apply_all_data_quality_cuts(self, batch, verbose=False):
        if verbose:
            print(f"\nBatch loaded with {len(batch)} events")
        if self.mPMT_data_quality_cuts:            
            batch = batch[batch["window_data_quality_mask"]==0]
            if verbose:
                print(f"After window_data_quality_mask cut: {len(batch)} events")

            batch["hit_pmt_calibrated_times"] =  batch["hit_pmt_calibrated_times"][batch["hit_pmt_readout_mask"]==0]
            batch["hit_pmt_charges"] =  batch["hit_pmt_charges"][batch["hit_pmt_readout_mask"]==0]
            batch["hit_mpmt_slot_ids"] =  batch["hit_mpmt_slot_ids"][batch["hit_pmt_readout_mask"]==0]
            batch["hit_pmt_position_ids"] =  batch["hit_pmt_position_ids"][batch["hit_pmt_readout_mask"]==0]
            batch["hit_pmt_readout_mask"] =  batch["hit_pmt_readout_mask"][batch["hit_pmt_readout_mask"]==0]

        if self.vme_event_quality_cuts:
                          
            batch = batch[(batch["vme_digi_issues_bitmask"]==0) & (batch["vme_evt_quality_bitmask"]==0)]
            if verbose:
                print(f"After vme_event_quality_cuts cut: {len(batch)} events")  
    
        if self.t5_event_quality_cuts:
            #a valid hit, only one hit in the main beam bunch, within time window
            batch = batch[(batch["T5_HasValidHit"]==True)&(batch["T5_HasMultipleScintillatorsHit"]==False)&(batch["T5_HasInTimeWindow"]==True)]
            if verbose:
                print(f"After t5_event_quality_cuts cut: {len(batch)} events")  

        if self.tagged_gamma_selection:
            keep = self._tagged_gamma_event_mask(batch)
            batch = batch[ak.Array(keep)]
            if verbose:
                print(f"After tagged-gamma selection: {len(batch)} events")

        return batch

    def _ensure_branches(self, branches):
        missing = set(branches) - self.available_branches
        if missing:
            raise ValueError("Tagged-gamma selection requires missing branches: "
                             + ", ".join(sorted(missing)))
        if self.branches_to_load is not None:
            for name in branches:
                if name not in self.branches_to_load:
                    self.branches_to_load.append(name)

    def apply_tagged_gamma_selection(
            self, *, t0_tdc_channels=(0, 1, 2, 3), t2_tdc_channels=(8,),
            t0_t2_tof_range_ns=None, tdc_channel_offsets_ns=None,
            require_clean_mpmt=True, require_hodoscope_hit=True,
            hd_tdc_channels=None, hc2_qdc_channels=(), hc2_qdc_max=None,
            hd_charge_thresholds=None, hd_qdc_channel_by_element=None,
            hd_tdc_channel_by_element=None, hd_pair_map=None,
            hd_pair_time_window_ns=10.0):
        """Enable a configurable tagged-gamma preselection.

        The defaults follow the tagged-gamma study: all four T0 channels and
        T2 channel 8 must be present, and at least one hodoscope TDC channel
        must be present. HD QDC cuts are disabled by default because their
        reliability is under investigation. No T5 or WCTE burst cut is added.

        ``t0_t2_tof_range_ns`` is optional. The slide's 3.012--4.296 ns range
        is meaningful only after the analysis-specific relative TDC-channel
        offsets are supplied in ``tdc_channel_offsets_ns``. Offsets are
        subtracted exactly once from reference-corrected times. HC2 and HD
        QDC channel mappings are explicit arguments rather than assumptions.
        If ``hd_pair_map`` is supplied, multiple charge-tagged elements must
        form configured front/back pairs and be time-coincident.
        """
        if t0_t2_tof_range_ns is not None:
            low, high = map(float, t0_t2_tof_range_ns)
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError("t0_t2_tof_range_ns must be a finite increasing pair")
            if tdc_channel_offsets_ns is None:
                raise ValueError("T0-T2 TOF cuts require tdc_channel_offsets_ns; "
                                 "the tagged-gamma TOF range includes relative channel corrections")
            missing_offsets = (set(map(int, t0_tdc_channels)) | set(map(int, t2_tdc_channels))) - set(
                map(int, tdc_channel_offsets_ns))
            if missing_offsets:
                raise ValueError("Supply an explicit relative offset (zero is allowed) for every T0/T2 channel; missing: "
                                 + ", ".join(map(str, sorted(missing_offsets))))
        offsets = {int(k): float(v) for k, v in (tdc_channel_offsets_ns or {}).items()}
        if any(not math.isfinite(value) for value in offsets.values()):
            raise ValueError("tdc_channel_offsets_ns values must be finite")
        if not math.isfinite(float(hd_pair_time_window_ns)) or hd_pair_time_window_ns < 0:
            raise ValueError("hd_pair_time_window_ns must be finite and nonnegative")
        if hc2_qdc_max is not None and not hc2_qdc_channels:
            raise ValueError("Specify hc2_qdc_channels when enabling the HC2 QDC veto")
        if hd_charge_thresholds is not None and hd_qdc_channel_by_element is None:
            raise ValueError("HD QDC cuts require hd_qdc_channel_by_element")

        default_hd_channels = {**{i: 32 + i for i in range(7)}, 7: 16,
                               **{i + 8: 17 + i for i in range(7)}}
        settings = {
            "t0_channels": tuple(map(int, t0_tdc_channels)),
            "t2_channels": tuple(map(int, t2_tdc_channels)),
            "tof_range": (tuple(map(float, t0_t2_tof_range_ns))
                          if t0_t2_tof_range_ns is not None else None),
            "tdc_offsets": offsets,
            "require_clean_mpmt": bool(require_clean_mpmt),
            "require_hodoscope_hit": bool(require_hodoscope_hit),
            "hd_tdc_channels": tuple(map(int, hd_tdc_channels or
                                          default_hd_channels.values())),
            "hc2_qdc_channels": tuple(map(int, hc2_qdc_channels)),
            "hc2_qdc_max": (float(hc2_qdc_max) if hc2_qdc_max is not None else None),
            "hd_charge_thresholds": ({int(k): float(v) for k, v in hd_charge_thresholds.items()}
                                      if hd_charge_thresholds is not None else None),
            "hd_qdc_channel_by_element": ({int(k): int(v) for k, v in hd_qdc_channel_by_element.items()}
                                           if hd_qdc_channel_by_element is not None else None),
            "hd_tdc_channel_by_element": ({int(k): int(v) for k, v in
                                           (hd_tdc_channel_by_element or default_hd_channels).items()}),
            "hd_pair_map": ({int(k): tuple(map(int, v)) for k, v in hd_pair_map.items()}
                            if hd_pair_map is not None else None),
            "hd_pair_time_window_ns": float(hd_pair_time_window_ns),
        }
        required = ["beamline_pmt_tdc_ids", "beamline_pmt_tdc_times"]
        if settings["require_clean_mpmt"]:
            required += ["window_data_quality_mask", "hit_pmt_calibrated_times", "hit_pmt_charges"]
        if settings["require_hodoscope_hit"] or settings["hd_pair_map"] is not None:
            required += ["beamline_pmt_tdc_ids"]
        if settings["hc2_qdc_max"] is not None or settings["hd_charge_thresholds"] is not None:
            required += ["beamline_pmt_qdc_ids", "beamline_pmt_qdc_charges"]
        self._ensure_branches(required)
        self.tagged_gamma_settings = settings
        self.tagged_gamma_selection = True

    def _tagged_gamma_event_mask(self, batch):
        cfg = self.tagged_gamma_settings
        tdc_ids = ak.to_list(batch["beamline_pmt_tdc_ids"])
        tdc_times = ak.to_list(batch["beamline_pmt_tdc_times"])
        n_events = len(tdc_ids)
        if cfg["require_clean_mpmt"]:
            quality = ak.to_list(batch["window_data_quality_mask"])
            pmt_times = ak.to_list(batch["hit_pmt_calibrated_times"])
            pmt_charges = ak.to_list(batch["hit_pmt_charges"])
        else:
            quality = [0] * n_events
            pmt_times = [[] for _ in range(n_events)]
            pmt_charges = [[] for _ in range(n_events)]
        if cfg["hc2_qdc_max"] is not None or cfg["hd_charge_thresholds"] is not None:
            qdc_ids = ak.to_list(batch["beamline_pmt_qdc_ids"])
            qdc_charges = ak.to_list(batch["beamline_pmt_qdc_charges"])
        else:
            qdc_ids = [[] for _ in range(n_events)]
            qdc_charges = [[] for _ in range(n_events)]

        keep = []
        for index in range(n_events):
            if cfg["require_clean_mpmt"]:
                if quality[index] is None or int(quality[index]) != 0:
                    keep.append(False)
                    continue
                if (pmt_times[index] is None or pmt_charges[index] is None
                        or len(pmt_times[index]) != len(pmt_charges[index])
                        or not all(x is not None and math.isfinite(float(x)) for x in pmt_times[index])
                        or not all(x is not None and math.isfinite(float(x)) for x in pmt_charges[index])):
                    keep.append(False)
                    continue

            if tdc_ids[index] is None or tdc_times[index] is None:
                keep.append(False)
                continue
            ids = [int(x) for x in tdc_ids[index]]
            raw_times = tdc_times[index]
            if (len(ids) != len(raw_times)
                    or any(x is None or not math.isfinite(float(x)) for x in raw_times)):
                keep.append(False)
                continue
            times = [float(x) for x in raw_times]
            if 31 not in ids or 46 not in ids:
                keep.append(False)
                continue
            ref31 = times[ids.index(31)]
            ref46 = times[ids.index(46)]
            corrected = {}
            for channel, time in zip(ids, times):
                if channel in (31, 46):
                    continue
                reference = ref31 if channel <= 31 else ref46
                corrected.setdefault(channel, time - reference - cfg["tdc_offsets"].get(channel, 0.0))
            if not all(channel in corrected for channel in cfg["t0_channels"]):
                keep.append(False)
                continue
            t2_values = [corrected[channel] for channel in cfg["t2_channels"] if channel in corrected]
            if not t2_values:
                keep.append(False)
                continue
            t0_time = sum(corrected[channel] for channel in cfg["t0_channels"]) / len(cfg["t0_channels"])
            t2_time = sum(t2_values) / len(t2_values)
            if cfg["tof_range"] is not None:
                low, high = cfg["tof_range"]
                tof = t2_time - t0_time
                if not low <= tof <= high:
                    keep.append(False)
                    continue

            if cfg["require_hodoscope_hit"] and not any(ch in corrected for ch in cfg["hd_tdc_channels"]):
                keep.append(False)
                continue

            charges = {}
            if (cfg["hc2_qdc_max"] is not None or cfg["hd_charge_thresholds"] is not None):
                if (qdc_ids[index] is None or qdc_charges[index] is None
                        or len(qdc_ids[index]) != len(qdc_charges[index])):
                    keep.append(False)
                    continue
            for channel, charge in zip(qdc_ids[index], qdc_charges[index]):
                if charge is not None and math.isfinite(float(charge)):
                    charges.setdefault(int(channel), float(charge))
            if cfg["hc2_qdc_max"] is not None and any(
                    charges.get(ch, float("-inf")) > cfg["hc2_qdc_max"]
                    for ch in cfg["hc2_qdc_channels"]):
                keep.append(False)
                continue

            thresholds = cfg["hd_charge_thresholds"]
            if thresholds is not None:
                qdc_map = cfg["hd_qdc_channel_by_element"]
                active = {element for element, threshold in thresholds.items()
                          if charges.get(qdc_map.get(element, -1), float("-inf")) >= threshold}
                if not active:
                    keep.append(False)
                    continue
                element_to_tdc = cfg["hd_tdc_channel_by_element"]
                if any(element not in element_to_tdc or element_to_tdc[element] not in corrected
                       for element in active):
                    keep.append(False)
                    continue
                if len(active) > 1 and cfg["hd_pair_map"] is not None:
                    valid_pair = False
                    for element in active:
                        pair = cfg["hd_pair_map"].get(element)
                        if pair is None or len(pair) != 2 or not set(pair).issubset(active):
                            continue
                        ch_a, ch_b = (element_to_tdc.get(pair[0]), element_to_tdc.get(pair[1]))
                        if ch_a in corrected and ch_b in corrected and abs(corrected[ch_a] - corrected[ch_b]) <= cfg["hd_pair_time_window_ns"]:
                            valid_pair = True
                            break
                    if not valid_pair:
                        keep.append(False)
                        continue
                elif len(active) > 1:
                    # The tagged-gamma criterion treats extra HD signals as
                    # valid only when a configured front/back pair coincides.
                    keep.append(False)
                    continue
            keep.append(True)
        return keep

    def apply_mPMT_data_quality_cuts(self):
        self.mPMT_data_quality_cuts = True
    
    def apply_vme_event_quality_cuts(self):
        self.vme_event_quality_cuts = True
    
    def apply_t5_event_quality_cuts(self):
        self.t5_event_quality_cuts = True

    def get_good_wcte_pmts(self):
        config = self.get_configuration()
        good_wcte_pmts = config["good_wcte_pmts"]
        good_wcte_pmts_slots = good_wcte_pmts//100
        good_wcte_pmts_positions = good_wcte_pmts%100
        return good_wcte_pmts_slots, good_wcte_pmts_positions
            
    def get_configuration(self):
        tree = self.file['Configuration']
        config = tree.arrays(library="ak", entry_start=0, entry_stop=1)
        return config[0]
    
    def get_data_quality_metrics(self):
        tree = self.file['DataQualityMetrics']
        data_quality_metrics = tree.arrays(library="ak", entry_start=0, entry_stop=1)
        return data_quality_metrics[0]
    
    def get_vme_analysis_scalar_results(self):
        tree = self.file['vme_analysis_scalar_results']
        vme_analysis_scalar_results = tree.arrays(library="ak", entry_start=0, entry_stop=1)
        return vme_analysis_scalar_results[0]
    
    def get_vme_analysis_run_info(self):
        tree = self.file['vme_analysis_run_info']
        vme_analysis_run_info = tree.arrays(library="ak", entry_start=0, entry_stop=1)
        return vme_analysis_run_info[0]
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()
