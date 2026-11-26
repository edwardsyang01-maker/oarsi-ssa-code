from __future__ import annotations

from types import MappingProxyType

import numpy as np
from PIL import Image


METHOD = "conservative_geometry_candidates"
VERSION = "1.1.0"

MIN_CROP_VALID_STD_ENGINEERING = 0.01


class InvalidCropContentError(ValueError):
    pass


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


POLICY = _freeze(
    {
        "version": VERSION,
        "min_short_side_native": 32,
        "max_work_side": 640,
        "column_band_y_fraction": (0.15, 0.85),
        "joint_band_y_fraction": (0.28, 0.76),
        "joint_y_prior": (0.54, 0.22),
        "bilateral_search_x_fractions": ((0.12, 0.46), (0.54, 0.88)),
        "single_search_x_fraction": (0.10, 0.90),
        "auto_margin_x_fraction": 0.08,
        "auto_peak_min_prominence_fraction": 0.18,
        "auto_peak_min_absolute_prominence": 0.02,
        "auto_peak_min_width_fraction": 0.05,
        "auto_min_separation_fraction": 0.12,
        "auto_central_trough_max_ratio": 0.55,
        "auto_central_band_x_fraction": (0.45, 0.55),
        "peak_support_drop_fraction": 0.35,
        "peak_support_min_absolute_drop": 0.01,
        "min_tissue_std_uncalibrated": 0.025,
        "min_gap_contrast_uncalibrated": 0.002,
        "min_crop_valid_std_engineering": MIN_CROP_VALID_STD_ENGINEERING,
        "min_shared_support_uncalibrated": 0.12,
        "max_independent_y_disagreement_fraction": 0.12,
        "side_tissue_radius_fraction": 0.09,
        "shared_joint_radius_fraction": 0.04,
        "shared_profile_window_fraction": 0.08,
        "bilateral_box_edge_fraction": 0.40,
        "bilateral_box_min_edge_fraction": 0.24,
        "bilateral_box_spacing_factor": 0.90,
        "bilateral_edge_rule": "min(0.40W, 0.90*center_distance, W, H); min 0.24W or reject pair",
        "single_box_edge_fraction": 0.55,
        "max_crop_boundary_shift_fraction": 0.08,
        "column_smoothing_fraction": 0.015,
        "row_broad_smoothing_fraction": 0.035,
        "row_fine_smoothing_fraction": 0.004,
    }
)


def _validate_gray_and_mask(gray, valid, *, min_short_side):
    array = np.asarray(gray)
    if array.ndim != 2:
        raise ValueError("gray must be a 2-D array")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError("gray must have a floating dtype; integer/bool images are not accepted")
    if not np.isfinite(array).all():
        raise ValueError("gray must be all finite")
    height, width = array.shape
    if height < min_short_side or width < min_short_side:
        raise ValueError(f"gray shortest side must be at least {min_short_side} pixels")
    if float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise ValueError("gray must be in the native [0,1] range")
    if valid is None:
        mask = np.ones(array.shape, dtype=bool)
    else:
        mask = np.asarray(valid)
        if mask.dtype != bool:
            raise ValueError("valid must be a bool mask")
        if mask.shape != array.shape:
            raise ValueError("valid must have the same shape as gray")
        if not mask.any():
            raise ValueError("valid mask must not be empty")
        mask = mask.copy()
    source = np.asarray(array, dtype=np.float32).copy()
    return source, mask


def _smooth_1d(values, sigma):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    sigma = max(float(sigma), 0.5)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _max_filter_1d(values, size):
    values = np.asarray(values, dtype=np.float64)
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    radius = size // 2
    padded = np.pad(values, radius, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    return windows.max(axis=1)


def _resize_work(source, mask, max_side):
    height, width = source.shape
    scale = min(1.0, float(max_side) / max(height, width))
    work_w = max(16, int(round(width * scale)))
    work_h = max(16, int(round(height * scale)))
    work = np.asarray(
        Image.fromarray(np.ascontiguousarray(source.astype(np.float32)), mode="F").resize(
            (work_w, work_h), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    work_mask = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
            (work_w, work_h), Image.Resampling.NEAREST
        )
    ) > 0
    work = np.where(work_mask, work, 0.0)
    return work, work_mask


def _row_gap_response(work, center_x, radius, policy):
    height = work.shape[0]
    left = max(0, int(center_x) - int(radius))
    right = min(work.shape[1], int(center_x) + int(radius))
    if right <= left:
        left, right = 0, work.shape[1]
    row = work[:, left:right].mean(axis=1)
    broad = _smooth_1d(row, max(0.6, height * policy["row_broad_smoothing_fraction"]))
    fine = _smooth_1d(row, max(0.6, height * policy["row_fine_smoothing_fraction"]))
    response = np.maximum(broad - fine, 0.0)
    prior_center, prior_sigma = policy["joint_y_prior"]
    prior = np.exp(
        -0.5
        * ((np.arange(height, dtype=np.float64) - height * prior_center)
           / max(1.0, height * prior_sigma))
        ** 2
    )
    return response * prior


def _side_evidence(work, center_x, policy):
    height, width = work.shape
    y0 = int(round(height * policy["column_band_y_fraction"][0]))
    y1 = int(round(height * policy["column_band_y_fraction"][1]))
    radius = max(2, int(round(width * policy["side_tissue_radius_fraction"])))
    left = max(0, int(center_x) - radius)
    right = min(width, int(center_x) + radius)
    tissue = work[y0:y1, left:right]
    tissue_std = float(tissue.std()) if tissue.size else 0.0
    response = _row_gap_response(work, center_x, radius, policy)
    jy0 = int(round(height * policy["joint_band_y_fraction"][0]))
    jy1 = int(round(height * policy["joint_band_y_fraction"][1]))
    if jy1 <= jy0:
        jy0, jy1 = 0, height
    patch = response[jy0:jy1]
    gap_contrast = float(patch.max()) if patch.size else 0.0
    independent_y = jy0 + int(np.argmax(patch)) if patch.size else height // 2
    return {
        "center_x_work": float(center_x),
        "tissue_std_uncalibrated": tissue_std,
        "gap_contrast_uncalibrated": gap_contrast,
        "independent_joint_y_work": int(independent_y),
        "row_response": response,
    }


def _clip_square_box(cx, cy, edge, width, height):
    edge = int(max(1, min(round(edge), width, height)))
    requested_x = int(round(cx - edge / 2.0))
    requested_y = int(round(cy - edge / 2.0))
    left = min(max(requested_x, 0), width - edge)
    top = min(max(requested_y, 0), height - edge)
    shift = max(abs(left - requested_x), abs(top - requested_y)) / float(edge)
    bbox = [int(left), int(top), int(left + edge), int(top + edge)]
    return bbox, float(shift)


def _boxes_overlap(box_a, box_b):
    overlap_w = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    overlap_h = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    return overlap_w > 0 and overlap_h > 0


def _bilateral_edge_plan(
    centers_x, width, height, policy, carried_fraction=None, apply_spacing=True
):
    dx = float(centers_x[1]) - float(centers_x[0])
    nominal_fraction = float(policy["bilateral_box_edge_fraction"])
    min_fraction = float(policy["bilateral_box_min_edge_fraction"])
    spacing_factor = float(policy["bilateral_box_spacing_factor"])
    nominal_px = float(width) * nominal_fraction
    spacing_px = spacing_factor * dx
    cap_px = float(min(width, height))
    carried_px = (
        float(width) * float(carried_fraction)
        if carried_fraction is not None
        else nominal_px
    )
    candidates = [
        (carried_px, "carried_work_fraction" if carried_fraction is not None else "nominal"),
        (nominal_px, "nominal"),
    ]
    if apply_spacing:
        candidates.append((spacing_px, "spacing"))
    candidates.append((cap_px, "width_or_height_cap"))
    best_index = min(range(len(candidates)), key=lambda index: candidates[index][0])
    selected_float = candidates[best_index][0]
    limited_by = candidates[best_index][1]
    selected_px = int(max(1, min(int(round(selected_float)), width, height)))
    min_px = float(width) * min_fraction
    below_minimum = bool(selected_float < min_px)
    spacing_limited = bool(
        apply_spacing and spacing_px < min(nominal_px, carried_px, cap_px)
    )
    return {
        "rule": "min(0.40W, 0.90*center_distance, W, H)",
        "nominal_edge_px": float(nominal_px),
        "nominal_edge_fraction": nominal_fraction,
        "spacing_edge_px": float(spacing_px),
        "spacing_edge_fraction": float(spacing_px / float(max(width, 1))),
        "carried_work_edge_px": float(carried_px),
        "carried_work_edge_fraction": (
            None if carried_fraction is None else float(carried_fraction)
        ),
        "cap_edge_px": float(cap_px),
        "center_distance_px": float(dx),
        "center_distance_fraction": float(dx / float(max(width, 1))),
        "minimum_edge_px": float(min_px),
        "minimum_edge_fraction": min_fraction,
        "spacing_limited": spacing_limited,
        "spacing_constraint_applied": bool(apply_spacing),
        "limited_by": limited_by,
        "below_minimum_edge": below_minimum,
        "candidate_edge_px": int(selected_px),
        "candidate_edge_fraction": float(selected_px / float(max(width, 1))),
        "selected_edge_px": None if below_minimum else int(selected_px),
        "selected_edge_fraction": (
            None if below_minimum else float(selected_px / float(max(width, 1)))
        ),
    }


def _fit_bilateral_boxes(centers, edge, width, height, min_edge_px):
    edge = int(max(1, min(int(edge), width, height)))
    while edge > 0:
        boxes = [
            _clip_square_box(center_x, center_y, edge, width, height)[0]
            for center_x, center_y in centers
        ]
        if not _boxes_overlap(boxes[0], boxes[1]):
            return edge, boxes
        overlap_w = min(boxes[0][2], boxes[1][2]) - max(boxes[0][0], boxes[1][0])
        overlap_h = min(boxes[0][3], boxes[1][3]) - max(boxes[0][1], boxes[1][1])
        if overlap_w <= 0 or overlap_h <= 0:
            return edge, boxes
        next_edge = edge - max(1, int(overlap_w))
        if next_edge < min_edge_px or next_edge <= 0:
            return None, None
        edge = next_edge
    return None, None


def _make_region(
    *,
    position,
    center_x,
    center_y,
    quality_ok,
    reject_reasons,
    diagnostics,
    edge,
    width,
    height,
    policy,
):
    bbox, shift = _clip_square_box(center_x, center_y, edge, width, height)
    reasons = list(reject_reasons)
    if shift > policy["max_crop_boundary_shift_fraction"]:
        reasons.append("requested_crop_exceeds_source_field")
    accepted = bool(quality_ok) and not reasons
    diagnostic = dict(diagnostics)
    diagnostic["crop_boundary_shift_fraction"] = shift
    diagnostic["proposed_bbox_xyxy_exclusive"] = list(bbox)
    diagnostic["scores_are_uncalibrated_diagnostics"] = True
    return {
        "image_position": position,
        "clinical_side": "UNKNOWN",
        "bbox_xyxy_exclusive": list(bbox) if accepted else None,
        "proposed_bbox_xyxy_exclusive": list(bbox),
        "center_xy_native": [float(center_x), float(center_y)],
        "quality_status": "candidate" if accepted else "rejected",
        "reject_reasons": reasons,
        "diagnostics": diagnostic,
    }


def _column_profile(work, mask, policy):
    height, width = work.shape
    y0 = int(round(height * policy["column_band_y_fraction"][0]))
    y1 = int(round(height * policy["column_band_y_fraction"][1]))
    band = np.where(mask[y0:y1], work[y0:y1], np.nan)
    counts = np.sum(mask[y0:y1], axis=0)
    sums = np.nansum(band, axis=0)
    column = np.divide(sums, counts, out=np.zeros(width, dtype=np.float64), where=counts > 0)
    sigma = max(1.0, width * policy["column_smoothing_fraction"])
    return _smooth_1d(column, sigma)


def _prominent_peaks(column, policy):
    width = len(column)
    margin = int(round(width * policy["auto_margin_x_fraction"]))
    begin = max(1, margin)
    end = max(begin + 1, width - margin - 1)
    baseline = float(np.median(column))
    dynamic = max(float(column.max()) - baseline, 1e-9)
    min_prominence = max(
        policy["auto_peak_min_absolute_prominence"],
        policy["auto_peak_min_prominence_fraction"] * dynamic,
    )
    min_width = max(3, int(round(width * policy["auto_peak_min_width_fraction"])))
    candidates = []
    for index in range(begin, end):
        value = float(column[index])
        if not (value >= column[index - 1] and value > column[index + 1]):
            continue
        prominence = value - baseline
        if prominence < min_prominence:
            continue
        threshold = value - max(
            policy["peak_support_min_absolute_drop"],
            policy["peak_support_drop_fraction"] * prominence,
        )
        left = index
        while left > 0 and column[left - 1] >= threshold:
            left -= 1
        right = index
        while right + 1 < width and column[right + 1] >= threshold:
            right += 1
        support_width = right - left + 1
        if support_width < min_width:
            continue
        candidates.append(
            {
                "index": int(index),
                "value_uncalibrated": value,
                "prominence_uncalibrated": float(prominence),
                "support_width_fraction": float(support_width / width),
            }
        )
    candidates.sort(key=lambda item: item["value_uncalibrated"], reverse=True)
    min_separation = max(1, int(round(width * policy["auto_min_separation_fraction"])))
    peaks = []
    for candidate in candidates:
        if all(
            abs(candidate["index"] - accepted["index"]) >= min_separation
            for accepted in peaks
        ):
            peaks.append(candidate)
    peaks.sort(key=lambda item: item["index"])
    return peaks


def _auto_layout(column, policy):
    width = len(column)
    peaks = _prominent_peaks(column, policy)
    center_begin = int(round(width * policy["auto_central_band_x_fraction"][0]))
    center_end = int(round(width * policy["auto_central_band_x_fraction"][1]))
    center_peak = (
        float(column[center_begin:center_end].max()) if center_end > center_begin else 0.0
    )
    baseline = float(np.median(column))
    diagnostics = {
        "peak_count": int(len(peaks)),
        "tissue_peaks": [
            {
                "x_fraction": float(peak["index"] / max(width, 1)),
                "value_uncalibrated": float(peak["value_uncalibrated"]),
                "support_width_fraction": float(peak["support_width_fraction"]),
            }
            for peak in peaks
        ],
        "central_trough_ratio_uncalibrated": None,
        "scores_are_uncalibrated_diagnostics": True,
    }
    if not peaks:
        return None, "no_supported_tissue_peak", diagnostics
    if len(peaks) == 1:
        return "single", "auto_single_dominant_tissue_peak", diagnostics
    if len(peaks) > 2:
        return None, "ambiguous_multiple_tissue_peaks", diagnostics
    first, second = peaks
    separation = (second["index"] - first["index"]) / float(max(width, 1))
    side_floor = min(first["value_uncalibrated"], second["value_uncalibrated"]) - baseline
    trough_ratio = (center_peak - baseline) / max(side_floor, 1e-9)
    diagnostics["peak_separation_fraction"] = float(separation)
    diagnostics["central_trough_ratio_uncalibrated"] = float(trough_ratio)
    on_opposite_sides = first["index"] < center_begin and second["index"] >= center_end
    if (
        on_opposite_sides
        and separation >= policy["auto_min_separation_fraction"]
        and trough_ratio <= policy["auto_central_trough_max_ratio"]
    ):
        return "bilateral", "auto_two_separated_tissue_peaks_with_central_trough", diagnostics
    return None, "ambiguous_two_tissue_peaks_without_clear_central_trough", diagnostics


def _locate_single(work, mask, policy):
    height, width = work.shape
    column = _column_profile(work, mask, policy)
    begin = int(round(width * policy["single_search_x_fraction"][0]))
    end = int(round(width * policy["single_search_x_fraction"][1]))
    segment = column[begin:end]
    if segment.size == 0:
        return [], "no_supported_tissue_peak"
    center_x = begin + int(np.argmax(segment))
    evidence = _side_evidence(work, center_x, policy)
    low_support = (
        evidence["tissue_std_uncalibrated"] < policy["min_tissue_std_uncalibrated"]
        or evidence["gap_contrast_uncalibrated"] < policy["min_gap_contrast_uncalibrated"]
    )
    baseline = float(np.median(column))
    peak_prominence = float(segment.max()) - baseline
    no_peak = peak_prominence < policy["auto_peak_min_absolute_prominence"]
    reasons = []
    if low_support:
        reasons.append("insufficient_single_tissue_or_gap_contrast")
    if no_peak:
        reasons.append("no_supported_tissue_peak")
    edge = min(round(width * policy["single_box_edge_fraction"]), width, height)
    diagnostics = {
        "tissue_std_uncalibrated": evidence["tissue_std_uncalibrated"],
        "gap_contrast_uncalibrated": evidence["gap_contrast_uncalibrated"],
        "tissue_peak_prominence_uncalibrated": peak_prominence,
        "scores_are_uncalibrated_diagnostics": True,
    }
    region = _make_region(
        position="single",
        center_x=float(center_x),
        center_y=float(evidence["independent_joint_y_work"]),
        quality_ok=not reasons,
        reject_reasons=reasons,
        diagnostics=diagnostics,
        edge=edge,
        width=width,
        height=height,
        policy=policy,
    )
    if no_peak and low_support:
        return [], "no_tissue_evidence"
    return [region], "single_region_candidate" if not reasons else "single_region_rejected"


def _locate_bilateral(work, mask, policy):
    height, width = work.shape
    column = _column_profile(work, mask, policy)
    searches = policy["bilateral_search_x_fractions"]
    centers = []
    evidences = []
    for begin_fraction, end_fraction in searches:
        begin = int(round(width * begin_fraction))
        end = int(round(width * end_fraction))
        segment = column[begin:end]
        if segment.size == 0:
            center_x = (begin + end) // 2
        else:
            center_x = begin + int(np.argmax(segment))
        centers.append(center_x)
        evidences.append(_side_evidence(work, center_x, policy))
    local_plausible = []
    local_reasons = []
    for evidence in evidences:
        low = (
            evidence["tissue_std_uncalibrated"] < policy["min_tissue_std_uncalibrated"]
            or evidence["gap_contrast_uncalibrated"] < policy["min_gap_contrast_uncalibrated"]
        )
        local_plausible.append(not low)
        local_reasons.append(["insufficient_bilateral_tissue_or_joint_contrast"] if low else [])
    both_plausible = all(local_plausible)
    independent = [evidence["independent_joint_y_work"] for evidence in evidences]
    disagreement = abs(independent[0] - independent[1]) / float(max(height, 1))

    profiles = []
    for evidence in evidences:
        response = evidence["row_response"]
        peak = float(response.max())
        profiles.append(response / max(peak, 1e-12))
    window = max(
        3, int(round(height * policy["shared_profile_window_fraction"])) | 1
    )
    common = np.sqrt(_max_filter_1d(profiles[0], window) * _max_filter_1d(profiles[1], window))
    prior_center, prior_sigma = policy["joint_y_prior"]
    prior = np.exp(
        -0.5
        * ((np.arange(height, dtype=np.float64) - height * prior_center)
           / max(1.0, height * prior_sigma))
        ** 2
    )
    common *= prior
    y0 = int(round(height * policy["joint_band_y_fraction"][0]))
    y1 = int(round(height * policy["joint_band_y_fraction"][1]))
    common_patch = common[y0:y1]
    shared_y = y0 + int(np.argmax(common_patch)) if common_patch.size else height // 2
    shared_support = float(common_patch.max()) if common_patch.size else 0.0
    radius = max(2, int(round(height * policy["shared_joint_radius_fraction"])))
    low_y = max(y0, shared_y - radius)
    high_y = min(y1, shared_y + radius + 1)
    joint = []
    for profile in profiles:
        patch = profile[low_y:high_y]
        joint.append(low_y + int(np.argmax(patch)) if patch.size else shared_y)

    centers_ordered = bool(centers[0] < centers[1])
    centers_within_ranges = True
    for index, (begin_fraction, end_fraction) in enumerate(searches):
        begin = int(round(width * begin_fraction))
        end = int(round(width * end_fraction))
        if not (begin <= centers[index] < end):
            centers_within_ranges = False
    centers_valid = bool(centers_ordered and centers_within_ranges)
    constraint_applied = bool(both_plausible and centers_valid)
    edge_plan = _bilateral_edge_plan(
        centers,
        width,
        height,
        policy,
        apply_spacing=constraint_applied,
    )
    edge_plan["constraint_applied"] = constraint_applied
    edge_plan["centers_ordered_image_left_to_right"] = centers_ordered
    edge_plan["centers_within_search_ranges"] = bool(centers_within_ranges)
    edge_plan["stage"] = "work"

    pair_reasons = []
    if both_plausible and shared_support < policy["min_shared_support_uncalibrated"]:
        pair_reasons.append("no_shared_bilateral_joint_height_support")
    if both_plausible and disagreement > policy["max_independent_y_disagreement_fraction"]:
        pair_reasons.append("independent_joint_heights_disagree")
    if not centers_valid:
        pair_reasons.append("bilateral_centers_invalid_order_or_search_range")
    elif constraint_applied and edge_plan["below_minimum_edge"]:
        pair_reasons.append("bilateral_centers_too_close_for_context")
    edge = edge_plan["selected_edge_px"]
    if edge is None:
        edge = int(
            max(
                1,
                min(
                    round(width * policy["bilateral_box_edge_fraction"]),
                    width,
                    height,
                ),
            )
        )

    regions = []
    any_plausible = any(local_plausible)
    for index, position in enumerate(("image_left", "image_right")):
        evidence = evidences[index]
        reasons = list(local_reasons[index]) + list(pair_reasons)
        diagnostics = {
            "tissue_std_uncalibrated": evidence["tissue_std_uncalibrated"],
            "gap_contrast_uncalibrated": evidence["gap_contrast_uncalibrated"],
            "independent_joint_y_fraction": independent[index] / float(max(height, 1)),
            "paired_joint_y_fraction": joint[index] / float(max(height, 1)),
            "independent_y_disagreement_fraction": float(disagreement),
            "shared_joint_support_uncalibrated": shared_support,
            "bilateral_edge_selection_work": dict(edge_plan),
            "scores_are_uncalibrated_diagnostics": True,
        }
        region = _make_region(
            position=position,
            center_x=float(centers[index]),
            center_y=float(joint[index]),
            quality_ok=not reasons,
            reject_reasons=reasons,
            diagnostics=diagnostics,
            edge=edge,
            width=width,
            height=height,
            policy=policy,
        )
        if not local_plausible[index] and not any_plausible:
            continue
        regions.append(region)
    if not regions:
        return [], "no_tissue_evidence"
    detected = sum(1 for region in regions if region["quality_status"] == "candidate")
    if detected == 2:
        status = "bilateral_pair_candidates"
    elif detected == 1:
        status = "single_sided_candidate_without_forcing_second_knee"
    else:
        status = "bilateral_candidates_rejected"
    return regions, status


def _update_native_box(region, edge, width, height, policy):
    center_x, center_y = region["center_xy_native"]
    clipped, shift = _clip_square_box(center_x, center_y, edge, width, height)
    region["proposed_bbox_xyxy_exclusive"] = list(clipped)
    region["diagnostics"]["proposed_bbox_xyxy_exclusive"] = list(clipped)
    region["diagnostics"]["crop_boundary_shift_fraction"] = float(shift)
    if shift > policy["max_crop_boundary_shift_fraction"]:
        if "requested_crop_exceeds_source_field" not in region["reject_reasons"]:
            region["reject_reasons"].append("requested_crop_exceeds_source_field")
    if region["quality_status"] == "candidate" and not region["reject_reasons"]:
        region["bbox_xyxy_exclusive"] = list(clipped)
    else:
        region["quality_status"] = "rejected"
        region["bbox_xyxy_exclusive"] = None


def _finalize_native_geometry(
    regions, chosen_layout, work_width, work_height, width, height, policy
):
    for region in regions:
        region["center_xy_native"] = [
            float(region["center_xy_native"][0] / work_width * width),
            float(region["center_xy_native"][1] / work_height * height),
        ]

    if chosen_layout == "single" or len(regions) < 2:
        edge_fraction = (
            policy["single_box_edge_fraction"]
            if chosen_layout == "single"
            else policy["bilateral_box_edge_fraction"]
        )
        edge = int(
            max(1, min(round(width * edge_fraction), width, height))
        )
        for region in regions:
            _update_native_box(region, edge, width, height, policy)
    else:
        work_blocks = [
            region["diagnostics"].get("bilateral_edge_selection_work", {})
            for region in regions
        ]
        constraint_applied = all(
            bool(block.get("constraint_applied")) for block in work_blocks
        )
        carried_values = [block.get("selected_edge_fraction") for block in work_blocks]
        carried_fraction = None
        if constraint_applied and all(value is not None for value in carried_values):
            carried_fraction = min(float(value) for value in carried_values)
        centers_x = [region["center_xy_native"][0] for region in regions]
        centers_y = [region["center_xy_native"][1] for region in regions]
        ordered = bool(centers_x[0] < centers_x[1])
        apply_spacing = bool(constraint_applied and ordered)
        plan = _bilateral_edge_plan(
            centers_x,
            width,
            height,
            policy,
            carried_fraction=carried_fraction,
            apply_spacing=apply_spacing,
        )
        plan["constraint_applied"] = apply_spacing
        plan["centers_ordered_image_left_to_right"] = ordered
        plan["carried_work_edge_fraction"] = carried_fraction
        plan["stage"] = "native"

        reasons = []
        if constraint_applied and not ordered:
            reasons.append("bilateral_centers_invalid_order_or_search_range")
        elif apply_spacing and plan["below_minimum_edge"]:
            reasons.append("bilateral_centers_too_close_for_context")

        fit_edge = plan["selected_edge_px"]
        if fit_edge is None:
            fit_edge = int(
                max(
                    1,
                    min(
                        round(width * policy["bilateral_box_edge_fraction"]),
                        width,
                        height,
                    ),
                )
            )
        if apply_spacing and not reasons:
            pre_round_edge = int(fit_edge)
            fit_edge, fit_boxes = _fit_bilateral_boxes(
                list(zip(centers_x, centers_y)),
                fit_edge,
                width,
                height,
                plan["minimum_edge_px"],
            )
            if fit_boxes is None:
                plan["rounding_fit_failed"] = True
                plan["selected_edge_px"] = None
                plan["selected_edge_fraction"] = None
                reasons.append("bilateral_centers_too_close_for_context")
            else:
                plan["selected_edge_px"] = int(fit_edge)
                plan["selected_edge_fraction"] = float(fit_edge / float(max(width, 1)))
                plan["rounding_adjusted"] = bool(fit_edge != pre_round_edge)
                plan["pre_rounding_edge_px"] = pre_round_edge

        if reasons:
            plan["selected_edge_px"] = None
            plan["selected_edge_fraction"] = None

        for region in regions:
            region["diagnostics"]["bilateral_edge_selection_native"] = dict(plan)
        if reasons:
            for region in regions:
                for reason in reasons:
                    if reason not in region["reject_reasons"]:
                        region["reject_reasons"].append(reason)
                _update_native_box(
                    region,
                    int(
                        max(
                            1,
                            min(
                                round(width * policy["bilateral_box_edge_fraction"]),
                                width,
                                height,
                            ),
                        )
                    ),
                    width,
                    height,
                    policy,
                )
        else:
            for region in regions:
                _update_native_box(region, int(fit_edge), width, height, policy)

    for region in regions:
        for key in (
            "tissue_std_uncalibrated",
            "gap_contrast_uncalibrated",
            "shared_joint_support_uncalibrated",
        ):
            value = region["diagnostics"].get(key)
            if value is not None:
                region["diagnostics"][key] = float(value)


def locate_knees(gray, valid=None, layout="auto"):
    if layout not in ("auto", "single", "bilateral"):
        raise ValueError("layout must be 'auto', 'single' or 'bilateral'")
    source, mask = _validate_gray_and_mask(
        gray, valid, min_short_side=POLICY["min_short_side_native"]
    )
    work, work_mask = _resize_work(source, mask, POLICY["max_work_side"])
    height, width = source.shape
    work_height, work_width = work.shape

    auto_diagnostics = {}
    chosen_layout = None
    top_reasons = []
    if layout == "auto":
        column = _column_profile(work, work_mask, POLICY)
        chosen_layout, auto_reason, auto_diagnostics = _auto_layout(column, POLICY)
        top_reasons.append(auto_reason)
        if chosen_layout is None:
            top_reasons.append("no_qualified_region")
            return {
                "method": METHOD,
                "version": VERSION,
                "layout_requested": layout,
                "layout_selected": None,
                "detected_count": 0,
                "regions": [],
                "reasons": top_reasons,
                "diagnostics": auto_diagnostics,
                "localization_validated": False,
                "training_ready": False,
                "scores_are_uncalibrated_diagnostics": True,
                "clinical_side": "UNKNOWN",
            }
    else:
        chosen_layout = layout

    if chosen_layout == "single":
        regions, status = _locate_single(work, work_mask, POLICY)
    else:
        regions, status = _locate_bilateral(work, work_mask, POLICY)

    _finalize_native_geometry(
        regions,
        chosen_layout,
        work_width,
        work_height,
        width,
        height,
        POLICY,
    )

    detected = sum(1 for region in regions if region["quality_status"] == "candidate")
    if detected == 0:
        top_reasons.append("no_qualified_region")
    return {
        "method": METHOD,
        "version": VERSION,
        "layout_requested": layout,
        "layout_selected": chosen_layout,
        "detected_count": int(detected),
        "regions": regions,
        "reasons": top_reasons,
        "diagnostics": {
            **auto_diagnostics,
            "status": status,
            "work_size_hw": [int(work_height), int(work_width)],
            "native_size_hw": [int(height), int(width)],
        },
        "localization_validated": False,
        "training_ready": False,
        "scores_are_uncalibrated_diagnostics": True,
        "clinical_side": "UNKNOWN",
    }


MODEL_IMAGE_SIZES = (224, 448)


def _validate_extraction_inputs(gray, valid):
    if valid is None:
        raise ValueError("extract_regions needs an explicit bool valid mask")
    return _validate_gray_and_mask(gray, valid, min_short_side=1)


def _validate_candidate_regions(regions, width, height):
    if not isinstance(regions, (list, tuple)):
        raise ValueError("regions must be a list of region dicts")
    if len(regions) < 1:
        raise ValueError("regions must not be empty")
    if len(regions) > 2:
        raise ValueError("only 1 or 2 candidate regions are accepted")
    boxes = []
    for index, region in enumerate(regions):
        if not isinstance(region, dict):
            raise ValueError(f"region[{index}] must be a dict")
        if region.get("quality_status") != "candidate":
            raise ValueError(f"region[{index}] is not a candidate; extraction refused")
        if region.get("reject_reasons"):
            raise ValueError(f"region[{index}] has reject_reasons; extraction refused")
        position = region.get("image_position")
        if position not in ("image_left", "image_right", "single"):
            raise ValueError(f"region[{index}] invalid image_position")
        bbox = region.get("bbox_xyxy_exclusive")
        if not isinstance(bbox, (list, tuple, np.ndarray)) or len(bbox) != 4:
            raise ValueError(f"region[{index}] bbox must be a 4-element sequence")
        values = []
        for value in bbox:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(f"region[{index}] bbox must be strict integers")
            values.append(int(value))
        x0, y0, x1, y1 = values
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"region[{index}] bbox must have positive area")
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            raise ValueError(f"region[{index}] bbox out of bounds; rejected, not clipped")
        boxes.append([x0, y0, x1, y1])
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            ax0, ay0, ax1, ay1 = boxes[first]
            bx0, by0, bx1, by1 = boxes[second]
            overlap_w = min(ax1, bx1) - max(ax0, bx0)
            overlap_h = min(ay1, by1) - max(ay0, by0)
            if overlap_w > 0 and overlap_h > 0:
                raise ValueError("candidate bboxes must not overlap")
    return boxes


def _resize_float_image(array, size):
    image = Image.fromarray(np.ascontiguousarray(array, dtype=np.float32), mode="F")
    return np.asarray(
        image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32
    )


def _resize_bool_mask(mask, size):
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 0


def _validate_crop_content(values, values_mask, description):
    valid_values = values[values_mask]
    if valid_values.size == 0:
        raise InvalidCropContentError(f"{description} has no valid pixels")
    if not np.isfinite(valid_values).all():
        raise InvalidCropContentError(f"{description} valid values contain nan/inf")
    if float(valid_values.std()) < MIN_CROP_VALID_STD_ENGINEERING:
        raise InvalidCropContentError(
            f"{description} valid values are degenerate (std below the engineering threshold "
            f"{MIN_CROP_VALID_STD_ENGINEERING})"
        )


def _letterbox_crop(crop, crop_mask, image_size):
    crop_height, crop_width = crop.shape
    scale = min(image_size / float(crop_width), image_size / float(crop_height))
    resized_width = max(1, min(image_size, int(round(crop_width * scale))))
    resized_height = max(1, min(image_size, int(round(crop_height * scale))))
    offset_x = (image_size - resized_width) // 2
    offset_y = (image_size - resized_height) // 2
    zeroed = np.where(crop_mask, crop, 0.0).astype(np.float32, copy=False)
    resized = _resize_float_image(zeroed, (resized_width, resized_height))
    valid = _resize_bool_mask(crop_mask, (resized_width, resized_height))
    image = np.zeros((image_size, image_size), dtype=np.float32)
    valid_mask = np.zeros((image_size, image_size), dtype=bool)
    image[
        offset_y : offset_y + resized_height, offset_x : offset_x + resized_width
    ] = resized
    valid_mask[
        offset_y : offset_y + resized_height, offset_x : offset_x + resized_width
    ] = valid
    image = np.where(valid_mask, image, 0.0).astype(np.float32, copy=False)
    transform = {
        "image_size": int(image_size),
        "source_bbox_xyxy_exclusive": [int(v) for v in (0, 0, crop_width, crop_height)],
        "source_crop_size_wh": [int(crop_width), int(crop_height)],
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
        "resized_size_wh": [int(resized_width), int(resized_height)],
        "offset_x": int(offset_x),
        "offset_y": int(offset_y),
        "offset_xy": [int(offset_x), int(offset_y)],
        "scale_x": float(resized_width / crop_width),
        "scale_y": float(resized_height / crop_height),
        "letterbox_scale": float(scale),
        "padding_value": 0.0,
    }
    return image, valid_mask, transform


def native_xy_to_model_xy(x, y, transform):
    bbox = transform["source_bbox_xyxy_exclusive"]
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    model_x = (x_values - bbox[0]) * transform["scale_x"] + transform["offset_x"]
    model_y = (y_values - bbox[1]) * transform["scale_y"] + transform["offset_y"]
    return model_x, model_y


def model_xy_to_native_xy(x, y, transform):
    bbox = transform["source_bbox_xyxy_exclusive"]
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    native_x = (x_values - transform["offset_x"]) / transform["scale_x"] + bbox[0]
    native_y = (y_values - transform["offset_y"]) / transform["scale_y"] + bbox[1]
    return native_x, native_y


def extract_regions(gray, valid, regions, image_size=448):
    if isinstance(image_size, (bool, np.bool_)) or not isinstance(
        image_size, (int, np.integer)
    ):
        raise ValueError("image_size must be the integer 224 or 448")
    if int(image_size) not in MODEL_IMAGE_SIZES:
        raise ValueError("image_size must be 224 or 448")
    image_size = int(image_size)
    source, mask = _validate_extraction_inputs(gray, valid)
    height, width = source.shape
    boxes = _validate_candidate_regions(regions, width, height)
    outputs = []
    for index, (region, bbox) in enumerate(zip(regions, boxes)):
        x0, y0, x1, y1 = bbox
        crop = source[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        _validate_crop_content(crop, crop_mask, f"region[{index}] native crop")
        image, valid_mask, transform = _letterbox_crop(crop, crop_mask, image_size)
        _validate_crop_content(image, valid_mask, f"region[{index}] crop after letterbox")
        transform["source_bbox_xyxy_exclusive"] = [int(x0), int(y0), int(x1), int(y1)]
        transform["source_crop_size_wh"] = [int(x1 - x0), int(y1 - y0)]
        outputs.append(
            {
                "region_index": int(index),
                "image_position": region["image_position"],
                "clinical_side": "UNKNOWN",
                "bbox_xyxy_exclusive": [int(x0), int(y0), int(x1), int(y1)],
                "image": image[np.newaxis, :, :].astype(np.float32, copy=False),
                "valid_mask": valid_mask[np.newaxis, :, :].astype(bool, copy=False),
                "transform": transform,
                "scores_are_uncalibrated_diagnostics": True,
            }
        )
    return outputs
