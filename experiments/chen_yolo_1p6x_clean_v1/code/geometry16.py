import json
import math
import sys
from pathlib import Path

PROJECT = Path('/path/to/drai')
sys.path.insert(0, str(PROJECT / 'experiments/chen_yolo_expand2x_v1/code'))
import preview_direct2x as previous

e, base, OUT = previous.e, previous.base, previous.OUT


def direct_box(item, native_hw):
    source_path = Path(item['parent_candidate_result'])
    assert e.p.sha(source_path) == item['parent_candidate_result_sha256']
    source = json.loads(source_path.read_text())
    if item['candidate_family'] == 'base':
        entry = next(x for x in source['yolo'] if x['image_position'] == item['image_position'] and x['status'] == 'saved')
    else:
        entry = e.saved(source, item['variant'], item['image_position'])
    assert entry['bbox'] == item['bbox']
    h, w = native_hw
    x0, y0, x1, y1 = map(float, entry['unclipped_bbox'])
    origin_y = 0
    if item['variant'].startswith('lower_'):
        origin = source[item['variant'] + '_detection']['source_field_origin_xy']
        assert origin == [0, round(h * .45)] and source['native_hw'] == list(native_hw)
        origin_y = origin[1]

    def bound(b, lower=0):
        return [max(0, min(w, math.floor(b[0]))), max(lower, min(h, math.floor(b[1]))),
                max(0, min(w, math.ceil(b[2]))), max(lower, min(h, math.ceil(b[3])))]

    assert bound([x0, y0, x1, y1], origin_y) == item['bbox']
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dw, dh = (x1 - x0) / 1.3, (y1 - y0) / 1.3
    requested = [cx - .8 * dw, cy - .8 * dh, cx + .8 * dw, cy + .8 * dh]
    box = bound(requested)
    old = item['bbox']
    assert box[0] <= old[0] and box[1] <= old[1] and box[2] >= old[2] and box[3] >= old[3]
    clipped = requested[0] < 0 or requested[1] < 0 or requested[2] > w or requested[3] > h
    return box, clipped, {'unclipped_1p3_bbox': entry['unclipped_bbox'], 'requested_1p6_bbox': requested,
                         'detector_native_size_wh': [dw, dh], 'parent_result_sha256': e.p.sha(source_path),
                         'old_detection_window_origin_xy': [0, origin_y], 'old_window_bbox_reproduced': True,
                         'new_crop_uses_full_native_field': True}
