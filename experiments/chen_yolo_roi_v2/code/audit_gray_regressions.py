import json
import numpy as np
from pydicom.pixels import apply_modality_lut
import pipeline as p
from drai.data.dicom_gray import read_pa_archive, presentation_values


def main():
    cases = json.loads((p.PILOT / 'outputs/selection_private.json').read_text())['cases']
    results = []
    for case_id in ['C10', 'C31', 'C38', 'C39', 'C44']:
        case = next(r for r in cases if r['case_id'] == case_id)
        ds, pixels, receipt = read_pa_archive(p.NAS / case['input_archive_relpath'],
                                             expected_bytes=int(case['archive_bytes']), source_root=p.NAS)
        if receipt['archive_sha256'] != case['archive_sha256'] or receipt['native_pixel_sha256'] != case['native_pixel_sha256']:
            raise RuntimeError('Input source changed')
        values, valid, presentation = presentation_values(pixels, ds)
        modality_values = apply_modality_lut(pixels, ds)
        if str(getattr(ds, 'VOILUTFunction', 'LINEAR')) != 'LINEAR' or getattr(ds, 'VOILUTSequence', []):
            raise RuntimeError('Audit boundary formula requires simple LINEAR window')
        center, width = float(ds.WindowCenter), float(ds.WindowWidth)
        low, high = center - .5 - (width - 1)/2, center - .5 + (width - 1)/2
        results.append({
            'case_id': case_id, 'source_hash_verified': True, 'presentation': presentation,
            'raw_percentiles_005_50_995': np.percentile(pixels[valid], [.5, 50, 99.5]).tolist(),
            'modality_percentiles_005_50_995': np.percentile(modality_values[valid], [.5, 50, 99.5]).tolist(),
            'window_bounds': [low, high], 'below_window_fraction': float((modality_values[valid] <= low).mean()),
            'above_window_fraction': float((modality_values[valid] > high).mean()),
            'presentation_top_fraction': float((values[valid] == values[valid].max()).mean()),
        })
    report = {'cases': results, 'gray_policy_changed': False,
              'conclusion': 'C10/C38/C39 contain values beyond the applied first window; display saturation is not by itself proof of acquisition overexposure. Any alternative gray policy requires a separate version and review.'}
    p.save(p.OUT / 'gray_window_audit.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
