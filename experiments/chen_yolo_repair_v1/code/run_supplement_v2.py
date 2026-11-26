import run_supplement as base
geometry_reasons=base.reasons


def reasons(result):
    result_reasons=geometry_reasons(result)
    pres=result.get('presentation',{})
    if pres.get('photometric_interpretation')=='MONOCHROME2' and pres.get('presentation_inverse_applied'):
        result_reasons.append('mono2_inverse_presentation_review')
    return result_reasons


if __name__=='__main__':
    base.reasons=reasons
    base.main()
