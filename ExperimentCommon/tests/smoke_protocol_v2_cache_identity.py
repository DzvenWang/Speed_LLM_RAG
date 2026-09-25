from pathlib import Path
import json
import tempfile

from experiment_common.cache import cache_dir, evaluation_access_granted, require_evaluation_cache_access


def cfg(root: Path, name='protocol_s40', severity=0.4, allowed=('development','formal','evaluation')):
    return {
        'project': {
            'cache_root': str(root/'ExperimentCache'),
            'protocol_artifact_root': str(root/'Protocol_artifacts'),
        },
        'experiment': {'subgraph':'TraffiDent_Net_06','severity':severity},
        'protocol': {
            'name':name,'version':'2.0.1',
            'paired_group_id':'TraffiDent_Net_06_multiseverity_v1',
            'allowed_stages':list(allowed),
        },
    }


def write_stage(c, root: Path, stage: str, h: str):
    sd = Path(c['project']['protocol_artifact_root'])/c['experiment']['subgraph']/c['protocol']['name']/stage
    sd.mkdir(parents=True, exist_ok=True)
    m = {
        'protocol_name':c['protocol']['name'],
        'protocol_version':c['protocol']['version'],
        'severity':c['experiment']['severity'],
        'stage_hash':h,
    }
    (sd/'stage_manifest.json').write_text(json.dumps(m), encoding='utf-8')
    (sd/'FROZEN.json').write_text(json.dumps({'stage_hash':h}), encoding='utf-8')


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    c40 = cfg(root)
    h40 = 'a'*64
    write_stage(c40, root, 'formal', h40)
    p40 = cache_dir(c40, 'formal')
    assert p40.parts[-3:] == ('s40','formal',h40)
    assert c40['protocol']['name'] in p40.parts

    c20 = cfg(root, name='protocol_s20', severity=0.2, allowed=('evaluation',))
    h20 = 'b'*64
    write_stage(c20, root, 'evaluation', h20)
    p20 = cache_dir(c20, 'evaluation')
    assert p20 != p40
    assert 's20' in p20.parts and 'protocol_s20' in p20.parts
    assert not evaluation_access_granted(c20)
    try:
        require_evaluation_cache_access(c20, 'evaluation')
        raise AssertionError('locked evaluation guard did not raise')
    except PermissionError:
        pass

    gate = root/'Protocol_artifacts'/'TraffiDent_Net_06'/'_paired'/'TraffiDent_Net_06_multiseverity_v1'/'EVALUATION_ACCESS_GRANTED.json'
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text('{}', encoding='utf-8')
    assert evaluation_access_granted(c20)
    require_evaluation_cache_access(c20, 'evaluation')

print('SMOKE PROTOCOL V2 CACHE IDENTITY PASS')
