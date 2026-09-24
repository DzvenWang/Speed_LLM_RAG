import numpy as np

from srd.fragment_completion import build_fragments, compose_fragment_selection


class ToyTrain:
    def __init__(self, rows): self.rows=rows
    def row(self,i): return self.rows[int(i)]


def cfg():
    return {
      'patch_transport': {'alignment_scope':'all','affine_min_points':2,'affine_scale_clip':[0.5,2.0],'eps':1e-6},
      'fragment_diagnostic': {
        'context_radius':1,'min_anchor_cells':1,'min_fragment_coverage':1.0,
        'local_level_weight':1.0,'local_delta_weight':0.0,'allow_oracle_abstain':True,
      }
    }


def make_query():
    # Last two steps are artificially missing for both nodes; earlier two are observed anchors.
    xfull=np.array([[1.,2.],[2.,3.],[10.,20.],[30.,40.]],dtype=np.float32)
    obs=np.array([[1,1],[1,1],[0,0],[0,0]],dtype=bool)
    xobs=np.where(obs,xfull,0.).astype(np.float32)
    return {
      'x_obs':xobs,'x_full':xfull,'history_observed':obs,
      'restorable':~obs,'full_valid':np.ones_like(obs,bool)
    }


def cand(a2,a3):
    x=np.array([[1.,2.],[2.,3.],a2,a3],dtype=np.float32)
    return {'x_full':x,'full_valid':np.ones_like(x,bool),'history_observed':np.ones_like(x,bool)}


def test_fragment_partitions_cover_target_exactly():
    q=make_query(); nodes=np.array([0,1])
    for geom,bs in [('whole_patch',12),('temporal',1),('temporal',2),('node_track',12),('node_block',1),('cell',1)]:
        fs=build_fragments(q,nodes,2,geom,bs)
        cover=np.zeros_like(q['restorable'],np.int16)
        for f in fs: cover += f.mask.astype(np.int16)
        target=q['restorable'] & np.array([[0,0],[0,0],[1,1],[1,1]],bool)
        assert np.array_equal(cover>0,target)
        assert cover.max() == 1


def test_temporal_fragment_oracle_composes_different_sources():
    q=make_query(); nodes=np.array([0,1])
    # Candidate 0 is exact at t=2 but poor at t=3; candidate 1 is the reverse.
    c0=cand([10.,20.],[90.,90.])
    c1=cand([80.,80.],[30.,40.])
    tr=ToyTrain([c0,c1])
    whole,_,_=compose_fragment_selection(q,tr,[0,1],nodes,2,'whole_patch',2,'raw_copy',cfg(),'oracle_completion')
    frag,st,details=compose_fragment_selection(q,tr,[0,1],nodes,2,'temporal',1,'raw_copy',cfg(),'oracle_completion')
    target=q['restorable']
    whole_err=np.abs(whole[target]-q['x_full'][target]).mean()
    frag_err=np.abs(frag[target]-q['x_full'][target]).mean()
    assert frag_err < whole_err
    assert np.allclose(frag[target],q['x_full'][target])
    assert st['unique_sources'] == 2
    assert sum(d.get('selected',False) for d in details) == 2


def test_local_fragment_selection_uses_observed_anchor_only():
    q=make_query(); nodes=np.array([0,1])
    # Candidate 0 matches the observed anchors; candidate 1 does not.
    c0=cand([10.,20.],[30.,40.])
    c1={'x_full':np.array([[20.,20.],[20.,20.],[10.,20.],[30.,40.]],np.float32),
        'full_valid':np.ones((4,2),bool),'history_observed':np.ones((4,2),bool)}
    tr=ToyTrain([c0,c1])
    out,st,details=compose_fragment_selection(q,tr,[0,1],nodes,2,'temporal',1,'raw_copy',cfg(),'local_observed')
    assert st['selected_fragments'] == 2
    assert all(d.get('candidate_id') == 0 for d in details if d.get('selected'))
    assert np.allclose(out[q['restorable']],q['x_full'][q['restorable']])
