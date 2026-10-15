"""Research-semantic checks, not only tensor shape smoke tests."""
import torch
import numpy as np
import pytest
from torch.utils.data import Dataset
from fare.traffic import TrafficGraph
from fare.teacher import generate_teacher,replay_loss


def test_directed_path_semantics_and_evidence_demand():
    # A -> B -> C; upstream of B is A, downstream is C.
    adj=torch.tensor([[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]])
    up=torch.tensor([[-1], [0], [1]])
    down=torch.tensor([[1], [2], [-1]])
    valid=torch.ones_like(up,dtype=torch.float)
    graph=TrafficGraph(adj,{'upstream_ids':up,'downstream_ids':down,
                            'upstream_weights':valid,'downstream_weights':valid},
                       region_size=2,max_hops=2,path_sources=4)
    path_1=graph.path_nodes[1][graph.path_valid[1]].tolist()
    types_1=graph.path_type[1][graph.path_valid[1]].tolist()
    assert (0,0) in zip(path_1,types_1)
    assert (2,1) in zip(path_1,types_1)
    # graph buffers are deterministic and never invent a physical distance.
    assert graph.metadata_source=='directional_neighbors'


def test_evidence_types_do_not_read_opposite_value(fixture_data):
    model,mem,batch=fixture_data
    model.eval()
    with torch.no_grad():
        full=model(batch,mem)
        short=full['candidate_pool'];support=full['support'];base=full['base_prediction']
        original=model.transfer(short,mem,model.graph,support,base)
        old=mem.residual.clone()
        mem.residual+=500
        changed=model.transfer(short,mem,model.graph,support,base)
        mem.residual=old
    # Evolution value must NOT change when only historical residual changes.
    assert torch.allclose(original['deltas'][:,:,:,1],changed['deltas'][:,:,:,1],atol=1e-5)


def test_set_candidate_order_equivariance(fixture_data):
    model,mem,batch=fixture_data
    model.eval()
    with torch.no_grad():
        full=model(batch,mem)
        short=full['candidate_pool'];prop=full['proposals'];support=full['support']
        p=torch.randperm(short['ids'].shape[2]);inv=torch.argsort(p)
        shuffled={k:(v[:,:,p] if torch.is_tensor(v) and v.ndim>=3 and v.shape[2]==len(p) else v)
                  for k,v in short.items()}
        proposal={k:(v[:,:,p] if torch.is_tensor(v) and v.ndim>=3 and v.shape[2]==len(p) else v)
                  for k,v in prop.items()}
        direct=model.utility(short,prop,support)
        reordered=model.utility(shuffled,proposal,support)
        # Exact scores should be permutation equivariant, not tied to a candidate index.
        assert torch.allclose(direct['all_utility'],reordered['all_utility'][:,:,inv],atol=1e-4)


def test_alpha_action_replay_teacher(fixture_data):
    model,mem,b=fixture_data
    class Mini(Dataset):
        def __len__(self):return 4
        def __getitem__(self,i):
            return {k:(v[i%2] if torch.is_tensor(v) and v.ndim else v)
                    if k!='row_id' else torch.tensor(i) for k,v in b.items()}
    records=generate_teacher(model,mem,Mini(),torch.device('cpu'),limit=4,batch_size=2,
                             nodes_per_query=3)
    assert len(records)==4
    r=records[next(iter(records))]
    assert 'short' in r and 'proposals' in r
    assert torch.isfinite(r['action_alpha']).all()
    loss,diag=replay_loss(model,records,torch.device('cpu'),max_replays=2)
    assert torch.isfinite(loss)
    assert diag['teacher_examples']>=1
    loss.backward()
    assert model.utility.embed[0].weight.grad is not None


def test_fixed_checkpoint_interventions_independent(fixture_data):
    model,mem,b=fixture_data
    model.eval()
    with torch.no_grad():
        full=model(b,mem)
        for mode in ('no_direction','no_propagation_time','no_capability_matching',
                     'no_set_interaction','no_swap_utility','no_residual_information',
                     'no_evolution_information','no_hidden_injection','no_risk_control'):
            alt=model.intervene_cached(b,mem,full,mode)
            assert torch.isfinite(alt['prediction']).all(),mode
        zero=model.intervene_cached(b,mem,full,'no_evidence')
        assert torch.allclose(zero['prediction'],zero['base_prediction'],atol=1e-6)


def test_train_only_support_sensitivity_teacher(fixture_data):
    from fare.support_teacher import build_support_teacher,support_auxiliary_loss
    model,mem,b=fixture_data
    class Small(Dataset):
        def __len__(self):return 4
        def __getitem__(self,i):
            return {k:(v[i%2] if torch.is_tensor(v) and v.ndim else v)
                    if k!='row_id' else torch.tensor(i) for k,v in b.items()}
    records=build_support_teacher(model,Small(),torch.device('cpu'),limit=4,batch_size=2)
    assert len(records)==4
    assert all(torch.isfinite(x['sensitivity']).all() for x in records.values())
    out=model(b,mem)
    aux,meta=support_auxiliary_loss(out,b,records,torch.device('cpu'))
    assert meta['support_teacher_examples']==2
    assert torch.isfinite(aux)
