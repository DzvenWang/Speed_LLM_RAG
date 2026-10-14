import torch
from fare.memory import EpisodeMemory

def test_strict_future_end_and_validity(fixture_data):
    _,mem,b=fixture_data
    legal=mem.valid_for(torch.tensor([15,100]),torch.device('cpu'))
    # first episode starts 0, H 12, P 12 => future end at 23
    assert not legal[0].any()
    assert legal[1].all()

def test_episode_key_does_not_depend_on_future(fixture_data):
    _,mem,b=fixture_data
    orig=mem.node_key_cpu.clone()
    mem.future.add_(10)
    assert torch.equal(mem.node_key_cpu,orig)

def test_support_direction_provenance(fixture_data):
    model,_,_=fixture_data
    assert model.graph.metadata_source=='undirected_topology'
