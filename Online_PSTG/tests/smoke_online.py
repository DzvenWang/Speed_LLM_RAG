#!/usr/bin/env python3
from collections import deque


def mature_sequence(starts,H=12,P=12):
    pending=deque(); out=[]
    for s in starts:
        origin=s+H; matured=[]
        while pending and pending[0][0] <= origin:
            matured.append(pending.popleft()[1])
        out.append((s,matured.copy()))
        pending.append((s+H+P,s))
    return out


def main():
    z=mature_sequence([0,12,24,36])
    assert z[0][1]==[]
    assert z[1][1]==[0], z
    assert z[2][1]==[12], z
    # With 5-min stride windows, a 12-step horizon should mature only after 12 raw steps.
    z2=mature_sequence([0,5,10,15,20,25])
    assert z2[1][1]==[] and z2[2][1]==[]
    assert z2[3][1]==[0], z2
    print('SMOKE PASS causal maturity')
if __name__=='__main__': main()
