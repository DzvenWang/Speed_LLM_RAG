from collections import deque

def main():
    history=12; horizon=12; starts=[0,1,5,12,20,30]
    pending=deque(); matured=[]
    for s in starts:
        origin=s+history
        while pending and pending[0][0] <= origin:
            matured.append(pending.popleft()[1])
        for end,old in pending:
            assert end > origin
        pending.append((s+history+horizon,s))
    assert 0 in matured
    print('CAUSALITY_SMOKE PASS',matured)
if __name__=='__main__': main()
