from collections import deque

def main():
    # H=P=12. A query starting at 0 matures at 24 and must not be available at origin 23.
    pending=deque([(24,0),(30,6)])
    mature=[]; origin=23
    while pending and pending[0][0] <= origin: mature.append(pending.popleft()[1])
    assert mature==[]
    origin=24
    while pending and pending[0][0] <= origin: mature.append(pending.popleft()[1])
    assert mature==[0]
    print('SMOKE CAUSALITY PASS')
if __name__=='__main__': main()
