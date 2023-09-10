import mdtraj as mdj

def build_mdtraj_top(N):
    new_top = mdj.Topology()
    chain = new_top.add_chain()
    
    for i in range(N):
        res = new_top.add_residue('gho',chain)
        _ = new_top.add_atom(f'Ar{i}',mdj.element.Element.getBySymbol('Ar'),res)

    return new_top

def extend_mdtraj_top(top,N):
    chain = top.add_chain()
    
    for i in range(N):
        res = top.add_residue('gho',chain)
        _ = top.add_atom(f'Ar{i}',mdj.element.Element.getBySymbol('Ar'),res)

    return top
