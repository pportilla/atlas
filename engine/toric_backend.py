"""Toric resolution of a diagonal compact Newton face in dimension three.

The C++ kernel constructs a regular fan. Cone exponents give all pullbacks
and orbit strata exactly by integer pairings, without solving chart equations.
"""
import subprocess
from .polynomial import LimitError, integer_text, rational_text
from .resolve import BUILD
import json
from .binomial_kernel import _integer, character, minimum_faces, pairings, run, signed_powers


def monomial(names, powers):
    return '*'.join(v if e==1 else f'{v}^{integer_text(e)}' for v,e in zip(names,powers) if e) or '1'


def resolve_toric(p, variables, mode, compute=None):
    if len(variables)!=3:
        return None
    rows = run(f"diagonal 3 {len(p)}", *p)
    if rows == ["none"]:
        return None
    level, *w = map(_integer, rows[0].split())
    principal = [tuple(map(_integer, row.split())) for row in rows[1:4]]
    exps = [tuple(map(_integer, row.split())) for row in rows[4:]]
    binary=BUILD/'toricfan'
    if not binary.is_file():
        raise RuntimeError('Toric engine is not built. Run make.')
    proc=subprocess.run([str(binary)],input=' '.join(map(integer_text,w))+'\n',text=True,capture_output=True,timeout=35)
    if not proc.stdout or proc.returncode:
        raise LimitError('Toric fan refinement exceeded its computation budget.')
    fan=json.loads(proc.stdout)
    if 'error' in fan:
        raise LimitError(fan['error'])
    rays=fan['rays']
    cones=fan['cones']
    if len(exps)*len(cones)>150_000:
        raise LimitError('Toric atlas exceeds the chart expansion budget.')
    # The Newton polyhedron is the diagonal simplex plus the positive orthant:
    # every higher-weight exponent lies above its supporting plane.
    valuations = pairings(rays, principal)
    orders, face = minimum_faces(valuations)
    from .chart_batch import transform_batch
    transformed,batch_info=transform_batch(exps,rays,cones,orders,compute=compute)
    characters = {}
    for edge in dict.fromkeys(f for f in face[3:] if mode == "intrinsic" and len(f) == 2):
        a, b = edge
        _, degree, direction, coefficient = character(principal[a], principal[b], p[principal[a]], p[principal[b]])
        characters[edge] = degree, tuple(v for v, in pairings(rays, (direction,))), rational_text(coefficient)
    coords=['u','v','w']
    def eid(i):
        return f'E{i-2}'
    divs={}
    vertices=[]
    roots={}
    for i in range(3,len(rays)):
        v={'id':eid(i),'ray':rays[i],'total_multiplicity':orders[i]}
        if mode=='embedded':
            vertices.append(v);divs[i]=[v['id']]
        elif len(face[i])==3:
            # Its open torus slice is the quotient of an irreducible Laurent
            # hypersurface by the radial G_m. Its strict closure is irreducible.
            v['label']='Central curve';vertices.append(v);divs[i]=[v['id']]
        elif len(face[i])==2:
            a,b=face[i];g=characters[face[i]][0]
            divs[i]=[]
            family=f'alpha{a+1}{b+1}'
            roots[family]={'degree':g,'power_value':characters[a,b][2],
                'definition':f'{family}_j is root j of T^{g} = {characters[a,b][2]}, ordered by real then imaginary part, starting at 1.'}
            for j in range(1,g+1):
                ident=f'{eid(i)}.{j}';divs[i].append(ident)
                vertices.append({'id':ident,'ray':rays[i],'root':f'{family}_{j}','ambient_divisor':eid(i)})
        else:
            divs[i]=[]
    face_rows = [(len(indices), *indices) for indices in face]
    incidence = run(f"diagonal_cells 3 {len(rays)} {len(cones)} {int(mode == 'intrinsic')}", *face_rows, *cones)
    cells = []
    for row in incidence:
        kind, *fields = row.split()
        indices = list(map(int, fields))
        if kind == "triangle" or mode == "embedded":
            cells.append({"vertices": [eid(i) for i in indices]})
            continue
        i, j, a, b = indices
        for t in range(characters[a,b][0]):
            cells.append({"vertices": [divs[i][t if len(face[i]) == 2 else 0], divs[j][t if len(face[j]) == 2 else 0]],
                          "stratum_root": f"alpha{a+1}{b+1}_{t+1}"})
    # A unimodular substitution is injective on monomials. Keep the input's
    # canonical term order across charts; no per-chart sorting or collecting
    # is needed. Reuse the same coefficient and monomial strings in the atlas.
    coeffs = []
    for i, e in enumerate(exps):
        text = rational_text(p[e])
        sign = "-" if text.startswith("-") else "+" if i else ""
        absolute = text.lstrip("-")
        coeffs.append((sign, "" if absolute == "1" else absolute))
    powers=[{0:'',1:v} for v in coords]
    def power(j,n):
        cache=powers[j]
        if n not in cache: cache[n]=f'{coords[j]}^{n}'
        return cache[n]
    def equation(triples):
        pieces=[]
        for (a,b,c),(sign,coeff) in zip(triples,coeffs):
            term='*'.join(t for t in (coeff,power(0,a),power(1,b),power(2,c)) if t)
            pieces.append(sign+(term or '1'))
        return ''.join(pieces)
    charts=[]
    for no,cone in enumerate(cones,1):
        rr=[rays[i] for i in cone];ords=[orders[i] for i in cone]
        maps={variables[j]:monomial(coords,[r[j] for r in rr]) for j in range(3)}
        exc=[]
        for k,i in enumerate(cone):
            if i<3:
                continue
            if mode=='embedded' or len(face[i])==3:
                exc.append({'id':eid(i),'equation':coords[k]})
            elif len(face[i])==2:
                a,b=face[i]
                chi=[characters[a,b][1][j] for j in cone]
                plus, minus = signed_powers(tuple(chi))
                pos, neg = monomial(coords, plus), monomial(coords, minus)
                for t,ident in enumerate(divs[i],1):
                    root=f'alpha{a+1}{b+1}_{t}'
                    ideal=[coords[k],f'{pos}-{root}*{neg}']
                    exc.append({'id':ident,'equation':', '.join(ideal),'ideal':ideal})
        charts.append({'id':f'T{no}','coordinates':coords,'map':maps,
            'strict_transform':equation(transformed[no-1]),'exceptional':exc,
            'pullback_factor':monomial(coords,ords),'ray_indices':cone})
    dual={'status':'complete','convention':('Reduced ambient exceptional divisor' if mode=='embedded'
            else 'Reduced exceptional curves on the resolved surface')+' over the origin, over the algebraic closure of Q.',
          'vertices':vertices,'cells':cells}
    return {'status':'resolved','engine':'native-toric','summary':f'Regular toric resolution with {len(cones)} affine charts.',
        'charts':charts,'dual_complex':dual,'steps':[{'id':'Toric','type':'toric modification','weight':list(w)}],
        'fan':fan,'root_families':roots,'certificate':{'method':'regular refinement of a diagonal compact Newton face',
            'chart_batch':batch_info,'weighted_degree':level,'weights':list(w),'principal_exponents':principal,
            'description':'Every maximal cone is unimodular. Each initial form over the exceptional fiber is a monomial, a separable binomial on a torus, or the smooth diagonal torus hypersurface.'},
        'notes':['Charts sharing rays glue by the monomial coordinate changes of the fan. The map is an isomorphism off the origin. Smoothness of the strict transform is certified along the origin fiber; the affine input may have other singularities. The resolution need not be minimal.']}
