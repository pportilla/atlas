"""Closed graphs with contracted fibres or several regularity denominators."""


def crossed_blowup_charts(api):
    x, y, z, a, b, c = api.symbols("x y z a b c")
    variables = ((x,y,z), (a,b,c))
    maps = ((x,y,x*z), (a,a*b,c))
    graph = (x-a, y-a*b, c-x*z)
    pairs = (
        ("exceptional-surfaces", (x,), (a,), 0),
        ("contracted-lines", (x,y), (a,c), 0),
        ("matching-strict-surfaces", (z,), (c,), 1),
        ("different-strict-surfaces", (z,), (b,), -1),
    )
    return variables, maps, ((), ()), graph, pairs


def implicit_cone_charts(api):
    x, y, z, w, a, b, c = api.symbols("x y z w a b c")
    variables = ((x,y,z,w), (a,b,c))
    maps = ((x,y,z,w), (a,a*b,c,c*b))
    relation = x*w-y*z
    graph = (x-a, z-c, y-a*b, w-c*b)
    pairs = (
        ("cone-x-zero", (relation,x,y), (a,), 1),
        ("cone-z-zero", (relation,z,w), (c,), 1),
    )
    return variables, maps, ((relation,), ()), graph, pairs
