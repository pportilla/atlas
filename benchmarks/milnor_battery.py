"""Resolve a list of three-variable germs in quick mode and compare the
Milnor number read off each monodromy zeta function (the degree of the zeta
function is the Euler characteristic of the Milnor fibre, 1 + mu for an
isolated surface singularity) with the one Singular computes.

Singular is used here as an outside yardstick only; nothing in Atlas depends
on it. Run from the project directory: python3 benchmarks/milnor_battery.py
A germ that Singular reports as not isolated (-1) is listed without a comparison.
"""
import json, re, subprocess, sys, time
germs = ["x^2+y^3+z^5", "x^2+y^3+z^3", "(x+y)^2+z^3+x^3", "(x*y-z^2)^2+x^5+y^5", "x^3+y^3+z^3-3*x*y*z+x^4+y^4+z^4",
         "z^3+y^4+(1/432)*x^6+x*y^2*z+x^3*y*z+x*y*z^2", "(y^2-x^3)^2+z^2*x^3+z^5", "x^2*y+y^4+z^3", "x^3+y^3+z^4",
         "(x+y+z)^2+x^3+y^3+z^3+x*y*z", "(x-y)^3+z^2+x^4+y^5", "x*y*z+x^4+y^4+z^4", "(x^2+y^2)^2+z^2+x^5",
         "(x+z)^2*(y+z)+x^4+y^4+z^5", "x^2+(y+z)^3+y^4", "(x*y+z^2)*(x+y)+x^5+y^5+z^5", "(x+2*y)^2+(y+z)^3+z^4",
         "x^2*y^2+y^2*z^2+z^2*x^2+x^5+y^5+z^5", "(x^2-y^3)^2+z^2+x^7", "(y-x^2)^2+z^2+x^5"]
def milnor(text):
    script = f'LIB "sing.lib"; ring r=0,(x,y,z),ds; poly f={text}; milnor(f); quit;'
    out = subprocess.run(["Singular", "-q"], input=script, text=True, capture_output=True, timeout=120).stdout.split()
    return int(out[-1]) if out else None
def degree(expression):
    num, _, den = expression.partition("/")
    def total(part):
        return sum(int(n)*int(e or 1) for n, e in re.findall(r"\(1-T(?:\^(\d+))?\)(?:\^(\d+))?", part.replace("(1-T)", "(1-T^1)")))
    if not den:
        return -total(num) if num.strip() != "1" else 0
    return total(den) - total(num)
bad = 0
for germ in germs:
    request = json.dumps({"polynomial": germ, "variables": ["x", "y", "z"], "mode": "embedded", "monodromy": True, "search": "quick"})
    start = time.time()
    done = subprocess.run([sys.executable, "-m", "engine.worker"], input=request, text=True, capture_output=True)
    try:
        status, data = json.loads(done.stdout)
    except Exception:
        print(f"{germ}: no answer ({done.stderr[-200:]})"); bad += 1; continue
    zeta = (data.get("monodromy_zeta") or {}).get("expression")
    line = f"{time.time()-start:5.1f}s {data.get('status'):12} {str(data.get('engine')):26}"
    if zeta:
        mine, theirs = degree(zeta) - 1, milnor(germ)
        if theirs is None or theirs < 0:
            line += f" chi(F) {mine + 1}, not isolated  Ztop {(data.get('topological_zeta') or {}).get('status')}"
        else:
            ok = mine == theirs
            bad += not ok
            line += f" mu {mine} {'=' if ok else '!='} {theirs}  Ztop {(data.get('topological_zeta') or {}).get('status')}"
    print(line, "|", germ, flush=True)
print("disagreements:", bad)
