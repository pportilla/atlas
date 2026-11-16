"""Measure reduced, monic QQ bases inside one Singular session per benchmark round."""
import os
import signal
import subprocess


_ORDERS = {"lex": "lp", "grevlex": "dp"}
_CALLS = {
    "std": "std(inputIdeal)",
    "std-pure": "std(inputIdeal)",
    "groebner": "groebner(inputIdeal)",
    "slimgb": "slimgb(inputIdeal)",
    "stdfglm": "stdfglm(inputIdeal)",
    "modStd": "modStd(inputIdeal,1)",
}

# Basis order is immaterial. Match each returned polynomial to exactly one
# reference polynomial using exact equality in the declared QQ ring.
_MATCH = """
proc sameMonicBasis(alias ideal actual, alias ideal expected)
{
    int n=size(actual);
    if(n!=size(expected)){return(0);}
    if(!n){return(1);}
    int i,j;
    // Singular commonly returns the reference polynomials in reverse order.
    for(i=1;i<=n;i++)
    {
        if(actual[i]!=expected[n+1-i]){break;}
    }
    if(i>n){return(1);}
    intvec used;
    used[n]=0;
    for(i=1;i<=n;i++)
    {
        for(j=1;j<=n;j++)
        {
            if(!used[j] && actual[i]==expected[j]){used[j]=1;break;}
        }
        if(j>n){return(0);}
    }
    return(1);
}
"""


def _ideal(polynomials):
    rows = []
    for poly in polynomials:
        terms = []
        for num, den, powers in poly:
            coefficient = str(num) if str(den) == "1" else f"{num}/{den}"
            factors = [f"x{i}^{power}" for i, power in enumerate(powers) if power]
            terms.append("*".join([f"({coefficient})", *factors]))
        rows.append("+".join(terms) or "0")
    return ",".join(rows) or "0"


def _measure_procedure(method, batch):
    compute = f"simplify({_CALLS[method]},1)"  # Normalize leading coefficients inside the timer.
    outputs = [f"answer{i}" for i in range(batch)]
    statements = [
        f"proc measure{batch}(alias ideal inputIdeal, alias ideal expected)",
        "{",
        f"ideal warm = {compute};",
        "kill warm;",
        *(f"ideal {out};" for out in outputs),
        "int start=rtimer;",
        *(f"{out}={compute};" for out in outputs),
        "int elapsed=rtimer-start;",
        *(f"if(!sameMonicBasis({out},expected)){{return(-1);}}" for out in outputs),
        "return(elapsed);",
        "}",
    ]
    return "\n".join(statements)


def singular_round(cases, method, batches, expected, singular="Singular", timeout=180):
    """Return milliseconds per solve, checking every timed answer outside the clock."""
    if method not in _CALLS:
        raise ValueError(f"Unknown Singular Gröbner method {method!r}.")
    if len(batches) != len(cases) or len(expected) != len(cases) or any(
        type(batch) is not int or batch < 1 for batch in batches
    ):
        raise ValueError("Supply one positive batch size and one expected basis per case.")
    script = ['system("--ticks-per-sec",1000000);', "option(redSB);", _MATCH]
    if method == "std-pure":
        script.append("option(pure_gb);")
    if method == "modStd":
        script.append('LIB "modstd.lib";')
    script.extend(_measure_procedure(method, batch) for batch in sorted(set(batches)))
    for i, (record, batch, basis) in enumerate(zip(cases, batches, expected)):
        variables = ",".join(f"x{j}" for j in range(record["n"]))
        script.extend([
            f'ring benchRing{i}=0,({variables}),{_ORDERS[record["order"]]};',
            f'ideal inputIdeal={_ideal(record["polys"])};',
            f'ideal expected={_ideal(basis)};',
            f'print("BENCH {i} "+string(measure{batch}(inputIdeal,expected)));',
            "kill inputIdeal; kill expected;",
        ])
    script.append("quit;")
    # modStd can fork workers; a timed-out method must not run alongside the next trial.
    with subprocess.Popen([singular, "-q"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate("\n".join(script), timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
    # Singular may print syntax/runtime errors to stdout and exit successfully.
    errors = [row for row in stdout.splitlines() if row.lstrip().startswith("?")]
    if process.returncode or errors or stderr.strip():
        raise RuntimeError(f"Singular {method} failed: " + "\n".join([*errors[:6], stderr.strip()]))
    times = []
    for row in stdout.splitlines():
        if row.startswith("BENCH "):
            _, index, elapsed = row.split()
            i = int(index)
            if i != len(times) or i >= len(batches):
                raise RuntimeError(f"Unexpected Singular timing record: {row}")
            if int(elapsed) < 0:
                raise RuntimeError(f"Singular {method} returned an incorrect reduced monic basis for case {i}.")
            times.append(int(elapsed) / (1000 * batches[i]))
    if len(times) != len(cases):
        raise RuntimeError(f"Singular {method} completed {len(times)} of {len(cases)} cases.")
    return times
