"""Decision-tree layer 4 (motion planning): a joint-space path from q_start to q_goal that the chosen world
representation calls collision-free.

OMPL (geometric):  prm, prmstar, rrt, rrtconnect, rrtstar, informedrrtstar, kpiece
Written here:      lazyprm, est, kinorrt (kinodynamic RRT), sst, astar, dijkstra, lattice, chomp, stomp, trajopt,
                   potential (potential fields), dampers (velocity dampers), straight (no planner)
The visual-servo branch's "direct visual approach" bypasses this layer (pipeline.py).
"""
import heapq
import time

import numpy as np
from scipy.optimize import minimize

MARGIN = 0.025         # required clearance (m): covers the ~1-2 cm tracking error measured in Phase 0
STEP = 0.03            # motion-check resolution (rad)


class Space:
    def __init__(self, lo, hi, clearance_fn, budget_s=2.0, seed=0):
        self.lo, self.hi, self.n = np.asarray(lo, float), np.asarray(hi, float), len(lo)
        self.clear = clearance_fn
        self.budget = budget_s
        self.rng = np.random.default_rng(seed)
        self.checks = 0

    def valid(self, q):
        self.checks += 1
        return bool(np.all(q >= self.lo) and np.all(q <= self.hi) and self.clear(q) > MARGIN)

    def motion_ok(self, a, b):
        k = max(2, int(np.ceil(np.max(np.abs(b - a)) / STEP)))
        return all(self.valid(a + (b - a) * t) for t in np.linspace(0, 1, k)[1:])

    def sample(self):
        return self.rng.uniform(self.lo, self.hi)


def shortcut(space, path, iters=1000):
    """Common post-processing for every planner, as OMPL's simplifier does: densify, take random shortcuts that
    are collision-free, then drop every vertex whose neighbours connect directly."""
    P = [np.asarray(q, float) for q in path]
    dense = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        k = max(1, int(np.ceil(np.max(np.abs(b - a)) / 0.2)))
        dense += [a + (b - a) * t for t in np.linspace(0, 1, k + 1)[1:]]
    P = dense
    for _ in range(iters):
        if len(P) < 3:
            break
        i, j = sorted(space.rng.choice(len(P), 2, replace=False))
        if j - i < 2:
            continue
        if space.motion_ok(P[i], P[j]):
            P = P[:i + 1] + P[j:]
    i = 0
    while i < len(P) - 2:
        if space.motion_ok(P[i], P[i + 2]):
            del P[i + 1]
        else:
            i += 1
    return P


def path_length(path):
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])))


# ------------------------------------------------------------------ OMPL geometric planners
def ompl_plan(space, q0, q1, name):
    from ompl import base as ob, geometric as og, util as ou
    ou.setLogLevel(ou.LogLevel.LOG_NONE)
    if hasattr(ou, "RNG") and hasattr(ou.RNG, "setSeed"):
        try:
            ou.RNG.setSeed(int(space.rng.integers(1, 2**31 - 1)))   # reproducible runs (global OMPL seed)
        except Exception:
            pass
    S = ob.RealVectorStateSpace(space.n)
    bd = ob.RealVectorBounds(space.n)
    for i in range(space.n):
        bd.setLow(i, float(space.lo[i])); bd.setHigh(i, float(space.hi[i]))
    S.setBounds(bd)
    ss = og.SimpleSetup(S)
    ss.setStateValidityChecker(lambda s: space.valid(np.array([s[i] for i in range(space.n)])))
    ss.getSpaceInformation().setStateValidityCheckingResolution(STEP / max(np.max(space.hi - space.lo), 1e-6))
    a, b = S.allocState(), S.allocState()
    for i in range(space.n):
        a[i], b[i] = float(q0[i]), float(q1[i])
    ss.setStartAndGoalStates(a, b)
    cls = {"prm": og.PRM, "prmstar": og.PRMstar, "rrt": og.RRT, "rrtconnect": og.RRTConnect, "rrtstar": og.RRTstar,
           "informedrrtstar": og.InformedRRTstar, "kpiece": og.KPIECE1}[name]
    ss.setPlanner(cls(ss.getSpaceInformation()))
    # anytime planners use the whole budget; feasibility planners stop at the first solution
    solved = ss.solve(space.budget)
    if not solved or not ss.haveExactSolutionPath():
        return None
    p = ss.getSolutionPath()
    return [np.array([p.getState(k)[i] for i in range(space.n)]) for k in range(p.getStateCount())]


# ------------------------------------------------------------------ sampling planners written here
def lazyprm(space, q0, q1, n_nodes=300, k=10):
    """Roadmap built without edge checks; edges are checked only on the candidate shortest path."""
    t0 = time.time()
    nodes = [q0, q1] + [space.sample() for _ in range(n_nodes)]
    nodes = [q for i, q in enumerate(nodes) if i < 2 or space.valid(q)]
    X = np.array(nodes)
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    nbr = np.argsort(D, axis=1)[:, 1:k + 1]
    bad = set()
    while time.time() - t0 < space.budget:
        dist, prev, pq = {0: 0.0}, {}, [(0.0, 0)]
        while pq:
            dcur, u = heapq.heappop(pq)
            if u == 1:
                break
            if dcur > dist.get(u, np.inf):
                continue
            for v in nbr[u]:
                if (u, v) in bad:
                    continue
                nd = dcur + D[u, v]
                if nd < dist.get(v, np.inf):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(pq, (nd, v))
        if 1 not in prev:
            return None
        chain = [1]
        while chain[-1] != 0:
            chain.append(prev[chain[-1]])
        chain = chain[::-1]
        ok = True
        for u, v in zip(chain[:-1], chain[1:]):
            if not space.motion_ok(X[u], X[v]):
                bad.add((u, v)); bad.add((v, u)); ok = False
                break
        if ok:
            return [X[i] for i in chain]
    return None


def est(space, q0, q1, step=0.25, goal_bias=0.1):
    """Expansive space trees: expand from nodes in sparsely covered regions (inverse neighbour density)."""
    t0 = time.time()
    V, parent = [q0], [-1]
    while time.time() - t0 < space.budget:
        X = np.array(V)
        dens = np.array([np.sum(np.linalg.norm(X - x, axis=1) < 2 * step) for x in X])
        i = space.rng.choice(len(V), p=(1 / dens) / np.sum(1 / dens))
        if space.rng.random() < goal_bias:
            d = q1 - V[i]
        else:
            d = space.rng.normal(size=space.n)
        q = V[i] + step * d / max(np.linalg.norm(d), 1e-9)
        q = np.clip(q, space.lo, space.hi)
        if space.motion_ok(V[i], q):
            V.append(q); parent.append(i)
            if np.linalg.norm(q - q1) < 1.0 and space.motion_ok(q, q1):   # goal connection attempt
                V.append(q1); parent.append(len(V) - 2)
                return _trace(V, parent)
    return None


def _trace(V, parent):
    out, i = [], len(V) - 1
    while i != -1:
        out.append(V[i]); i = parent[i]
    return out[::-1]


def kinorrt(space, q0, q1, sst=False, dt=0.1, steps=3, amax=3.0, tol=0.08, d_bn=0.2, d_s=0.1):
    """Kinodynamic RRT on a joint-space double integrator (state q, qd; control qdd). sst=True adds the SST
    best-near selection and witness pruning."""
    t0 = time.time()
    n = space.n
    V = [(q0.copy(), np.zeros(n))]; parent = [-1]; cost = [0.0]; witnesses = []
    while time.time() - t0 < space.budget:
        target = q1 if space.rng.random() < 0.15 else space.sample()
        Q = np.array([v[0] for v in V])
        dists = np.linalg.norm(Q - target, axis=1)
        if sst:
            near = np.where(np.linalg.norm(Q - target, axis=1) < dists.min() + d_bn)[0]
            i = int(near[np.argmin(np.array(cost)[near])])
        else:
            i = int(np.argmin(dists))
        q, qd = V[i][0].copy(), V[i][1].copy()
        u = np.clip(4.0 * (target - q) - 2.0 * qd + space.rng.normal(0, 0.5, n), -amax, amax)
        ok, path_cost = True, cost[i]
        for _ in range(steps):
            qn = q + qd * dt + 0.5 * u * dt * dt
            qd = qd + u * dt
            if not space.motion_ok(q, qn):
                ok = False; break
            path_cost += np.linalg.norm(qn - q); q = qn
        if not ok:
            continue
        if sst:
            if witnesses and np.min(np.linalg.norm(np.array(witnesses) - q, axis=1)) < d_s:
                continue
            witnesses.append(q.copy())
        V.append((q, qd)); parent.append(i); cost.append(path_cost)
        if np.linalg.norm(q - q1) < 0.8:
            # steering function: roll the double integrator under PD toward the goal and keep it if it stays valid
            qs, qds, seg, ok = q.copy(), qd.copy(), [], True
            for _ in range(40):
                u = np.clip(9.0 * (q1 - qs) - 6.0 * qds, -amax, amax)
                qn = qs + qds * dt + 0.5 * u * dt * dt
                qds = qds + u * dt
                if not space.motion_ok(qs, qn):
                    ok = False; break
                seg.append(qn); qs = qn
                if np.linalg.norm(qs - q1) < tol:
                    break
            if ok and np.linalg.norm(qs - q1) < tol and space.motion_ok(qs, q1):
                path = _trace([v[0] for v in V], parent)
                return path + seg + [q1]
    return None


# ------------------------------------------------------------------ graph search on a joint lattice
def lattice_search(space, q0, q1, heuristic=True, diagonal=False, res=0.1, max_nodes=30000):
    """A* (heuristic=True), Dijkstra (heuristic=False) or a state lattice (diagonal=True: coupled multi-joint
    primitives plus a straight-to-goal primitive) on a joint grid of resolution res."""
    t0 = time.time()
    n = space.n
    prims = [s * np.eye(n)[i] * res for i in range(n) for s in (1, -1)]
    if diagonal:
        rng = np.random.default_rng(0)
        prims += [np.sign(rng.normal(size=n)) * res for _ in range(4 * n)]
    start = tuple(np.round(q0 / res).astype(int))
    h = (lambda q: np.linalg.norm(q - q1)) if heuristic else (lambda q: 0.0)
    pq = [(h(q0), 0.0, start, None)]
    came, g, real = {}, {start: 0.0}, {start: q0}
    while pq and len(came) < max_nodes and time.time() - t0 < space.budget:
        f, gc, key, par = heapq.heappop(pq)
        if key in came:
            continue
        came[key] = par
        q = real[key]
        if np.linalg.norm(q - q1) < res * 1.5 and space.motion_ok(q, q1):
            path = [q1]
            while key is not None:
                path.append(real[key]); key = came[key]
            return path[::-1]
        moves = list(prims)
        if diagonal:
            d = q1 - q
            moves.append(d / max(np.linalg.norm(d), 1e-9) * res * 3)
        for mv in moves:
            qn = q + mv
            kn = tuple(np.round(qn / res).astype(int))
            if kn in came:
                continue
            ng = gc + np.linalg.norm(mv)
            if ng < g.get(kn, np.inf) and space.motion_ok(q, qn):
                g[kn], real[kn] = ng, qn
                heapq.heappush(pq, (ng + h(qn), ng, kn, key))
    return None


# ------------------------------------------------------------------ trajectory optimizers
def _init_traj(q0, q1, T):
    return np.linspace(q0, q1, T)


def _clear_grad(space, q, eps=1e-3):
    c0 = space.clear(q)
    g = np.zeros(space.n)
    for i in range(space.n):
        dq = q.copy(); dq[i] += eps
        g[i] = (space.clear(dq) - c0) / eps
    return c0, g


def _finite_diff_A(T, n):
    K = np.zeros((T - 1, T))
    for t in range(T - 1):
        K[t, t], K[t, t + 1] = -1, 1
    return K.T @ K + 1e-3 * np.eye(T)


def chomp(space, q0, q1, T=24, iters=150, eta=0.05, eps=0.04):
    """Covariant gradient descent on smoothness + obstacle cost (clearance gradient from the distance field)."""
    t0 = time.time()
    Q = _init_traj(q0, q1, T)
    A = _finite_diff_A(T, space.n)[1:-1, 1:-1]
    Ainv = np.linalg.inv(A)
    for _ in range(iters):
        if time.time() - t0 > space.budget:
            break
        G = np.zeros_like(Q)
        for t in range(1, T - 1):
            c, g = _clear_grad(space, Q[t])
            if c < eps:
                G[t] = -(eps - c) * g * 50.0          # push away along the clearance gradient
        smooth = (_finite_diff_A(T, space.n) @ Q)[1:-1]     # gradient of the squared-velocity smoothness cost
        Q[1:-1] -= eta * Ainv @ (smooth + G[1:-1])
        Q = np.clip(Q, space.lo, space.hi)
        if all(space.motion_ok(a, b) for a, b in zip(Q[:-1], Q[1:])):
            return list(Q)
    return None


def stomp(space, q0, q1, T=24, iters=60, K=12, sigma=0.15):
    """Stochastic trajectory optimization: noisy rollouts drawn from the smoothness covariance, cost-weighted."""
    t0 = time.time()
    Q = _init_traj(q0, q1, T)
    R = _finite_diff_A(T, space.n)[1:-1, 1:-1]
    L = np.linalg.cholesky(np.linalg.inv(R))
    L /= np.max(np.abs(L))

    def cost(Qc):
        c = [max(0.0, 0.04 - space.clear(q)) for q in Qc[1:-1]]
        return 100 * np.sum(np.square(c)) + 0.1 * path_length(Qc)

    for _ in range(iters):
        if time.time() - t0 > space.budget:
            break
        noise = [L @ space.rng.normal(0, sigma, (T - 2, space.n)) for _ in range(K)]
        costs = np.array([cost(np.vstack([q0, np.clip(Q[1:-1] + e, space.lo, space.hi), q1])) for e in noise])
        w = np.exp(-10 * (costs - costs.min()) / (costs.max() - costs.min() + 1e-9)); w /= w.sum()
        Q[1:-1] = np.clip(Q[1:-1] + sum(wi * e for wi, e in zip(w, noise)), space.lo, space.hi)
        if all(space.motion_ok(a, b) for a, b in zip(Q[:-1], Q[1:])):
            return list(Q)
    return None


def trajopt(space, q0, q1, T=16, outer=6):
    """Sequential convex optimization: minimize squared path length subject to linearized clearance >= margin
    at every waypoint, inside a shrinking trust region (SLSQP per outer iteration)."""
    t0 = time.time()
    Q = _init_traj(q0, q1, T)
    n, trust = space.n, 0.3
    for _ in range(outer):
        if time.time() - t0 > space.budget:
            break
        lin = [_clear_grad(space, Q[t]) for t in range(1, T - 1)]
        x0 = Q[1:-1].reshape(-1)

        def f(x):
            X = np.vstack([q0, x.reshape(T - 2, n), q1])
            return float(np.sum(np.diff(X, axis=0) ** 2))

        cons = [{"type": "ineq", "fun": (lambda x, t=t: lin[t][0] + lin[t][1] @ (x.reshape(T - 2, n)[t] - x0.reshape(T - 2, n)[t]) - 2 * MARGIN)}
                for t in range(T - 2)]
        bounds = [(max(lo, v - trust), min(hi, v + trust)) for v, lo, hi in zip(x0, np.tile(space.lo, T - 2), np.tile(space.hi, T - 2))]
        res = minimize(f, x0, method="SLSQP", constraints=cons, bounds=bounds, options={"maxiter": 60})
        Q[1:-1] = res.x.reshape(T - 2, n)
        if all(space.motion_ok(a, b) for a, b in zip(Q[:-1], Q[1:])):
            return list(Q)
        trust *= 0.7
    return None


# ------------------------------------------------------------------ local reactive methods
def potential(space, q0, q1, step=0.03, k_rep=0.004, d0=0.06, max_steps=600):
    """Gradient descent on attractive + repulsive potentials; may stall in a local minimum (tree note)."""
    q, path = q0.copy(), [q0.copy()]
    for _ in range(max_steps):
        if np.linalg.norm(q - q1) < step:
            path.append(q1); return path
        f = q1 - q
        c, g = _clear_grad(space, q)
        if c < d0:
            f += k_rep * (1 / max(c, 1e-3) - 1 / d0) / max(c, 1e-3) ** 2 * g
        qn = np.clip(q + step * f / max(np.linalg.norm(f), 1e-9), space.lo, space.hi)
        if not space.motion_ok(q, qn):
            return None
        if len(path) > 30 and np.linalg.norm(qn - path[-30]) < step:
            return None                                     # stalled
        q = qn; path.append(q.copy())
    return None


def dampers(space, q0, q1, step=0.03, d_i=0.08, d_s=MARGIN, xi=1.0, max_steps=600):
    """Velocity dampers: move toward the goal with the clearance rate bounded, d_dot >= -xi (d - d_s)/(d_i - d_s),
    solved each step as a small least-squares problem with one linear inequality."""
    q, path = q0.copy(), [q0.copy()]
    for _ in range(max_steps):
        if np.linalg.norm(q - q1) < step:
            path.append(q1); return path
        v = q1 - q; v = step * v / max(np.linalg.norm(v), 1e-9)
        c, g = _clear_grad(space, q)
        if c < d_i and g @ v < -xi * (c - d_s) / (d_i - d_s) * step:
            bound = -xi * (c - d_s) / (d_i - d_s) * step
            v = v + (bound - g @ v) * g / max(g @ g, 1e-9)    # project onto the damper constraint
        qn = np.clip(q + v, space.lo, space.hi)
        if not space.motion_ok(q, qn) or (len(path) > 30 and np.linalg.norm(qn - path[-30]) < step):
            return None
        q = qn; path.append(q.copy())
    return None


def straight(space, q0, q1):
    return [q0, q1] if space.motion_ok(q0, q1) else None


PLANNERS = {
    **{k: (lambda k: lambda s, a, b: ompl_plan(s, a, b, k))(k)
       for k in ["prm", "prmstar", "rrt", "rrtconnect", "rrtstar", "informedrrtstar", "kpiece"]},
    "lazyprm": lazyprm, "est": est,
    "kinorrt": lambda s, a, b: kinorrt(s, a, b, sst=False), "sst": lambda s, a, b: kinorrt(s, a, b, sst=True),
    "astar": lambda s, a, b: lattice_search(s, a, b, heuristic=True),
    "dijkstra": lambda s, a, b: lattice_search(s, a, b, heuristic=False),
    "lattice": lambda s, a, b: lattice_search(s, a, b, heuristic=True, diagonal=True),
    "chomp": chomp, "stomp": stomp, "trajopt": trajopt,
    "potential": potential, "dampers": dampers, "straight": straight,
}
