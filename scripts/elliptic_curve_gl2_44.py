"""
=============================================================================
[Project RDL] 결과 #44 — 타원곡선 L-함수 ξ-다발 검증 (GL(2) 첫 사례)
=============================================================================
대상: 11a1 (y² + y = x³ - x² - 10x - 20)
  - conductor N = 11, rank 0, root number ε = +1
  - weight k = 2, newform on Γ₀(11)
  - LMFDB: https://www.lmfdb.org/EllipticCurve/Q/11/a/1

임계선: σ = 1 (⚠️ GL(1)의 σ=1/2와 다름!)

검증 항목 (4성질):
  1. σ-유일성: 위상 점프가 σ=1에서 최대
  2. 모노드로미: 영점 주위 ±π 양자화 (폐곡선 적분)
  3. κ 집중: near(σ=1)/far(σ≠1) >> 1
  4. 블라인드 예측: κ 피크로 영점 위치 예측

핵심 구현:
  - aₙ: 11a1 점 세기(E(𝔽ₚ)) + multiplicative 재귀
  - L(E,s): AFE (근사 함수 방정식, upper incomplete gamma)
  - Λ(E,s) = (√N/2π)^s Γ(s) L(E,s)
  - Λ'/Λ = 수치 중앙차분 (h=1e-6)

결과 파일: results/elliptic_curve_11a1.txt
=============================================================================
"""

import sys, os, time
import numpy as np
import mpmath

sys.path.insert(0, os.path.expanduser('~/Desktop/gdl_unified'))
sys.path.insert(0, os.path.expanduser('~/Desktop/gdl_unified/scripts'))

mpmath.mp.dps = 80  # AFE 상쇄 대비 높은 정밀도

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 상수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

N_COND = 11           # conductor
EPSILON = 1           # root number +1
SIGMA_CRIT = 1.0      # ⚠️ 임계선 σ = 1 (GL(2)!)
T_MIN, T_MAX = 0.5, 30.0
DELTA_OFFSET = 0.03   # 영점 위 직접 측정 금지 → 오프셋
N_MAX_COEFF = 80      # AFE 항 수 (N=11이면 n~100에서 e^{-190} ≈ 0)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: aₙ 계수 계산 — 11a1 점 세기 + multiplicative 재귀
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_ap_11a1(p):
    """
    aₚ for 11a1: y² + y = x³ - x² - 10x - 20

    p=11 (bad prime, split multiplicative): a₁₁ = 1
    p=2: 직접 열거
    p≠2,11 (good odd prime): Legendre 기호로 점 세기
        변환: (2y+1)² = 4x³ - 4x² - 40x - 79
        aₚ = p - Σ_{x=0}^{p-1} (1 + legendre(disc_x, p)) + p
        = -Σ legendre(disc_x, p)
    """
    if p == 11:
        return 1  # split multiplicative reduction

    if p == 2:
        # 직접 열거: y² + y = x³ - x² - 10x - 20 (mod 2)
        count_pts = 1  # point at infinity
        for x in range(2):
            for y in range(2):
                if (y * y + y - x * x * x + x * x + 10 * x + 20) % 2 == 0:
                    count_pts += 1
        return p + 1 - count_pts

    # 홀수 good prime: Legendre 기호 방식
    # disc(x) = 4x³ - 4x² - 40x - 79
    # #affine pts = Σ (1 + legendre(disc(x), p))
    # #E(𝔽ₚ) = 1 + #affine pts
    # aₚ = p + 1 - #E(𝔽ₚ) = p - #affine pts
    affine_count = 0
    for x in range(p):
        disc = (4 * x * x * x - 4 * x * x - 40 * x - 79) % p
        if disc == 0:
            affine_count += 1  # 1 solution (double root)
        else:
            leg = pow(disc, (p - 1) // 2, p)
            if leg == 1:
                affine_count += 2  # 2 solutions (QR)
            # else: leg == p-1 → QNR → 0 solutions

    return p - affine_count


def compute_an_table(n_max):
    """aₙ (n=1..n_max) via multiplicativity."""
    # 소수 체
    sieve = [True] * (n_max + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n_max**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n_max + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, n_max + 1) if sieve[i]]

    # aₚ for all primes
    ap = {}
    for p in primes:
        ap[p] = compute_ap_11a1(p)

    # a_{p^k} 재귀
    apk = {}
    for p in primes:
        apk[(p, 0)] = 1
        apk[(p, 1)] = ap[p]
        pk = p
        k = 1
        while pk * p <= n_max:
            pk *= p
            k += 1
            if p == 11:
                # bad prime: a_{p^k} = aₚ^k (for split mult, aₚ=1 → always 1)
                apk[(p, k)] = ap[p] ** k
            else:
                # good prime: a_{p^k} = aₚ·a_{p^{k-1}} - p·a_{p^{k-2}}
                apk[(p, k)] = ap[p] * apk[(p, k - 1)] - p * apk[(p, k - 2)]

    # 전체 aₙ (multiplicativity)
    an = [0] * (n_max + 1)
    an[1] = 1

    for n in range(2, n_max + 1):
        temp = n
        result = 1
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                k = 0
                while temp % p == 0:
                    k += 1
                    temp //= p
                result *= apk[(p, k)]
        if temp > 1:
            result *= ap[temp]
        an[n] = result

    return an


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: AFE (Approximate Functional Equation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Λ(E,s) = Σ_{n=1}^{n_max} aₙ · h(n,s)
# h(n,s) = xₙ^{-s} Γ(s, xₙ) + ε · xₙ^{-(2-s)} Γ(2-s, xₙ)
# where xₙ = 2πn/√N, Γ(s,x) = upper incomplete gamma
#
# L(E,s) = Λ(E,s) / [(√N/2π)^s · Γ(s)]
#
# 수렴: xₙ ~ 1.894n → n=45에서 e^{-85} < 10^{-37}, n=80에서 e^{-151.5} < 10^{-66}
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_an_cache = None
_precomp_cache = None  # [(mpf(aₙ), xₙ), ...]


def _init_tables():
    """aₙ + 전처리 테이블 초기화"""
    global _an_cache, _precomp_cache
    if _an_cache is not None:
        return

    print("  [초기화] aₙ 계수 계산 (n ≤ %d)..." % N_MAX_COEFF, flush=True)
    t0 = time.time()
    _an_cache = compute_an_table(N_MAX_COEFF)

    # 전처리: xₙ 값 + 0이 아닌 항만 필터
    sqrt_N = mpmath.sqrt(mpmath.mpf(N_COND))
    two_pi = 2 * mpmath.pi
    _precomp_cache = []
    for n in range(1, N_MAX_COEFF + 1):
        if _an_cache[n] == 0:
            continue
        x_n = two_pi * n / sqrt_N
        _precomp_cache.append((mpmath.mpf(_an_cache[n]), x_n))

    print(f"  [초기화] 완료 ({time.time()-t0:.1f}초, 비영 항 {len(_precomp_cache)}개)", flush=True)

    # 검증: 알려진 aₚ 값
    known = {2: -2, 3: -1, 5: 1, 7: -2, 11: 1, 13: 4, 17: -2, 19: 0, 23: -1}
    for p, expected in known.items():
        actual = _an_cache[p]
        if actual != expected:
            print(f"  ⚠️ a_{p} = {actual}, expected {expected}!", flush=True)
    print(f"  검증: a₂={_an_cache[2]}, a₃={_an_cache[3]}, a₅={_an_cache[5]}, "
          f"a₇={_an_cache[7]}, a₁₁={_an_cache[11]}, a₁₃={_an_cache[13]}", flush=True)


def Lambda_E(s):
    """
    Λ(E,s) via AFE:
    Λ(E,s) = Σ aₙ · [xₙ^{-s} Γ(s, xₙ) + ε · xₙ^{-(2-s)} Γ(2-s, xₙ)]

    where xₙ = 2πn/√N, Γ(s,x) = upper incomplete gamma
    """
    _init_tables()
    s_mp = mpmath.mpc(s)
    s_conj = 2 - s_mp
    eps = mpmath.mpf(EPSILON)

    result = mpmath.mpc(0)
    for an_val, x_n in _precomp_cache:
        term1 = mpmath.power(x_n, -s_mp) * mpmath.gammainc(s_mp, x_n)
        term2 = eps * mpmath.power(x_n, -s_conj) * mpmath.gammainc(s_conj, x_n)
        result += an_val * (term1 + term2)

    return result


def L_E(s):
    """L(E,s) = Λ(E,s) / [(√N/2π)^s · Γ(s)]"""
    s_mp = mpmath.mpc(s)
    Lambda_val = Lambda_E(s_mp)

    sqrt_N = mpmath.sqrt(mpmath.mpf(N_COND))
    two_pi = 2 * mpmath.pi
    prefactor = mpmath.power(sqrt_N / two_pi, s_mp) * mpmath.gamma(s_mp)

    if abs(prefactor) < mpmath.mpf(10)**(-mpmath.mp.dps + 10):
        return mpmath.mpc(0)

    return Lambda_val / prefactor


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: Connection, Curvature
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def connection_Lambda(s):
    """Λ'/Λ via 중앙차분 (h=1e-6)"""
    s_mp = mpmath.mpc(s)
    h = mpmath.mpf('1e-6')

    L0 = Lambda_E(s_mp)
    if abs(L0) < mpmath.mpf(10)**(-mpmath.mp.dps + 15):
        return mpmath.mpc(1e10, 0)

    L_plus = Lambda_E(s_mp + h)
    L_minus = Lambda_E(s_mp - h)

    return (L_plus - L_minus) / (2 * h * L0)


def curvature_at(s):
    """κ(s) = |Λ'/Λ|²"""
    conn = connection_Lambda(s)
    k = float(abs(conn)**2)
    return k if np.isfinite(k) else 1e12


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: 영점 탐색 (σ = 1 임계선)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_zeros_elliptic(t_min=T_MIN, t_max=T_MAX, n_scan=1500):
    """
    σ=1 임계선에서 Re(Λ) 부호 변화 → findroot 정밀화.
    Λ(E, 1+it) is real (∵ ε=+1, 함수 방정식) → Re(Λ) 부호 변화 = 영점.
    """
    print(f"\n[영점 탐색] t ∈ [{t_min}, {t_max}], n_scan={n_scan}", flush=True)
    ts = np.linspace(t_min, t_max, n_scan)
    zeros = []
    fail_count = 0

    # ── Re(Λ) 부호 변화 스캔 ──
    prev_re, prev_t = None, None
    for i, t in enumerate(ts):
        s = mpmath.mpc(SIGMA_CRIT, t)
        try:
            val = Lambda_E(s)
            curr_re = float(mpmath.re(val))
        except Exception as e:
            print(f"  WARNING: Lambda_E 실패 t={t:.3f}: {e}", flush=True)
            prev_re, prev_t = None, float(t)
            continue

        if prev_re is not None and prev_re * curr_re < 0:
            mid = (prev_t + float(t)) / 2
            try:
                def f_re(t_var):
                    sv = mpmath.mpc(SIGMA_CRIT, mpmath.mpf(t_var))
                    return mpmath.re(Lambda_E(sv))

                tz = float(mpmath.findroot(f_re, mpmath.mpf(str(mid))))

                # 확인: |Λ| 충분히 작은지
                sv = mpmath.mpc(SIGMA_CRIT, tz)
                lv = float(abs(Lambda_E(sv)))

                if not zeros or abs(tz - zeros[-1]) > 0.05:
                    zeros.append(tz)
                    print(f"  ✓ γ_{len(zeros)} = {tz:.8f}  (|Λ| = {lv:.2e})", flush=True)
            except Exception as e:
                fail_count += 1
                if fail_count <= 5:
                    print(f"  WARNING: findroot 실패 t≈{mid:.3f}: {e}", flush=True)

        prev_re, prev_t = curr_re, float(t)

        if (i + 1) % 500 == 0:
            elapsed_frac = (i + 1) / n_scan
            print(f"  ... 스캔 {i+1}/{n_scan} ({elapsed_frac*100:.0f}%, 영점 {len(zeros)}개)", flush=True)

    if len(zeros) == 0:
        print("  ⚠️ 영점 0개 — Re(Λ) 스캔으로 발견 못함, |Λ| 최소화 시도...", flush=True)
        # Fallback: |Λ(1+it)| 최소화
        ts_fine = np.linspace(t_min + 1.0, t_max - 1.0, 500)
        abs_vals = []
        for t in ts_fine:
            try:
                val = abs(Lambda_E(mpmath.mpc(SIGMA_CRIT, t)))
                abs_vals.append((float(val), t))
            except:
                abs_vals.append((1e20, t))

        abs_vals.sort()
        for val, t in abs_vals[:20]:
            if val < 1.0:
                try:
                    def f_abs(t_var):
                        sv = mpmath.mpc(SIGMA_CRIT, mpmath.mpf(t_var))
                        lam = Lambda_E(sv)
                        return mpmath.re(lam)

                    tz = float(mpmath.findroot(f_abs, mpmath.mpf(str(t))))
                    if not any(abs(tz - z) < 0.05 for z in zeros):
                        zeros.append(tz)
                        print(f"  ✓ (fallback) γ = {tz:.8f}", flush=True)
                except Exception:
                    pass

    if len(zeros) == 0:
        print("  ⚠️⚠️ 영점 완전 실패 — AFE 구현 점검 필요!", flush=True)

    if fail_count > 0:
        print(f"  findroot 실패: {fail_count}회", flush=True)

    zeros.sort()
    print(f"  총 {len(zeros)}개 영점 발견", flush=True)
    return np.array(zeros)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 5: 4성질 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def measure_sigma_uniqueness(zeros, n_t_scan=300):
    """
    σ-유일성: 각 σ에서 Re(Λ(σ+it)) 부호 변화 카운트.
    σ=1(임계선)에서 최대 점프 기대.
    """
    sigmas = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    print(f"\n[σ-유일성] σ = {sigmas}, n_t_scan = {n_t_scan}", flush=True)
    results = {}

    for sigma in sigmas:
        t0 = time.time()
        ts = np.linspace(T_MIN + 0.5, T_MAX - 0.5, n_t_scan)
        jumps = 0
        prev_re = None

        for t in ts:
            s = mpmath.mpc(sigma, t)
            try:
                val = Lambda_E(s)
                curr_re = float(mpmath.re(val))
            except Exception:
                prev_re = None
                continue

            if prev_re is not None and prev_re * curr_re < 0:
                jumps += 1
            prev_re = curr_re

        results[sigma] = jumps
        marker = " ← 임계선" if abs(sigma - SIGMA_CRIT) < 0.01 else ""
        print(f"  σ={sigma:.1f}: jumps = {jumps}{marker}  ({time.time()-t0:.0f}초)", flush=True)

    return results


def measure_monodromy_contour(zeros, radius=0.3, n_steps=48):
    """
    모노드로미: 폐곡선 적분 (σ=1 중심, 반지름 radius).
    arg(Λ) 누적 → 영점이 원 안에 있으면 ≈±2π (winding number 1).
    |mono|/π 보고.
    """
    print(f"\n[모노드로미 — 폐곡선 적분] radius={radius}, n_steps={n_steps}", flush=True)

    if len(zeros) == 0:
        return None, None

    monos = []

    for tz in zeros:
        center = mpmath.mpc(SIGMA_CRIT, tz)
        phase_accum = mpmath.mpf(0)
        prev_val = None
        ok = True

        for j in range(n_steps + 1):
            theta = 2 * mpmath.pi * j / n_steps
            s = center + radius * mpmath.exp(1j * theta)
            try:
                val = Lambda_E(s)
                if abs(val) < mpmath.mpf(10)**(-mpmath.mp.dps + 15):
                    ok = False
                    break
            except Exception:
                ok = False
                break

            if prev_val is not None:
                ratio = val / prev_val
                phase_accum += mpmath.im(mpmath.log(ratio))
            prev_val = val

        if not ok:
            print(f"  ⚠ γ={tz:.4f}: 계산 실패 (영점 경유 또는 underflow)", flush=True)
            continue

        mono = float(phase_accum)
        monos.append(mono)
        print(f"  γ={tz:.6f}: mono = {mono:.4f} rad, mono/π = {mono/np.pi:.6f}", flush=True)

    if not monos:
        return None, None

    abs_monos_over_pi = [abs(m) / np.pi for m in monos]
    # 영점에서 winding number = 1 → |mono| = 2π → |mono|/π = 2
    # 또는 ξ-다발 프레임워크 정의에 따라 |mono|/π 보고
    mean_abs = float(np.mean(abs_monos_over_pi))
    deviations = [abs(m - round(m)) for m in abs_monos_over_pi]  # 가장 가까운 정수로부터 편차
    mean_dev = float(np.mean(deviations))

    return mean_dev, mean_abs


def measure_monodromy_logspace(zeros, eps_offset=0.005):
    """
    모노드로미: log-space arg 보조 측정 (GL(1) 교차비교와 동일 방식).
    Λ(E, 1+it)는 실수이므로, 영점에서 부호 반전 → |Δarg| = π.
    """
    print(f"\n[모노드로미 — log-space arg] eps={eps_offset}", flush=True)

    if len(zeros) == 0:
        return None, None

    monos = []

    for tz in zeros:
        # Λ(1+i(tz±eps)) is real → arg = 0 or π
        s_plus = mpmath.mpc(SIGMA_CRIT, tz + eps_offset)
        s_minus = mpmath.mpc(SIGMA_CRIT, tz - eps_offset)

        try:
            val_plus = Lambda_E(s_plus)
            val_minus = Lambda_E(s_minus)

            if abs(val_plus) < 1e-50 or abs(val_minus) < 1e-50:
                continue

            # log-space: Im(log(Λ)) = arg(Λ). Λ real → arg = 0 or π.
            arg_plus = float(mpmath.arg(val_plus))
            arg_minus = float(mpmath.arg(val_minus))
            delta = arg_plus - arg_minus

            # 부호 변화 → delta ≈ ±π
            monos.append(delta)
            print(f"  γ={tz:.6f}: Δarg = {delta:.6f}, |Δarg|/π = {abs(delta)/np.pi:.6f}", flush=True)
        except Exception as e:
            print(f"  ⚠ γ={tz:.4f}: {e}", flush=True)

    if not monos:
        return None, None

    abs_monos_over_pi = [abs(m) / np.pi for m in monos]
    mean_abs = float(np.mean(abs_monos_over_pi))
    deviations = [abs(abs(m) / np.pi - 1.0) for m in monos]
    mean_dev = float(np.mean(deviations))

    return mean_dev, mean_abs


def measure_kappa_concentration(zeros, n_generic=40):
    """
    κ 집중도: median(κ near zeros) / median(κ far from zeros).
    GL(2) 임계선 σ=1, 오프셋 δ=0.03.
    """
    print(f"\n[κ 집중도] δ={DELTA_OFFSET}", flush=True)

    if len(zeros) == 0:
        return None

    # Near: 각 영점에서 δ 오프셋
    near_k = []
    for tz in zeros:
        s = mpmath.mpc(SIGMA_CRIT, tz + DELTA_OFFSET)
        try:
            k = curvature_at(s)
            if np.isfinite(k) and k < 1e11:
                near_k.append(k)
                print(f"    near γ={tz:.4f}: κ = {k:.1f}", flush=True)
        except Exception as e:
            print(f"  WARNING: near κ 실패 t={tz:.2f}: {e}", flush=True)

    # Far: 영점에서 1.0 이상 떨어진 랜덤 점
    far_k = []
    rng = np.random.RandomState(42)
    attempts = 0
    while len(far_k) < n_generic and attempts < n_generic * 5:
        t = rng.uniform(T_MIN + 1.0, T_MAX - 1.0)
        if len(zeros) > 0 and np.min(np.abs(zeros - t)) < 1.0:
            attempts += 1
            continue
        s = mpmath.mpc(SIGMA_CRIT, t)
        try:
            k = curvature_at(s)
            if np.isfinite(k) and k < 1e11:
                far_k.append(k)
        except Exception:
            pass
        attempts += 1

    if not near_k or not far_k:
        print("  ⚠ near 또는 far κ 부족", flush=True)
        return None

    near_med = float(np.median(near_k))
    far_med = float(np.median(far_k))
    ratio = near_med / far_med if far_med > 0 else float('inf')

    print(f"  near median: {near_med:.1f} ({len(near_k)} pts)", flush=True)
    print(f"  far median:  {far_med:.1f} ({len(far_k)} pts)", flush=True)
    print(f"  ratio: {ratio:.1f}×", flush=True)

    return ratio


def measure_blind_prediction(zeros, t_scan_min=1.0, t_scan_max=28.0, n_scan=200):
    """
    블라인드 예측: κ(1+it) 스캔 → 피크 탐지 → 영점 매칭.
    """
    print(f"\n[블라인드 예측] t ∈ [{t_scan_min}, {t_scan_max}], n_scan={n_scan}", flush=True)

    ts = np.linspace(t_scan_min, t_scan_max, n_scan)
    kappas = np.zeros(n_scan)

    for i, t in enumerate(ts):
        s = mpmath.mpc(SIGMA_CRIT, t)
        try:
            k = curvature_at(s)
            kappas[i] = k if np.isfinite(k) else 0
        except Exception:
            kappas[i] = 0
        if (i + 1) % 50 == 0:
            print(f"  ... κ 스캔 {i+1}/{n_scan}", flush=True)

    # 피크 탐지 (scipy)
    from scipy.signal import find_peaks
    threshold = max(np.median(kappas) * 5, 100)
    peaks, _ = find_peaks(kappas, height=threshold, distance=3)
    predicted = ts[peaks]

    print(f"  κ 피크 {len(predicted)}개 (threshold={threshold:.0f})", flush=True)
    for p in predicted:
        print(f"    predicted: t = {p:.2f}, κ = {kappas[np.argmin(np.abs(ts-p))]:.0f}", flush=True)

    # 영점과 매칭 (tolerance = 0.5)
    tol = 0.5
    in_range = zeros[(zeros >= t_scan_min) & (zeros <= t_scan_max)]

    matches = 0
    for p in predicted:
        if len(in_range) > 0 and np.min(np.abs(in_range - p)) < tol:
            matches += 1

    # 역방향: 실제 영점 중 예측에 의해 커버된 비율
    covered = 0
    for z in in_range:
        if len(predicted) > 0 and np.min(np.abs(predicted - z)) < tol:
            covered += 1

    print(f"  실제 영점 (범위 내): {len(in_range)}개", flush=True)
    print(f"  예측→실제 매칭: {matches}/{len(predicted)}", flush=True)
    print(f"  실제→예측 커버: {covered}/{len(in_range)}", flush=True)

    return predicted, in_range, covered


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    t_start = time.time()
    out_path = os.path.expanduser('~/Desktop/gdl_unified/results/elliptic_curve_11a1.txt')
    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    log("=" * 72)
    log("=== Elliptic Curve L-function: 11a1 (GL(2), conductor 11) ===")
    log("=" * 72)
    log(f"Critical line: σ = {SIGMA_CRIT}")
    log(f"Functional equation: Λ(E, 2-s) = ε·Λ(E, s), ε = {EPSILON}")
    log(f"AFE terms: n_max = {N_MAX_COEFF}")
    log(f"mpmath precision: {mpmath.mp.dps} digits")
    log()

    # ══════════════════════════════════════════════════════════════════════
    # [A] aₙ 계수 검증
    # ══════════════════════════════════════════════════════════════════════
    _init_tables()
    an = _an_cache

    log("[A] aₙ 계수 검증 — 11a1 (LMFDB 참조)")
    # LMFDB: 1, -2, -1, 2, 1, 2, -2, 0, -2, -2, 1, -2, 4, 4, -1, -4, -2, 4, 0, 2
    lmfdb_an = [1, -2, -1, 2, 1, 2, -2, 0, -2, -2, 1, -2, 4, 4, -1, -4, -2, 4, 0, 2]
    all_ok = True
    for i, expected in enumerate(lmfdb_an):
        n = i + 1
        actual = an[n]
        ok = "✅" if actual == expected else "❌"
        if actual != expected:
            all_ok = False
        log(f"  a_{n:2d} = {actual:4d} (LMFDB: {expected:4d}) {ok}")
    log(f"  aₙ 검증 (n=1..20): {'전부 PASS ✅' if all_ok else 'FAIL ❌'}")
    log()

    if not all_ok:
        log("⚠️ aₙ 불일치 — 실험 중단")
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines))
        return

    # ══════════════════════════════════════════════════════════════════════
    # [B] AFE 검증: L(E, 2) + 함수 방정식
    # ══════════════════════════════════════════════════════════════════════
    log("[B] AFE 검증")

    # B1: L(E, 2) — 직접합(Re(s)>3/2에서 수렴)과 비교
    L_afe = L_E(mpmath.mpf(2))
    direct_sum = sum(an[n] / n**2 for n in range(1, N_MAX_COEFF + 1))
    L_afe_re = float(mpmath.re(L_afe))
    diff_L2 = abs(L_afe_re - direct_sum)
    log(f"  L(E, 2) AFE    = {L_afe_re:.12f}")
    log(f"  L(E, 2) 직접합 = {direct_sum:.12f}")
    log(f"  차이: {diff_L2:.2e} {'✅' if diff_L2 < 1e-8 else '❌'}")
    log()

    # B2: 중심값 L(E, 1) — LMFDB: ≈ 0.253842...
    L_center = L_E(mpmath.mpc(1, 0))
    L_center_re = float(mpmath.re(L_center))
    log(f"  L(E, 1) = {L_center_re:.10f}")
    log(f"  (LMFDB 참조: ≈ 0.2538418609...)")
    log()

    # B3: 함수 방정식 Λ(s) = εΛ(2-s)
    log("  [함수 방정식 검증] Λ(E, s) = ε·Λ(E, 2-s)")
    test_pts = [
        mpmath.mpc('1.3', '5.0'),
        mpmath.mpc('0.7', '8.0'),
        mpmath.mpc('1.0', '3.0'),
        mpmath.mpc('0.5', '12.0'),
    ]
    fe_ok = True
    for sp in test_pts:
        L1 = Lambda_E(sp)
        L2 = Lambda_E(2 - sp)
        denom = max(float(abs(L1)), 1e-50)
        rel_err = float(abs(L1 - EPSILON * L2)) / denom
        ok = "✅" if rel_err < 1e-10 else "❌"
        if rel_err >= 1e-10:
            fe_ok = False
        log(f"    s={float(mpmath.re(sp)):.1f}+{float(mpmath.im(sp)):.1f}i: "
            f"|Λ(s)-εΛ(2-s)|/|Λ(s)| = {rel_err:.2e} {ok}")
    log(f"  함수 방정식: {'PASS ✅' if fe_ok else 'FAIL ❌'}")
    log()

    if not fe_ok:
        log("⚠️ 함수 방정식 불만족 — AFE 구현 점검 필요")

    # ══════════════════════════════════════════════════════════════════════
    # [C] 영점 탐색
    # ══════════════════════════════════════════════════════════════════════
    zeros = find_zeros_elliptic()

    log()
    log("[C] Zeros — t ∈ [%.1f, %.1f]" % (T_MIN, T_MAX))
    for i, tz in enumerate(zeros):
        log(f"  γ_{i+1} = {tz:.8f}")
    log(f"  Total: {len(zeros)} zeros")
    log()

    if len(zeros) == 0:
        log("⚠️ 영점 0개 — 실험 중단. AFE 또는 영점 탐색 점검 필요.")
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"\n결과 저장: {out_path}", flush=True)
        return

    # LMFDB 교차검증 (가용하면)
    log("[C'] LMFDB 교차검증")
    log(f"  첫 영점 γ₁ = {zeros[0]:.8f}")
    log(f"  (LMFDB 참조: γ₁ ≈ 6.3622... — 정밀값 확인 필요)")
    if len(zeros) >= 2:
        log(f"  γ₂ = {zeros[1]:.8f}")
    log()

    # ══════════════════════════════════════════════════════════════════════
    # [D] 4성질 검증
    # ══════════════════════════════════════════════════════════════════════

    # ── D1: σ-유일성 ──
    sigma_results = measure_sigma_uniqueness(zeros)
    log()
    log("[D1] σ-uniqueness")
    for sigma in sorted(sigma_results):
        jumps = sigma_results[sigma]
        marker = " ← should be maximal" if abs(sigma - SIGMA_CRIT) < 0.01 else ""
        log(f"  σ={sigma:.1f}: jumps = {jumps}{marker}")

    max_sigma = max(sigma_results, key=sigma_results.get)
    off_crit_max = max(v for k, v in sigma_results.items() if abs(k - SIGMA_CRIT) > 0.01)
    on_crit = sigma_results.get(SIGMA_CRIT, 0)
    sigma_pass = abs(max_sigma - SIGMA_CRIT) < 0.01
    log(f"  최대 점프: σ={max_sigma:.1f} (on-crit={on_crit}, off-crit max={off_crit_max})")
    log(f"  판정: {'PASS ✅' if sigma_pass else 'FAIL ❌'}")
    log()

    # ── D2: 모노드로미 (폐곡선 적분 + log-space 보조) ──
    contour_dev, contour_abs = measure_monodromy_contour(zeros)

    log()
    log("[D2a] Monodromy — contour integral (radius=0.3)")
    if contour_abs is not None:
        # 폐곡선 적분: 단순 영점이면 |mono|/π ≈ 2 (winding number = 1)
        log(f"  Mean |mono|/π = {contour_abs:.6f}")
        log(f"  Mean deviation from nearest integer = {contour_dev:.6f}")
        contour_pass = contour_abs > 1.9  # ≈2 기대
        log(f"  (단순 영점 기대값: |mono|/π ≈ 2.0)")
        log(f"  판정: {'PASS ✅' if contour_pass else 'FAIL ❌'} (기준: > 1.9)")
    else:
        contour_pass = False
        log("  계산 실패")

    logspace_dev, logspace_abs = measure_monodromy_logspace(zeros)

    log()
    log("[D2b] Monodromy — log-space arg (보조)")
    if logspace_abs is not None:
        # log-space: Λ 실수, 부호 반전 → |Δarg|/π ≈ 1
        log(f"  Mean |Δarg|/π = {logspace_abs:.6f}")
        log(f"  Mean deviation from 1 = {logspace_dev:.6f}")
        logspace_pass = logspace_abs > 0.95
        log(f"  판정: {'PASS ✅' if logspace_pass else 'FAIL ❌'} (기준: > 0.95)")
    else:
        logspace_pass = False
        log("  계산 실패")

    mono_pass = contour_pass or logspace_pass
    log(f"  [D2 종합] 모노드로미: {'PASS ✅' if mono_pass else 'FAIL ❌'}")
    log()

    # ── D3: κ 집중도 ──
    kappa_ratio = measure_kappa_concentration(zeros)
    log()
    log(f"[D3] κ concentration (δ={DELTA_OFFSET} from σ={SIGMA_CRIT})")
    if kappa_ratio is not None:
        log(f"  κ(near)/κ(far) = {kappa_ratio:.1f}×")
        kappa_pass = kappa_ratio > 10
        log(f"  판정: {'PASS ✅' if kappa_pass else 'FAIL ❌'} (기준: > 10×)")
    else:
        kappa_pass = False
        log("  κ 계산 실패")
    log()

    # ── D4: 블라인드 예측 ──
    predicted, actual_in_range, covered = measure_blind_prediction(zeros)
    log()
    log("[D4] Blind prediction")
    log(f"  Predicted: {[f'{p:.2f}' for p in predicted]}")
    log(f"  Actual (in range): {[f'{z:.2f}' for z in actual_in_range]}")
    log(f"  Covered: {covered}/{len(actual_in_range)} (tol=0.5)")

    if len(actual_in_range) > 0:
        coverage = covered / len(actual_in_range)
        blind_pass = coverage >= 0.7
        log(f"  Coverage: {coverage*100:.0f}%")
        log(f"  판정: {'PASS ✅' if blind_pass else 'FAIL ❌'} (기준: ≥ 70%)")
    else:
        blind_pass = False
        log("  영점 없음")
    log()

    # ══════════════════════════════════════════════════════════════════════
    # [E] 종합 판정
    # ══════════════════════════════════════════════════════════════════════
    log("=" * 72)
    log("[E] 종합 판정")
    log("=" * 72)

    pass_count = 0
    total = 4

    log(f"  1. σ-유일성:    {'PASS ✅' if sigma_pass else 'FAIL ❌'} (σ={max_sigma:.1f} 최대)")
    if sigma_pass:
        pass_count += 1

    log(f"  2. 모노드로미:  {'PASS ✅' if mono_pass else 'FAIL ❌'}")
    if mono_pass:
        pass_count += 1

    log(f"  3. κ 집중도:    {'PASS ✅' if kappa_pass else 'FAIL ❌'}" +
        (f" ({kappa_ratio:.1f}×)" if kappa_ratio else ""))
    if kappa_pass:
        pass_count += 1

    log(f"  4. 블라인드예측: {'PASS ✅' if blind_pass else 'FAIL ❌'}" +
        (f" ({covered}/{len(actual_in_range)})" if len(actual_in_range) > 0 else ""))
    if blind_pass:
        pass_count += 1

    log()
    log(f"  통과: {pass_count}/{total}")

    if pass_count >= 3:
        log(f"  ★ 양성 — GL(2) ξ-다발 프레임워크 성립")
        log(f"  → 프레임워크가 GL(1)에 국한되지 않음 입증")
    elif pass_count >= 2:
        log(f"  ⚠ 조건부 양성 — 추가 검증 필요")
    elif pass_count >= 1:
        log(f"  ⚠ 약한 양성 — 부분적 증거만")
    else:
        log(f"  ❌ 음성 — GL(2) 확장 실패")

    log("=" * 72)

    elapsed = time.time() - t_start
    log(f"\n소요 시간: {elapsed:.0f}초 ({elapsed/60:.1f}분)")

    # 결과 저장
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n결과 저장: {out_path}", flush=True)


if __name__ == '__main__':
    main()
