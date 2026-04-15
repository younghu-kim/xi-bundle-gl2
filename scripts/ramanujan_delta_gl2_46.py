"""
=============================================================================
[Project RDL] 결과 #46 — 라마누잔 Δ L-함수 GL(2) 검증 (weight 12, level 1)
=============================================================================
대상: Ramanujan Δ function — Δ(z) = q∏(1-qⁿ)²⁴
  - Level N = 1, Weight k = 12
  - Root number ε = +1
  - 임계선: σ = k/2 = 6 (⚠️ 타원곡선의 σ=1과 다름!)
  - L(Δ, s) = Σ τ(n) n⁻ˢ, τ = Ramanujan tau function
  - LMFDB: ModularForm/GL2/Q/holomorphic/1/12/a/a

검증 항목 (4성질):
  1. σ-유일성: 위상 점프가 σ=6에서 최대
  2. 모노드로미: 영점 주위 ±π 양자화 (폐곡선 적분)
  3. κ 집중: near(σ=6)/far(σ≠6) >> 1
  4. 블라인드 예측: κ 피크로 영점 위치 예측

핵심 구현:
  - τ(n): q-전개 Δ(q) = q∏(1-qⁿ)²⁴ (정수 연산, 정확)
  - L(Δ,s): AFE (근사 함수 방정식, upper incomplete gamma)
  - Λ(Δ,s) = (2π)⁻ˢ Γ(s) L(Δ,s)  [N=1이므로 √N=1]
  - 함수 방정식: Λ(Δ, 12-s) = Λ(Δ, s)
  - Λ'/Λ = 수치 중앙차분 (h=1e-6)

결과 파일: results/ramanujan_delta_46.txt
=============================================================================
"""

import sys, os, time
import numpy as np
import mpmath
from math import comb

sys.path.insert(0, os.path.expanduser('~/Desktop/gdl_unified'))
sys.path.insert(0, os.path.expanduser('~/Desktop/gdl_unified/scripts'))

mpmath.mp.dps = 80  # 고정밀도 필수 (τ(n) ~ n^{11/2} 크기)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 상수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

N_COND = 1              # Level (conductor)
WEIGHT = 12             # Weight
EPSILON = 1             # Root number +1
SIGMA_CRIT = 6.0        # ⚠️ 임계선 σ = k/2 = 6 (GL(2) weight 12!)
T_MIN, T_MAX = 0.5, 30.0
DELTA_OFFSET = 0.03     # 영점 위 직접 측정 금지 → 오프셋
N_MAX_COEFF = 80        # AFE 항 수 (x_n = 2πn, n=20에서 e^{-126} ≈ 0)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: τ(n) 계수 계산 — q-전개 Δ(q) = q∏(1-qⁿ)²⁴
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_tau_qexpansion(n_max):
    """
    τ(n) via q-expansion: Δ(q) = q · ∏_{m≥1}(1-q^m)²⁴

    정수 연산 (정확), n=1..n_max.
    ∏(1-q^m)²⁴ 계수를 계산한 뒤, Δ = q·product 이므로 τ(n) = product_coeff[n-1].
    """
    N = n_max  # 곱의 계수를 0..n_max-1까지 필요
    coeffs = [0] * N
    coeffs[0] = 1

    for m in range(1, N):
        # (1-q^m)²⁴ = Σ_{k=0}^{24} C(24,k)(-1)^k q^{m·k}
        # 역순으로 처리하면 in-place 가능 (k > 0은 모두 양의 shift)
        # 하지만 binomial 합이므로 새 배열 필요
        new = [0] * N
        for j in range(N):
            if coeffs[j] == 0:
                continue
            for k in range(25):
                idx = j + m * k
                if idx >= N:
                    break
                sign = 1 if k % 2 == 0 else -1
                new[idx] += coeffs[j] * comb(24, k) * sign
        coeffs = new

    # Δ(q) = q · product → τ(n) = coeffs[n-1]
    tau = [0] * (n_max + 1)
    for n in range(1, n_max + 1):
        tau[n] = coeffs[n - 1]

    return tau


def verify_tau_hecke(tau, n_max):
    """Hecke 재귀 검증: τ(p²) = τ(p)² - p¹¹ (good primes, 여기선 모든 소수)"""
    # 소수 체
    sieve = [True] * (n_max + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n_max**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n_max + 1, i):
                sieve[j] = False
    primes = [p for p in range(2, n_max + 1) if sieve[p]]

    ok_count = 0
    fail_count = 0
    for p in primes:
        p2 = p * p
        if p2 > n_max:
            break
        expected = tau[p] * tau[p] - p**11
        actual = tau[p2]
        if actual == expected:
            ok_count += 1
        else:
            fail_count += 1
            print(f"  ⚠️ Hecke 실패: τ({p}²) = {actual}, expected {expected}", flush=True)

    return ok_count, fail_count


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: AFE (Approximate Functional Equation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Λ(Δ,s) = Σ_{n=1}^{n_max} τ(n) · h(n,s)
# h(n,s) = xₙ^{-s} Γ(s, xₙ) + ε · xₙ^{-(k-s)} Γ(k-s, xₙ)
# where xₙ = 2πn/√N = 2πn (N=1), Γ(s,x) = upper incomplete gamma
#
# L(Δ,s) = Λ(Δ,s) / [(√N/2π)^s · Γ(s)] = Λ(Δ,s) / [(2π)^{-s} · Γ(s)]
#
# 수렴: xₙ = 2πn ≈ 6.28n → n=10에서 e^{-62.8} < 10^{-27}
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_tau_cache = None
_precomp_cache = None


def _init_tables():
    """τ(n) + 전처리 테이블 초기화"""
    global _tau_cache, _precomp_cache
    if _tau_cache is not None:
        return

    print("  [초기화] τ(n) 계수 계산 (n ≤ %d)..." % N_MAX_COEFF, flush=True)
    t0 = time.time()
    _tau_cache = compute_tau_qexpansion(N_MAX_COEFF)

    # LMFDB 참조 검증
    lmfdb_tau = {
        1: 1, 2: -24, 3: 252, 4: -1472, 5: 4830,
        6: -6048, 7: -16744, 8: 84480, 9: -113643, 10: -115920,
        11: 534612, 12: -370944, 13: -577738, 14: 401856,
        15: 1217160, 16: 987136, 17: -6905934, 18: 2727432,
        19: 10661420, 20: -7109760
    }
    mismatch = 0
    for n, expected in sorted(lmfdb_tau.items()):
        actual = _tau_cache[n]
        if actual != expected:
            mismatch += 1
            print(f"  ⚠️ τ({n}) = {actual}, LMFDB = {expected}!", flush=True)
    print(f"  τ(n) LMFDB 검증 (n=1..20): {20 - mismatch}/20 일치", flush=True)

    # Hecke 검증
    hok, hfail = verify_tau_hecke(_tau_cache, N_MAX_COEFF)
    print(f"  Hecke 검증: {hok} PASS, {hfail} FAIL", flush=True)

    # 전처리: xₙ 값 + 0이 아닌 항만 필터
    sqrt_N = mpmath.mpf(1)  # √1 = 1
    two_pi = 2 * mpmath.pi
    _precomp_cache = []
    for n in range(1, N_MAX_COEFF + 1):
        if _tau_cache[n] == 0:
            continue
        x_n = two_pi * n / sqrt_N  # = 2πn
        _precomp_cache.append((mpmath.mpf(_tau_cache[n]), x_n))

    print(f"  [초기화] 완료 ({time.time()-t0:.1f}초, 비영 항 {len(_precomp_cache)}개)", flush=True)
    print(f"  τ(2)={_tau_cache[2]}, τ(3)={_tau_cache[3]}, τ(5)={_tau_cache[5]}, "
          f"τ(7)={_tau_cache[7]}, τ(11)={_tau_cache[11]}, τ(13)={_tau_cache[13]}", flush=True)


def Lambda_Delta(s):
    """
    Λ(Δ,s) via AFE:
    Λ(Δ,s) = Σ τ(n) · [xₙ^{-s} Γ(s, xₙ) + ε · xₙ^{-(12-s)} Γ(12-s, xₙ)]

    where xₙ = 2πn (N=1), Γ(s,x) = upper incomplete gamma
    """
    _init_tables()
    s_mp = mpmath.mpc(s)
    s_conj = WEIGHT - s_mp  # 12 - s
    eps = mpmath.mpf(EPSILON)

    result = mpmath.mpc(0)
    for tau_val, x_n in _precomp_cache:
        term1 = mpmath.power(x_n, -s_mp) * mpmath.gammainc(s_mp, x_n)
        term2 = eps * mpmath.power(x_n, -s_conj) * mpmath.gammainc(s_conj, x_n)
        result += tau_val * (term1 + term2)

    return result


def L_Delta(s):
    """L(Δ,s) = Λ(Δ,s) / [(2π)⁻ˢ · Γ(s)]"""
    s_mp = mpmath.mpc(s)
    Lambda_val = Lambda_Delta(s_mp)

    # Λ = (√N/2π)^s Γ(s) L = (2π)^{-s} Γ(s) L  [N=1]
    two_pi = 2 * mpmath.pi
    prefactor = mpmath.power(two_pi, -s_mp) * mpmath.gamma(s_mp)

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

    L0 = Lambda_Delta(s_mp)
    if abs(L0) < mpmath.mpf(10)**(-mpmath.mp.dps + 15):
        return mpmath.mpc(1e10, 0)

    L_plus = Lambda_Delta(s_mp + h)
    L_minus = Lambda_Delta(s_mp - h)

    return (L_plus - L_minus) / (2 * h * L0)


def curvature_at(s):
    """κ(s) = |Λ'/Λ|²"""
    conn = connection_Lambda(s)
    k = float(abs(conn)**2)
    return k if np.isfinite(k) else 1e12


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: 영점 탐색 (σ = 6 임계선)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_zeros_delta(t_min=T_MIN, t_max=T_MAX, n_scan=1500):
    """
    σ=6 임계선에서 Re(Λ) 부호 변화 → findroot 정밀화.
    Λ(Δ, 6+it) is real (∵ ε=+1, 함수 방정식 Λ(12-s)=Λ(s)) → Re(Λ) 부호 변화 = 영점.
    """
    print(f"\n[영점 탐색] σ={SIGMA_CRIT}, t ∈ [{t_min}, {t_max}], n_scan={n_scan}", flush=True)
    ts = np.linspace(t_min, t_max, n_scan)
    zeros = []
    fail_count = 0

    prev_re, prev_t = None, None
    for i, t in enumerate(ts):
        s = mpmath.mpc(SIGMA_CRIT, t)
        try:
            val = Lambda_Delta(s)
            curr_re = float(mpmath.re(val))
        except Exception as e:
            print(f"  WARNING: Lambda_Delta 실패 t={t:.3f}: {e}", flush=True)
            prev_re, prev_t = None, float(t)
            continue

        if prev_re is not None and prev_re * curr_re < 0:
            mid = (prev_t + float(t)) / 2
            try:
                def f_re(t_var):
                    sv = mpmath.mpc(SIGMA_CRIT, mpmath.mpf(t_var))
                    return mpmath.re(Lambda_Delta(sv))

                tz = float(mpmath.findroot(f_re, mpmath.mpf(str(mid))))

                # 확인: |Λ| 충분히 작은지
                sv = mpmath.mpc(SIGMA_CRIT, tz)
                lv = float(abs(Lambda_Delta(sv)))

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
        ts_fine = np.linspace(t_min + 1.0, t_max - 1.0, 500)
        abs_vals = []
        for t in ts_fine:
            try:
                val = abs(Lambda_Delta(mpmath.mpc(SIGMA_CRIT, t)))
                abs_vals.append((float(val), t))
            except Exception as e:
                print(f"  WARNING: fallback 실패 t={t:.2f}: {e}", flush=True)
                abs_vals.append((1e20, t))

        abs_vals.sort()
        for val, t in abs_vals[:20]:
            if val < 1.0:
                try:
                    def f_abs(t_var):
                        sv = mpmath.mpc(SIGMA_CRIT, mpmath.mpf(t_var))
                        lam = Lambda_Delta(sv)
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
    σ=6(임계선)에서 최대 점프 기대.
    ⚠️ weight 12: Γ(s) 진동이 weight 2와 다름 → FAIL 가능 (수학자 예상)
    """
    # σ 범위: 임계선 6 주변 탐색
    sigmas = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
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
                val = Lambda_Delta(s)
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
    모노드로미: 폐곡선 적분 (σ=6 중심, 반지름 radius).
    arg(Λ) 누적 → 영점이 원 안에 있으면 ≈±2π (winding number 1).
    |mono|/π 보고.
    """
    print(f"\n[모노드로미 — 폐곡선 적분] σ_crit={SIGMA_CRIT}, radius={radius}, n_steps={n_steps}", flush=True)

    if len(zeros) == 0:
        return None, None

    monos = []
    individual = []  # 개별 결과 기록

    for tz in zeros:
        center = mpmath.mpc(SIGMA_CRIT, tz)
        phase_accum = mpmath.mpf(0)
        prev_val = None
        ok = True

        for j in range(n_steps + 1):
            theta = 2 * mpmath.pi * j / n_steps
            s = center + radius * mpmath.exp(1j * theta)
            try:
                val = Lambda_Delta(s)
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
        mono_over_pi = mono / np.pi
        individual.append((tz, mono_over_pi))
        print(f"  γ={tz:.6f}: mono = {mono:.4f} rad, mono/π = {mono_over_pi:.6f}", flush=True)

    if not monos:
        return None, None, []

    abs_monos_over_pi = [abs(m) / np.pi for m in monos]
    mean_abs = float(np.mean(abs_monos_over_pi))
    deviations = [abs(m - round(m)) for m in abs_monos_over_pi]
    mean_dev = float(np.mean(deviations))

    return mean_dev, mean_abs, individual


def measure_monodromy_logspace(zeros, eps_offset=0.005):
    """
    모노드로미: log-space arg 보조 측정.
    Λ(Δ, 6+it)는 실수 (ε=+1) → 영점에서 부호 반전 → |Δarg| = π.
    """
    print(f"\n[모노드로미 — log-space arg] eps={eps_offset}", flush=True)

    if len(zeros) == 0:
        return None, None

    monos = []
    individual = []

    for tz in zeros:
        s_plus = mpmath.mpc(SIGMA_CRIT, tz + eps_offset)
        s_minus = mpmath.mpc(SIGMA_CRIT, tz - eps_offset)

        try:
            val_plus = Lambda_Delta(s_plus)
            val_minus = Lambda_Delta(s_minus)

            if abs(val_plus) < 1e-50 or abs(val_minus) < 1e-50:
                continue

            arg_plus = float(mpmath.arg(val_plus))
            arg_minus = float(mpmath.arg(val_minus))
            delta = arg_plus - arg_minus

            monos.append(delta)
            individual.append((tz, abs(delta) / np.pi))
            print(f"  γ={tz:.6f}: Δarg = {delta:.6f}, |Δarg|/π = {abs(delta)/np.pi:.6f}", flush=True)
        except Exception as e:
            print(f"  ⚠ γ={tz:.4f}: {e}", flush=True)

    if not monos:
        return None, None, []

    abs_monos_over_pi = [abs(m) / np.pi for m in monos]
    mean_abs = float(np.mean(abs_monos_over_pi))
    deviations = [abs(abs(m) / np.pi - 1.0) for m in monos]
    mean_dev = float(np.mean(deviations))

    return mean_dev, mean_abs, individual


def measure_kappa_concentration(zeros, n_generic=40):
    """
    κ 집중도: median(κ near zeros) / median(κ far from zeros).
    GL(2) 임계선 σ=6, 오프셋 δ=0.03.
    원시값 기록 (검토자 피드백 반영).
    """
    print(f"\n[κ 집중도] σ_crit={SIGMA_CRIT}, δ={DELTA_OFFSET}", flush=True)

    if len(zeros) == 0:
        return None, [], []

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
        return None, near_k, far_k

    near_med = float(np.median(near_k))
    far_med = float(np.median(far_k))
    ratio = near_med / far_med if far_med > 0 else float('inf')

    print(f"\n  near: {len(near_k)} pts, range [{min(near_k):.1f}, {max(near_k):.1f}], median = {near_med:.1f}", flush=True)
    print(f"  far:  {len(far_k)} pts, range [{min(far_k):.1f}, {max(far_k):.1f}], median = {far_med:.1f}", flush=True)
    print(f"  ratio: {ratio:.1f}×", flush=True)

    return ratio, near_k, far_k


def measure_blind_prediction(zeros, t_scan_min=1.0, t_scan_max=28.0, n_scan=200):
    """
    블라인드 예측: κ(6+it) 스캔 → 피크 탐지 → 영점 매칭.
    """
    print(f"\n[블라인드 예측] σ={SIGMA_CRIT}, t ∈ [{t_scan_min}, {t_scan_max}], n_scan={n_scan}", flush=True)

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
    out_path = os.path.expanduser('~/Desktop/gdl_unified/results/ramanujan_delta_46.txt')
    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    log("=" * 72)
    log("=== Ramanujan Δ L-function (GL(2), weight 12, level 1) ===")
    log("=" * 72)
    log(f"L-function: L(Δ, s) = Σ τ(n) n⁻ˢ")
    log(f"Critical line: σ = {SIGMA_CRIT} (= k/2 = 12/2)")
    log(f"Functional equation: Λ(Δ, 12-s) = ε·Λ(Δ, s), ε = {EPSILON}")
    log(f"AFE terms: n_max = {N_MAX_COEFF}")
    log(f"mpmath precision: {mpmath.mp.dps} digits")
    log()

    # ══════════════════════════════════════════════════════════════════════
    # [A] τ(n) 계수 검증
    # ══════════════════════════════════════════════════════════════════════
    _init_tables()
    tau = _tau_cache

    log("[A] τ(n) 계수 검증 — Ramanujan tau (LMFDB 참조)")
    lmfdb_tau = {
        1: 1, 2: -24, 3: 252, 4: -1472, 5: 4830,
        6: -6048, 7: -16744, 8: 84480, 9: -113643, 10: -115920,
        11: 534612, 12: -370944, 13: -577738, 14: 401856,
        15: 1217160, 16: 987136, 17: -6905934, 18: 2727432,
        19: 10661420, 20: -7109760
    }
    all_ok = True
    for n in range(1, 21):
        expected = lmfdb_tau[n]
        actual = tau[n]
        ok = "✅" if actual == expected else "❌"
        if actual != expected:
            all_ok = False
        log(f"  τ({n:2d}) = {actual:>12d} (LMFDB: {expected:>12d}) {ok}")
    log(f"  τ(n) 검증 (n=1..20): {'전부 PASS ✅' if all_ok else 'FAIL ❌'}")
    log()

    # Hecke 검증
    hok, hfail = verify_tau_hecke(tau, N_MAX_COEFF)
    log(f"  Hecke 재귀 검증 (τ(p²)=τ(p)²-p¹¹): {hok} PASS, {hfail} FAIL")
    log(f"  Deligne 경계: |τ(p)| ≤ 2p^{{11/2}}")
    # 일부 소수에 대해 Deligne 확인
    for p in [2, 3, 5, 7, 11, 13]:
        bound = 2 * p**(11/2)
        actual = abs(tau[p])
        ok = "✅" if actual <= bound + 1 else "❌"  # +1 for float imprecision
        log(f"    p={p:2d}: |τ(p)| = {actual:>12d}, 2p^{{11/2}} = {bound:.0f} {ok}")
    log()

    if not all_ok:
        log("⚠️ τ(n) 불일치 — 실험 중단")
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines))
        return

    # ══════════════════════════════════════════════════════════════════════
    # [B] AFE 검증: L(Δ, 7) 직접합 비교 + 함수 방정식
    # ══════════════════════════════════════════════════════════════════════
    log("[B] AFE 검증")

    # B1: L(Δ, 7) — 직접합(Re(s) > 13/2 = 6.5에서 절대수렴) 비교
    # s=7 > 6.5이므로 직접합 수렴
    L_afe = L_Delta(mpmath.mpf(7))
    direct_sum = mpmath.nsum(lambda n: mpmath.mpf(_tau_cache[int(n)]) / mpmath.power(n, 7),
                             [1, N_MAX_COEFF])
    L_afe_re = float(mpmath.re(L_afe))
    direct_sum_f = float(direct_sum)
    diff_L7 = abs(L_afe_re - direct_sum_f)
    log(f"  L(Δ, 7) AFE    = {L_afe_re:.12f}")
    log(f"  L(Δ, 7) 직접합 = {direct_sum_f:.12f}")
    log(f"  차이: {diff_L7:.2e} {'✅' if diff_L7 < 1e-6 else '❌'}")
    log()

    # B2: 중심값 L(Δ, 6) — 임계선 위의 값
    L_center = L_Delta(mpmath.mpc(6, 0))
    L_center_re = float(mpmath.re(L_center))
    log(f"  L(Δ, 6) = {L_center_re:.10f}")
    log(f"  (LMFDB 참조: 확인 필요)")
    log()

    # B3: 함수 방정식 Λ(Δ,s) = Λ(Δ,12-s) (ε=+1)
    log("  [함수 방정식 검증] Λ(Δ, s) = ε·Λ(Δ, 12-s)")
    test_pts = [
        mpmath.mpc('7.3', '5.0'),
        mpmath.mpc('4.7', '8.0'),
        mpmath.mpc('6.0', '3.0'),
        mpmath.mpc('5.5', '12.0'),
    ]
    fe_ok = True
    for sp in test_pts:
        L1 = Lambda_Delta(sp)
        L2 = Lambda_Delta(WEIGHT - sp)  # 12 - s
        denom = max(float(abs(L1)), 1e-50)
        rel_err = float(abs(L1 - EPSILON * L2)) / denom
        ok = "✅" if rel_err < 1e-10 else "❌"
        if rel_err >= 1e-10:
            fe_ok = False
        sp_re = float(mpmath.re(sp))
        sp_im = float(mpmath.im(sp))
        log(f"    s={sp_re:.1f}+{sp_im:.1f}i: "
            f"|Λ(s)-εΛ(12-s)|/|Λ(s)| = {rel_err:.2e} {ok}")
    log(f"  함수 방정식: {'PASS ✅' if fe_ok else 'FAIL ❌'}")
    log()

    if not fe_ok:
        log("⚠️ 함수 방정식 불만족 — AFE 구현 점검 필요")

    # B4: Λ(Δ, 6+it) 실수 확인 (ε=+1)
    log("  [Λ 실수성 확인] Λ(Δ, 6+it)가 실수인지 (ε=+1)")
    for t_test in [3.0, 7.0, 15.0]:
        s_test = mpmath.mpc(SIGMA_CRIT, t_test)
        val = Lambda_Delta(s_test)
        re_part = float(mpmath.re(val))
        im_part = float(mpmath.im(val))
        ratio = abs(im_part) / max(abs(re_part), 1e-50)
        ok = "✅" if ratio < 1e-10 else "❌"
        log(f"    t={t_test}: Re={re_part:.6e}, Im={im_part:.6e}, |Im/Re|={ratio:.2e} {ok}")
    log()

    # ══════════════════════════════════════════════════════════════════════
    # [C] 영점 탐색
    # ══════════════════════════════════════════════════════════════════════
    zeros = find_zeros_delta()

    log()
    log("[C] Zeros — σ=%.1f, t ∈ [%.1f, %.1f]" % (SIGMA_CRIT, T_MIN, T_MAX))
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

    # LMFDB 교차검증
    log("[C'] LMFDB 교차검증")
    log(f"  첫 영점 γ₁ = {zeros[0]:.8f}")
    log(f"  (LMFDB 참조: γ₁ ≈ 9.2224 — 대수적/해석적 정규화 동일)")
    if abs(zeros[0] - 9.2224) < 0.1:
        log(f"  γ₁ 교차검증: ✅ PASS (Δ < 0.1)")
    else:
        log(f"  γ₁ 교차검증: ❌ (기대값 ~9.22, 실제 {zeros[0]:.4f})")
    if len(zeros) >= 2:
        log(f"  γ₂ = {zeros[1]:.8f}")
    log()

    # ══════════════════════════════════════════════════════════════════════
    # [D] 4성질 검증
    # ══════════════════════════════════════════════════════════════════════

    # ── D1: σ-유일성 ──
    sigma_results = measure_sigma_uniqueness(zeros)
    log()
    log("[D1] σ-uniqueness (σ-유일성)")
    for sigma in sorted(sigma_results):
        jumps = sigma_results[sigma]
        marker = " ← 임계선" if abs(sigma - SIGMA_CRIT) < 0.01 else ""
        log(f"  σ={sigma:.1f}: jumps = {jumps}{marker}")

    max_sigma = max(sigma_results, key=sigma_results.get)
    off_crit_max = max(v for k, v in sigma_results.items() if abs(k - SIGMA_CRIT) > 0.01)
    on_crit = sigma_results.get(SIGMA_CRIT, 0)
    sigma_pass = abs(max_sigma - SIGMA_CRIT) < 0.01
    log(f"  최대 점프: σ={max_sigma:.1f} (on-crit={on_crit}, off-crit max={off_crit_max})")
    log(f"  판정: {'PASS ✅' if sigma_pass else 'FAIL ❌'}")
    log()

    # ── D2: 모노드로미 (폐곡선 적분 + log-space 보조) ──
    contour_result = measure_monodromy_contour(zeros)
    contour_dev, contour_abs = contour_result[0], contour_result[1]
    contour_individual = contour_result[2] if len(contour_result) > 2 else []

    log()
    log("[D2a] Monodromy — contour integral (radius=0.3)")
    if contour_abs is not None:
        log(f"  Mean |mono|/π = {contour_abs:.6f}")
        log(f"  Mean deviation from nearest integer = {contour_dev:.6f}")
        contour_pass = contour_abs > 1.9
        log(f"  (단순 영점 기대값: |mono|/π ≈ 2.0)")

        # 개별 결과 기록
        n_exact = sum(1 for _, m in contour_individual if abs(abs(m) - 2.0) < 0.01)
        n_total = len(contour_individual)
        log(f"  정수 정확: {n_exact}/{n_total} (|mono/π - 2| < 0.01)")
        log(f"  판정: {'PASS ✅' if contour_pass else 'FAIL ❌'} (기준: > 1.9)")
    else:
        contour_pass = False
        log("  계산 실패")

    logspace_result = measure_monodromy_logspace(zeros)
    logspace_dev, logspace_abs = logspace_result[0], logspace_result[1]
    logspace_individual = logspace_result[2] if len(logspace_result) > 2 else []

    log()
    log("[D2b] Monodromy — log-space arg (보조)")
    if logspace_abs is not None:
        log(f"  Mean |Δarg|/π = {logspace_abs:.6f}")
        log(f"  Mean deviation from 1 = {logspace_dev:.6f}")
        logspace_pass = logspace_abs > 0.95

        n_exact_ls = sum(1 for _, m in logspace_individual if abs(m - 1.0) < 0.01)
        n_total_ls = len(logspace_individual)
        log(f"  정수 정확: {n_exact_ls}/{n_total_ls} (|Δarg/π - 1| < 0.01)")
        log(f"  판정: {'PASS ✅' if logspace_pass else 'FAIL ❌'} (기준: > 0.95)")
    else:
        logspace_pass = False
        log("  계산 실패")

    mono_pass = contour_pass or logspace_pass
    log(f"  [D2 종합] 모노드로미: {'PASS ✅' if mono_pass else 'FAIL ❌'}")
    log()

    # ── D3: κ 집중도 ──
    kappa_result = measure_kappa_concentration(zeros)
    kappa_ratio = kappa_result[0]
    near_k = kappa_result[1] if len(kappa_result) > 1 else []
    far_k = kappa_result[2] if len(kappa_result) > 2 else []

    log()
    log(f"[D3] κ concentration (δ={DELTA_OFFSET} from σ={SIGMA_CRIT})")
    if kappa_ratio is not None:
        near_med = float(np.median(near_k))
        far_med = float(np.median(far_k))
        log(f"  near: {len(near_k)} pts, range [{min(near_k):.1f}, {max(near_k):.1f}], median = {near_med:.1f}")
        log(f"  far:  {len(far_k)} pts, range [{min(far_k):.1f}, {max(far_k):.1f}], median = {far_med:.1f}")
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
        log(f"  ★ 양성 — Ramanujan Δ GL(2) weight 12 검증 성공")
        log(f"  → GL(2) 프레임워크가 weight에 무관 → 진정한 보편성")
    elif pass_count >= 2:
        log(f"  ⚠ 조건부 양성 — 추가 검증 필요")
    elif pass_count >= 1:
        log(f"  ⚠ 약한 양성 — 부분적 증거만")
    else:
        log(f"  ❌ 음성 — weight 12에서 프레임워크 실패")

    log()

    # ── 비교표: 11a1 vs 37a1 vs Δ ──
    log("=" * 72)
    log("[F] GL(2) 비교표: 11a1 vs 37a1 vs Ramanujan Δ")
    log("=" * 72)
    log(f"  | 항목        | 11a1 (wt 2) | 37a1 (wt 2) | Δ (wt 12)  |")
    log(f"  |-------------|-------------|-------------|------------|")
    log(f"  | conductor   | 11          | 37          | 1          |")
    log(f"  | weight      | 2           | 2           | 12         |")
    log(f"  | rank        | 0           | 1           | 0 (N/A)    |")
    log(f"  | root number | +1          | -1          | +1         |")
    log(f"  | σ_crit      | 1           | 1           | 6          |")
    log(f"  | σ-유일성    | FAIL        | FAIL        | {'PASS' if sigma_pass else 'FAIL'}        |")
    log(f"  | 모노드로미  | PASS        | PASS        | {'PASS' if mono_pass else 'FAIL'}        |")
    kappa_str = f"{kappa_ratio:.0f}×" if kappa_ratio else "N/A"
    log(f"  | κ 집중도    | 972×        | 2684×       | {kappa_str:<10s} |")
    cov_str = f"{covered}/{len(actual_in_range)}" if len(actual_in_range) > 0 else "N/A"
    log(f"  | 블라인드    | 15/15       | 20/20       | {cov_str:<10s} |")
    log(f"  | 통과        | 3/4         | 3/4         | {pass_count}/4       |")

    log("=" * 72)

    elapsed = time.time() - t_start
    log(f"\n소요 시간: {elapsed:.0f}초 ({elapsed/60:.1f}분)")

    # 결과 저장
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n결과 저장: {out_path}", flush=True)


if __name__ == '__main__':
    main()
