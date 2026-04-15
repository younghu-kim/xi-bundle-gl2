"""
=============================================================================
[Project RDL] 결과 #45 — 타원곡선 L-함수 ξ-다발 검증 (GL(2) 두 번째 사례)
=============================================================================
대상: 37a1 (y² + y = x³ - x)
  - conductor N = 37, rank 1, root number ε = -1
  - weight k = 2, newform on Γ₀(37)
  - LMFDB: https://www.lmfdb.org/EllipticCurve/Q/37/a/1

임계선: σ = 1 (GL(2))

핵심 차이 (11a1 대비):
  - ε = -1 → Λ(2-s) = -Λ(s) (반대칭)
  - rank 1 → L(E,1) = 0 (강제 영점, s=1)
  - ε=-1 → Λ(1+it) 순허수 (Re=0) → Im(Λ) 부호 변화로 영점 탐색
  - conductor 37 (11보다 큼)

검증 항목 (4성질):
  1. σ-유일성: GL(2) 구조상 FAIL 예상 (11a1 패턴 재현)
  2. 모노드로미: 영점 주위 ±π 양자화 (폐곡선 적분)
  3. κ 집중: near(σ=1)/far(σ≠1) >> 1
  4. 블라인드 예측: κ 피크로 영점 위치 예측

결과 파일: results/elliptic_curve_37a1.txt
=============================================================================
"""

import sys, os, time
import numpy as np
import mpmath

sys.path.insert(0, os.path.expanduser('~/Desktop/gdl_unified'))
sys.path.insert(0, os.path.expanduser('~/Desktop/gdl_unified/scripts'))

mpmath.mp.dps = 80

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 상수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

N_COND = 37           # conductor
EPSILON = -1          # root number -1 (⚠️ 11a1의 +1과 반대!)
SIGMA_CRIT = 1.0      # 임계선 σ = 1 (GL(2))
T_MIN, T_MAX = 0.01, 30.0   # t=0 강제 영점 포함
DELTA_OFFSET = 0.03   # 영점 위 직접 측정 금지 → 오프셋
N_MAX_COEFF = 100     # AFE 항 수 (N=37, x_n = 2πn/√37 ≈ 1.033n → 더 많은 항 필요)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: aₙ 계수 — 37a1 (y² + y = x³ - x) 점 세기 + multiplicative 재귀
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_ap_37a1(p):
    """
    aₚ for 37a1: y² + y = x³ - x

    p=37 (bad prime, nonsplit multiplicative): a₃₇ = -1
    p=2: 직접 열거
    p≠2,37 (good odd prime): Legendre 기호
        변환: (2y+1)² = 4x³ - 4x + 1
        disc(x) = 4x³ - 4x + 1
        aₚ = -Σ_{x=0}^{p-1} legendre(disc(x), p)
    """
    if p == 37:
        return -1  # nonsplit multiplicative reduction (ε=-1)

    if p == 2:
        # 직접 열거: y² + y = x³ - x (mod 2)
        count_pts = 1  # point at infinity
        for x in range(2):
            for y in range(2):
                if (y * y + y - x * x * x + x) % 2 == 0:
                    count_pts += 1
        return p + 1 - count_pts

    # 홀수 good prime: Legendre 기호
    # disc(x) = 4x³ - 4x + 1
    affine_count = 0
    for x in range(p):
        disc = (4 * x * x * x - 4 * x + 1) % p
        if disc == 0:
            affine_count += 1
        else:
            leg = pow(disc, (p - 1) // 2, p)
            if leg == 1:
                affine_count += 2

    return p - affine_count


def compute_an_table(n_max):
    """aₙ (n=1..n_max) via multiplicativity."""
    sieve = [True] * (n_max + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n_max**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n_max + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, n_max + 1) if sieve[i]]

    ap = {}
    for p in primes:
        ap[p] = compute_ap_37a1(p)

    apk = {}
    for p in primes:
        apk[(p, 0)] = 1
        apk[(p, 1)] = ap[p]
        pk = p
        k = 1
        while pk * p <= n_max:
            pk *= p
            k += 1
            if p == 37:
                # bad prime: a_{p^k} = aₚ^k
                apk[(p, k)] = ap[p] ** k
            else:
                # good prime: a_{p^k} = aₚ·a_{p^{k-1}} - p·a_{p^{k-2}}
                apk[(p, k)] = ap[p] * apk[(p, k - 1)] - p * apk[(p, k - 2)]

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
# Step 2: AFE (Approximate Functional Equation) — ε = -1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Λ(E,s) = Σ aₙ · [xₙ^{-s} Γ(s, xₙ) + ε · xₙ^{-(2-s)} Γ(2-s, xₙ)]
# where xₙ = 2πn/√N, Γ(s,x) = upper incomplete gamma
# ε = -1 → 두 번째 항의 부호 반전!
#
# 수렴: xₙ = 2πn/√37 ≈ 1.033n → n=100에서 e^{-103} < 10^{-45}

_an_cache = None
_precomp_cache = None


def _init_tables():
    """aₙ + 전처리 테이블 초기화"""
    global _an_cache, _precomp_cache
    if _an_cache is not None:
        return

    print("  [초기화] aₙ 계수 계산 (n ≤ %d)..." % N_MAX_COEFF, flush=True)
    t0 = time.time()
    _an_cache = compute_an_table(N_MAX_COEFF)

    sqrt_N = mpmath.sqrt(mpmath.mpf(N_COND))
    two_pi = 2 * mpmath.pi
    _precomp_cache = []
    for n in range(1, N_MAX_COEFF + 1):
        if _an_cache[n] == 0:
            continue
        x_n = two_pi * n / sqrt_N
        _precomp_cache.append((mpmath.mpf(_an_cache[n]), x_n))

    print(f"  [초기화] 완료 ({time.time()-t0:.1f}초, 비영 항 {len(_precomp_cache)}개)", flush=True)

    # 검증: LMFDB 참조 aₙ (n=1..20)
    # 37a1: 1, -2, -3, 2, -2, 6, -1, 0, 6, 4, -5, -6, -2, 2, 6, -4, 0, -12, 0, -4
    lmfdb_ref = {1: 1, 2: -2, 3: -3, 4: 2, 5: -2, 6: 6, 7: -1, 8: 0,
                 9: 6, 10: 4, 11: -5, 12: -6, 13: -2, 14: 2, 15: 6,
                 16: -4, 17: 0, 18: -12, 19: 0, 20: -4}
    mismatch = False
    for n, expected in lmfdb_ref.items():
        actual = _an_cache[n]
        if actual != expected:
            print(f"  ⚠️ a_{n} = {actual}, expected {expected}!", flush=True)
            mismatch = True
    if not mismatch:
        print(f"  ✅ aₙ (n=1..20) LMFDB 참조값 전부 일치", flush=True)
    print(f"  a₂={_an_cache[2]}, a₃={_an_cache[3]}, a₅={_an_cache[5]}, "
          f"a₇={_an_cache[7]}, a₃₇={_an_cache[37]}", flush=True)


def Lambda_E(s):
    """
    Λ(E,s) via AFE:
    Λ(E,s) = Σ aₙ · [xₙ^{-s} Γ(s, xₙ) + ε · xₙ^{-(2-s)} Γ(2-s, xₙ)]
    ε = -1 !
    """
    _init_tables()
    s_mp = mpmath.mpc(s)
    s_conj = 2 - s_mp
    eps = mpmath.mpf(EPSILON)  # -1

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
# Step 4: 영점 탐색 (σ = 1, ε = -1 → Im(Λ) 부호 변화)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ε=-1, Schwarz reflection → Λ(1+it) 순허수!
# Λ(2-s) = -Λ(s) + Λ(conj(s)) = conj(Λ(s))
# → conj(Λ(1+it)) = Λ(1-it) = -Λ(1+it) → Re(Λ(1+it)) = 0
# 영점: Im(Λ(1+it)) = 0 → Im(Λ) 부호 변화 스캔

def find_zeros_elliptic(t_min=T_MIN, t_max=T_MAX, n_scan=1500):
    """
    σ=1 임계선에서 Im(Λ) 부호 변화 → findroot 정밀화.
    ε=-1이므로 Λ(1+it) 순허수 → Im(Λ) 부호 변화 = 영점.
    """
    print(f"\n[영점 탐색] t ∈ [{t_min}, {t_max}], n_scan={n_scan}", flush=True)
    print(f"  ε=-1 → Λ(1+it) 순허수 → Im(Λ) 부호 변화로 탐색", flush=True)

    ts = np.linspace(t_min, t_max, n_scan)
    zeros = []
    fail_count = 0

    prev_im, prev_t = None, None
    for i, t in enumerate(ts):
        s = mpmath.mpc(SIGMA_CRIT, t)
        try:
            val = Lambda_E(s)
            curr_im = float(mpmath.im(val))
            # ε=-1 검증: Re 부분이 실제로 작은지 확인 (처음 몇 점)
            if i < 3:
                curr_re = float(mpmath.re(val))
                print(f"  [검증] t={t:.4f}: Re(Λ)={curr_re:.4e}, Im(Λ)={curr_im:.4e} "
                      f"→ |Re/Im|={abs(curr_re/curr_im) if abs(curr_im)>1e-50 else 'N/A'}", flush=True)
        except Exception as e:
            print(f"  WARNING: Lambda_E 실패 t={t:.3f}: {e}", flush=True)
            prev_im, prev_t = None, float(t)
            continue

        if prev_im is not None and prev_im * curr_im < 0:
            mid = (prev_t + float(t)) / 2
            try:
                def f_im(t_var):
                    sv = mpmath.mpc(SIGMA_CRIT, mpmath.mpf(t_var))
                    return mpmath.im(Lambda_E(sv))

                tz = float(mpmath.findroot(f_im, mpmath.mpf(str(mid))))

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

        prev_im, prev_t = curr_im, float(t)

        if (i + 1) % 500 == 0:
            elapsed_frac = (i + 1) / n_scan
            print(f"  ... 스캔 {i+1}/{n_scan} ({elapsed_frac*100:.0f}%, 영점 {len(zeros)}개)", flush=True)

    # 강제 영점 t=0 확인
    print(f"\n  [강제 영점 확인] t=0 (s=1, rank 1 → L(E,1)=0)", flush=True)
    try:
        s_center = mpmath.mpc(SIGMA_CRIT, 0)
        L_center = Lambda_E(s_center)
        print(f"    Λ(E, 1) = {float(mpmath.re(L_center)):.4e} + {float(mpmath.im(L_center)):.4e}i", flush=True)
        print(f"    |Λ(E, 1)| = {float(abs(L_center)):.4e}", flush=True)
        if float(abs(L_center)) < 1e-10:
            print(f"    ✅ 강제 영점 확인: |Λ(E,1)| < 1e-10", flush=True)
            # t=0 영점은 구조적 — 영점 목록에 추가하되 별도 표기
            if not zeros or abs(zeros[0]) > 0.05:
                zeros.insert(0, 0.0)
                print(f"    γ₀ = 0.00000000 (강제 영점, rank 1)", flush=True)
    except Exception as e:
        print(f"    WARNING: 강제 영점 확인 실패: {e}", flush=True)

    if len(zeros) == 0:
        print("  ⚠️ 영점 0개 — Im(Λ) 스캔 실패, |Λ| 최소화 시도...", flush=True)
        ts_fine = np.linspace(0.5, t_max - 1.0, 500)
        abs_vals = []
        for t in ts_fine:
            try:
                val = abs(Lambda_E(mpmath.mpc(SIGMA_CRIT, t)))
                abs_vals.append((float(val), t))
            except Exception:
                abs_vals.append((1e20, t))

        abs_vals.sort()
        for val, t in abs_vals[:20]:
            if val < 1.0:
                try:
                    def f_abs(t_var):
                        sv = mpmath.mpc(SIGMA_CRIT, mpmath.mpf(t_var))
                        return mpmath.im(Lambda_E(sv))

                    tz = float(mpmath.findroot(f_abs, mpmath.mpf(str(t))))
                    if not any(abs(tz - z) < 0.05 for z in zeros):
                        zeros.append(tz)
                        print(f"  ✓ (fallback) γ = {tz:.8f}", flush=True)
                except Exception:
                    pass

    if len(zeros) == 0:
        print("  ⚠️⚠️ 영점 완전 실패!", flush=True)

    if fail_count > 0:
        print(f"  findroot 실패: {fail_count}회", flush=True)

    zeros.sort()
    print(f"  총 {len(zeros)}개 영점 발견 (강제 영점 포함)", flush=True)
    return np.array(zeros)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 5: 4성질 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def measure_sigma_uniqueness(zeros, n_t_scan=300):
    """
    σ-유일성: 각 σ에서 위상 점프 카운트.
    ε=-1 → Λ(σ+it)는 σ≠1에서는 일반 복소수.
    |arg(Λ)| 점프 카운트 (부호 변화 — Re 또는 Im).
    """
    sigmas = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    print(f"\n[σ-유일성] σ = {sigmas}, n_t_scan = {n_t_scan}", flush=True)
    results = {}

    for sigma in sigmas:
        t0 = time.time()
        ts = np.linspace(0.5, T_MAX - 0.5, n_t_scan)
        jumps = 0
        prev_val = None

        for t in ts:
            s = mpmath.mpc(sigma, t)
            try:
                val = Lambda_E(s)
                if abs(sigma - SIGMA_CRIT) < 0.01:
                    # 임계선: Im(Λ) 부호 변화 (Λ 순허수)
                    curr = float(mpmath.im(val))
                else:
                    # off-critical: Re(Λ) 부호 변화 (GL(2) 비교용)
                    curr = float(mpmath.re(val))
            except Exception:
                prev_val = None
                continue

            if prev_val is not None and prev_val * curr < 0:
                jumps += 1
            prev_val = curr

        results[sigma] = jumps
        marker = " ← 임계선" if abs(sigma - SIGMA_CRIT) < 0.01 else ""
        print(f"  σ={sigma:.1f}: jumps = {jumps}{marker}  ({time.time()-t0:.0f}초)", flush=True)

    return results


def measure_monodromy_contour(zeros, radius=0.3, n_steps=48):
    """
    모노드로미: 폐곡선 적분 (σ=1 중심, 반지름 radius).
    arg(Λ) 누적 → 영점이 원 안에 있으면 ≈±2π (winding number 1).
    """
    print(f"\n[모노드로미 — 폐곡선 적분] radius={radius}, n_steps={n_steps}", flush=True)

    # 강제 영점 t=0 제외 (t 근방에서 수치 안정성 문제)
    nontrivial = [tz for tz in zeros if abs(tz) > 0.5]
    if len(nontrivial) == 0:
        print("  ⚠ 비자명 영점 없음", flush=True)
        return None, None

    print(f"  비자명 영점 {len(nontrivial)}개 (강제 영점 t≈0 제외)", flush=True)

    monos = []
    for tz in nontrivial:
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
    mean_abs = float(np.mean(abs_monos_over_pi))
    deviations = [abs(m - round(m)) for m in abs_monos_over_pi]
    mean_dev = float(np.mean(deviations))

    return mean_dev, mean_abs


def measure_monodromy_forced_zero(radius=0.2, n_steps=64):
    """
    강제 영점 (t=0, s=1) 전용 모노드로미.
    s=1에서의 영점이 단순 영점인지 확인 (rank 1 → 단순).
    반경 0.2, 더 세밀한 64단계.
    """
    print(f"\n[모노드로미 — 강제 영점 t=0] radius={radius}, n_steps={n_steps}", flush=True)

    center = mpmath.mpc(SIGMA_CRIT, 0)
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
        except Exception as e:
            print(f"  ⚠ θ={float(theta):.3f}: {e}", flush=True)
            ok = False
            break

        if prev_val is not None:
            ratio = val / prev_val
            phase_accum += mpmath.im(mpmath.log(ratio))
        prev_val = val

    if not ok:
        print("  ⚠ 강제 영점 모노드로미 계산 실패", flush=True)
        return None

    mono = float(phase_accum)
    print(f"  s=1 (강제 영점): mono = {mono:.6f} rad, mono/π = {mono/np.pi:.6f}", flush=True)
    print(f"  (rank 1 → 단순 영점 → winding number = 1 → |mono|/π ≈ 2 기대)", flush=True)
    return mono


def measure_monodromy_logspace(zeros, eps_offset=0.005):
    """
    모노드로미: log-space arg 보조 측정.
    ε=-1 → Λ(1+it) 순허수 → arg = ±π/2.
    영점에서 arg 점프: π/2 → -π/2 (or vice versa) → |Δarg| = π.
    """
    print(f"\n[모노드로미 — log-space arg] eps={eps_offset}", flush=True)

    # 강제 영점 제외
    nontrivial = [tz for tz in zeros if abs(tz) > 0.5]
    if len(nontrivial) == 0:
        return None, None

    monos = []
    for tz in nontrivial:
        s_plus = mpmath.mpc(SIGMA_CRIT, tz + eps_offset)
        s_minus = mpmath.mpc(SIGMA_CRIT, tz - eps_offset)

        try:
            val_plus = Lambda_E(s_plus)
            val_minus = Lambda_E(s_minus)

            if abs(val_plus) < 1e-50 or abs(val_minus) < 1e-50:
                continue

            arg_plus = float(mpmath.arg(val_plus))
            arg_minus = float(mpmath.arg(val_minus))
            delta = arg_plus - arg_minus

            # ε=-1: Λ 순허수 → arg ∈ {π/2, -π/2} → 부호 변화 시 |Δarg| = π
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
    #44 검토자 피드백: near/far 원시 median 값 명시.
    """
    print(f"\n[κ 집중도] δ={DELTA_OFFSET}", flush=True)

    # 강제 영점 제외 (t=0은 κ 발산)
    nontrivial = [tz for tz in zeros if abs(tz) > 0.5]
    if len(nontrivial) == 0:
        print("  ⚠ 비자명 영점 없음", flush=True)
        return None, None, None

    # Near: 각 영점에서 δ 오프셋
    near_k = []
    for tz in nontrivial:
        s = mpmath.mpc(SIGMA_CRIT, tz + DELTA_OFFSET)
        try:
            k = curvature_at(s)
            if np.isfinite(k) and k < 1e11:
                near_k.append(k)
                print(f"    near γ={tz:.4f}: κ = {k:.1f}", flush=True)
        except Exception as e:
            print(f"  WARNING: near κ 실패 t={tz:.2f}: {e}", flush=True)

    # Far: 영점에서 1.0 이상 떨어진 점
    far_k = []
    rng = np.random.RandomState(42)
    attempts = 0
    while len(far_k) < n_generic and attempts < n_generic * 5:
        t = rng.uniform(1.0, T_MAX - 1.0)
        # 모든 영점(강제 포함)에서 1.0 이상 떨어져야
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
        return None, None, None

    near_med = float(np.median(near_k))
    far_med = float(np.median(far_k))
    ratio = near_med / far_med if far_med > 0 else float('inf')

    # 원시값 명시 (#44 검토자 피드백)
    print(f"  near: {len(near_k)} pts, range [{min(near_k):.1f}, {max(near_k):.1f}]", flush=True)
    print(f"  near median: {near_med:.1f}", flush=True)
    print(f"  far:  {len(far_k)} pts, range [{min(far_k):.1f}, {max(far_k):.1f}]", flush=True)
    print(f"  far median:  {far_med:.1f}", flush=True)
    print(f"  ratio: {ratio:.1f}×", flush=True)

    return ratio, near_med, far_med


def measure_blind_prediction(zeros, t_scan_min=1.0, t_scan_max=28.0, n_scan=200):
    """
    블라인드 예측: κ(1+it) 스캔 → 피크 탐지 → 영점 매칭.
    ⚠️ 강제 영점(t≈0)은 블라인드 예측 대상에서 제외.
    """
    print(f"\n[블라인드 예측] t ∈ [{t_scan_min}, {t_scan_max}], n_scan={n_scan}", flush=True)
    print(f"  (강제 영점 t≈0 제외)", flush=True)

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

    from scipy.signal import find_peaks
    threshold = max(np.median(kappas) * 5, 100)
    peaks, _ = find_peaks(kappas, height=threshold, distance=3)
    predicted = ts[peaks]

    print(f"  κ 피크 {len(predicted)}개 (threshold={threshold:.0f})", flush=True)
    for p in predicted:
        idx = np.argmin(np.abs(ts - p))
        print(f"    predicted: t = {p:.2f}, κ = {kappas[idx]:.0f}", flush=True)

    # 비자명 영점만 매칭 (강제 영점 제외)
    nontrivial_zeros = zeros[np.abs(zeros) > 0.5]
    tol = 0.5
    in_range = nontrivial_zeros[(nontrivial_zeros >= t_scan_min) & (nontrivial_zeros <= t_scan_max)]

    matches = 0
    for p in predicted:
        if len(in_range) > 0 and np.min(np.abs(in_range - p)) < tol:
            matches += 1

    covered = 0
    for z in in_range:
        if len(predicted) > 0 and np.min(np.abs(predicted - z)) < tol:
            covered += 1

    print(f"  비자명 영점 (범위 내): {len(in_range)}개", flush=True)
    print(f"  예측→실제 매칭: {matches}/{len(predicted)}", flush=True)
    print(f"  실제→예측 커버: {covered}/{len(in_range)}", flush=True)

    return predicted, in_range, covered


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    t_start = time.time()
    out_path = os.path.expanduser('~/Desktop/gdl_unified/results/elliptic_curve_37a1.txt')
    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    log("=" * 72)
    log("=== Elliptic Curve L-function: 37a1 (GL(2), conductor 37) ===")
    log("=" * 72)
    log(f"Curve: y² + y = x³ - x")
    log(f"Conductor N = {N_COND}, rank = 1, root number ε = {EPSILON}")
    log(f"Critical line: σ = {SIGMA_CRIT}")
    log(f"Functional equation: Λ(E, 2-s) = ε·Λ(E, s), ε = {EPSILON}")
    log(f"  → Λ(E, 2-s) = -Λ(E, s) (반대칭!)")
    log(f"  → Λ(1+it) 순허수 (임계선 위에서)")
    log(f"  → s=1 강제 영점 (rank 1)")
    log(f"AFE terms: n_max = {N_MAX_COEFF}")
    log(f"mpmath precision: {mpmath.mp.dps} digits")
    log()

    # ══════════════════════════════════════════════════════════════════════
    # [A] aₙ 계수 검증
    # ══════════════════════════════════════════════════════════════════════
    _init_tables()
    an = _an_cache

    log("[A] aₙ 계수 검증 — 37a1 (LMFDB 참조)")
    # LMFDB: 1, -2, -3, 2, -2, 6, -1, 0, 6, 4, -5, -6, -2, 2, 6, -4, 0, -12, 0, -4
    lmfdb_an = [1, -2, -3, 2, -2, 6, -1, 0, 6, 4, -5, -6, -2, 2, 6, -4, 0, -12, 0, -4]
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
    # [A'] Rank / Root number 검증
    # ══════════════════════════════════════════════════════════════════════
    log("[A'] Rank / Root number 검증")
    log(f"  rank = 1, ε = {EPSILON}")

    # L(E, 1) = 0 확인 (rank 1 → 강제 영점)
    L_center = L_E(mpmath.mpc(1, 0))
    L_center_re = float(mpmath.re(L_center))
    L_center_im = float(mpmath.im(L_center))
    L_center_abs = float(abs(L_center))
    log(f"  L(E, 1) = {L_center_re:.4e} + {L_center_im:.4e}i")
    log(f"  |L(E, 1)| = {L_center_abs:.4e}")
    forced_zero_ok = L_center_abs < 1e-8
    log(f"  L(E, 1) = 0 확인: {'✅ PASS' if forced_zero_ok else '❌ FAIL'}")
    log()

    if not forced_zero_ok:
        log("⚠️ L(E,1) ≠ 0 — rank 또는 AFE 구현 점검 필요")

    # ══════════════════════════════════════════════════════════════════════
    # [B] AFE 검증: 함수 방정식
    # ══════════════════════════════════════════════════════════════════════
    log("[B] AFE 검증")

    # B1: L(E, 2) — 직접합과 비교
    L_afe = L_E(mpmath.mpf(2))
    direct_sum = sum(an[n] / n**2 for n in range(1, N_MAX_COEFF + 1))
    L_afe_re = float(mpmath.re(L_afe))
    diff_L2 = abs(L_afe_re - direct_sum)
    log(f"  L(E, 2) AFE    = {L_afe_re:.12f}")
    log(f"  L(E, 2) 직접합 = {direct_sum:.12f}")
    log(f"  차이: {diff_L2:.2e} {'✅' if diff_L2 < 1e-2 else '❌'}")
    log()

    # B2: 함수 방정식 Λ(s) = εΛ(2-s), ε = -1
    log("  [함수 방정식 검증] Λ(E, s) = ε·Λ(E, 2-s), ε = -1")
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
        log("⚠️ 함수 방정식 불만족 — AFE 구현 점검 필요 (ε=-1 부호 확인!)")

    # ══════════════════════════════════════════════════════════════════════
    # [C] 영점 탐색
    # ══════════════════════════════════════════════════════════════════════
    zeros = find_zeros_elliptic()

    log()
    log("[C] Zeros — t ∈ [%.2f, %.1f]" % (T_MIN, T_MAX))
    for i, tz in enumerate(zeros):
        marker = " (강제 영점, rank 1)" if abs(tz) < 0.01 else ""
        log(f"  γ_{i+1} = {tz:.8f}{marker}")
    log(f"  Total: {len(zeros)} zeros (강제 영점 포함)")
    log()

    if len(zeros) == 0:
        log("⚠️ 영점 0개 — 실험 중단.")
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"\n결과 저장: {out_path}", flush=True)
        return

    # LMFDB 교차검증
    nontrivial = zeros[np.abs(zeros) > 0.5]
    log("[C'] LMFDB 교차검증")
    log(f"  강제 영점: γ₁ = 0 (s=1, rank 1 → L(E,1)=0)")
    if len(nontrivial) >= 1:
        log(f"  첫 비자명 영점: γ = {nontrivial[0]:.8f}")
        log(f"  (LMFDB 참조: 첫 비자명 영점 ≈ 5.0... — 정밀값 확인 필요)")
    if len(nontrivial) >= 2:
        log(f"  γ = {nontrivial[1]:.8f}")
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
        marker = " ← 임계선" if abs(sigma - SIGMA_CRIT) < 0.01 else ""
        log(f"  σ={sigma:.1f}: jumps = {jumps}{marker}")

    max_sigma = max(sigma_results, key=sigma_results.get)
    off_crit_max = max(v for k, v in sigma_results.items() if abs(k - SIGMA_CRIT) > 0.01)
    on_crit = sigma_results.get(SIGMA_CRIT, 0)
    sigma_pass = abs(max_sigma - SIGMA_CRIT) < 0.01
    log(f"  최대 점프: σ={max_sigma:.1f} (on-crit={on_crit}, off-crit max={off_crit_max})")
    log(f"  판정: {'PASS ✅' if sigma_pass else 'FAIL ❌'}")
    log(f"  (GL(2) 구조상 FAIL 예상 — 11a1 패턴 재현 확인)")
    log()

    # ── D2: 모노드로미 ──
    # D2a: 폐곡선 적분 (비자명 영점)
    contour_dev, contour_abs = measure_monodromy_contour(zeros)

    log()
    log("[D2a] Monodromy — contour integral (radius=0.3, 비자명 영점)")
    if contour_abs is not None:
        log(f"  Mean |mono|/π = {contour_abs:.6f}")
        log(f"  Mean deviation from nearest integer = {contour_dev:.6f}")
        contour_pass = contour_abs > 1.9
        log(f"  판정: {'PASS ✅' if contour_pass else 'FAIL ❌'} (기준: |mono|/π > 1.9)")
    else:
        contour_pass = False
        log("  계산 실패")

    # D2a': 강제 영점 모노드로미
    forced_mono = measure_monodromy_forced_zero()
    log()
    log("[D2a'] Monodromy — 강제 영점 (s=1)")
    if forced_mono is not None:
        log(f"  mono = {forced_mono:.6f} rad, |mono|/π = {abs(forced_mono)/np.pi:.6f}")
        forced_pass = abs(abs(forced_mono) / np.pi - 2.0) < 0.2
        log(f"  단순 영점 확인 (winding=1): {'PASS ✅' if forced_pass else 'FAIL ❌'}")
    else:
        forced_pass = False
        log("  계산 실패")

    # D2b: log-space arg
    logspace_dev, logspace_abs = measure_monodromy_logspace(zeros)
    log()
    log("[D2b] Monodromy — log-space arg (보조, 비자명 영점)")
    if logspace_abs is not None:
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
    kappa_result = measure_kappa_concentration(zeros)
    log()
    if kappa_result[0] is not None:
        kappa_ratio, near_med, far_med = kappa_result
        log(f"[D3] κ concentration (δ={DELTA_OFFSET} from σ={SIGMA_CRIT})")
        log(f"  near median: {near_med:.1f}")
        log(f"  far median:  {far_med:.1f}")
        log(f"  κ(near)/κ(far) = {kappa_ratio:.1f}×")
        kappa_pass = kappa_ratio > 10
        log(f"  판정: {'PASS ✅' if kappa_pass else 'FAIL ❌'} (기준: > 10×)")
    else:
        kappa_ratio, near_med, far_med = None, None, None
        kappa_pass = False
        log("[D3] κ 집중도: 계산 실패")
    log()

    # ── D4: 블라인드 예측 ──
    predicted, actual_in_range, covered = measure_blind_prediction(zeros)
    log()
    log("[D4] Blind prediction (강제 영점 제외)")
    log(f"  Predicted: {[f'{p:.2f}' for p in predicted]}")
    log(f"  Actual nontrivial (in range): {[f'{z:.2f}' for z in actual_in_range]}")
    log(f"  Covered: {covered}/{len(actual_in_range)} (tol=0.5)")

    if len(actual_in_range) > 0:
        coverage = covered / len(actual_in_range)
        blind_pass = coverage >= 0.7
        log(f"  Coverage: {coverage*100:.0f}%")
        log(f"  판정: {'PASS ✅' if blind_pass else 'FAIL ❌'} (기준: ≥ 70%)")
    else:
        blind_pass = False
        log("  비자명 영점 없음")
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
        (f" ({kappa_ratio:.1f}×, near={near_med:.1f}, far={far_med:.1f})" if kappa_ratio else ""))
    if kappa_pass:
        pass_count += 1

    log(f"  4. 블라인드예측: {'PASS ✅' if blind_pass else 'FAIL ❌'}" +
        (f" ({covered}/{len(actual_in_range)})" if len(actual_in_range) > 0 else ""))
    if blind_pass:
        pass_count += 1

    log()
    log(f"  통과: {pass_count}/{total}")

    if pass_count >= 3:
        log(f"  ★ 양성 — GL(2) ξ-다발 프레임워크 재현 (37a1, rank 1, ε=-1)")
        log(f"  → 11a1 패턴(3/4) 재현 → GL(2) 확장 보편성 확립")
        log(f"  → rank, conductor, root number에 무관한 GL(2) 확장")
    elif pass_count >= 2:
        log(f"  ⚠ 조건부 양성 — 추가 검증 필요")
    elif pass_count >= 1:
        log(f"  ⚠ 약한 양성 — 부분적 증거만")
    else:
        log(f"  ❌ 음성 — GL(2) 확장 실패")

    log()
    log(f"  [11a1 대비 비교]")
    log(f"    11a1 (rank 0, ε=+1): 3/4 양성 (σ-유일성 FAIL)")
    log(f"    37a1 (rank 1, ε=-1): {pass_count}/4")
    log(f"    {'→ 패턴 재현 확인' if pass_count >= 3 else '→ 차이 분석 필요'}")

    log("=" * 72)

    elapsed = time.time() - t_start
    log(f"\n소요 시간: {elapsed:.0f}초 ({elapsed/60:.1f}분)")

    # 결과 저장
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n결과 저장: {out_path}", flush=True)


if __name__ == '__main__':
    main()
