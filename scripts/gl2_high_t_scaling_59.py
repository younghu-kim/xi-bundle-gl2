"""
=============================================================================
[Project RDL] 결과 #59 — GL(2) 11a1 고 t 스케일링 검증 (t ≈ 50–150)
=============================================================================

목표:
  - #31에서 GL(1) ζ 고 t 스케일링 확인 (t=1000/5000/10000, κ≈1116–1120 안정)
  - GL(2) 11a1에서도 동일한 안정성이 성립하는지 검증
  - 3개 이상 고 t 영점에서 κ, 모노드로미, σ-프로파일 측정
  - #52 결과 (t<30)와 비교 → 스케일링 안정성 판정

프로토콜:
  1. AFE 함수방정식 검증: Λ(s) = ε·Λ(2-s) 고 t에서도 성립 확인
  2. 3구간 (t≈50, 100, 150)에서 영점 탐색 (κ 스윕 → 피크 → findroot)
  3. LMFDB 교차검증: 저 t 영점으로 AFE 정확성 확인 + 고 t 영점 위치 검증
  4. 각 영점에서 3종 측정:
     (a) κ at σ=1+0.03
     (b) 모노드로미 (r=0.3, n=64)
     (c) σ-프로파일 (peak_σ, FWHM)
  5. #52 저 t 영점에서 동일 측정 (기준선) → 비교

성공 기준:
  - [필수] 3개 이상 고 t 영점에서 측정 완료
  - [필수] LMFDB 교차검증으로 영점 위치 정확성 확인 (오차 < 0.1)
  - [양성] κ 비율이 #52 (t<30) 결과와 30% 이내 일관
  - [양성] mono/π = 2.000 ± 0.01 유지
  - [양성] peak_σ → 1.0 수렴 (GL(2) 임계선)

결과 파일: results/gl2_high_t_scaling_59.txt
=============================================================================
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.expanduser('~/Desktop/gdl_unified'))
sys.path.insert(0, os.path.expanduser('~/Desktop/gdl_unified/scripts'))

import mpmath

try:
    from scipy.signal import find_peaks as scipy_find_peaks
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

OUTFILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "gl2_high_t_scaling_59.txt"
)
os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 파라미터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SIGMA_CRIT = 1.0       # GL(2) 임계선 (★ GL(1)의 0.5와 다름!)
DELTA = 0.03           # κ 측정 오프셋 (영점 위 직접 측정 금지)
DPS_BASE = 60          # 기본 정밀도
DPS_PRECOMP = 150      # AFE 전처리 정밀도 (고 t 안전 마진)
N_MAX = 100            # AFE 항 수 (t=150에서 필요: ~17항, 100으로 충분)
N_COND = 11            # conductor
EPS_11 = 1             # 함수방정식 부호 ε = +1


def get_dps(t_val):
    """
    적응적 DPS: 고 t에서 AFE 내부 cancellation 보상.
    t<60: dps=60 (검증 완료)
    t≈100: dps≈100 (경험적 — dps=60 실패, dps=120 성공)
    t≈150: dps≈135
    공식: max(60, int(0.7*|t| + 30))
    """
    return max(DPS_BASE, int(0.7 * abs(t_val) + 30))

# 모노드로미 (수학자 지시: r=0.3, n=64)
MONO_RADIUS = 0.3
MONO_STEPS = 64

# 영점 탐색 구간
SEARCH_ZONES = [
    (48, 55, 0.2),      # t≈50 구간
    (95, 105, 0.2),     # t≈100 구간
    (140, 155, 0.3),    # t≈150 구간 (간격 넓힘: 계산량 절감)
]

# 저 t 기준선 영점 (LMFDB, #52에서 검증 완료)
LOW_T_REFS = [6.36261389, 13.56863906, 20.37926046]  # γ₁, γ₅, γ₁₀

# LMFDB 영점 전체 (17개, #52에서 사용)
LMFDB_ZEROS_LOW = np.array([
    6.36261389, 8.60353962, 10.03550910, 11.45125861, 13.56863906,
    15.91407260, 17.03361032, 17.94143357, 19.18572497, 20.37926046,
    22.17249029, 23.30141550, 25.20986842, 25.87640308, 27.06763523,
    28.68390988, 29.97485995
])

# σ-프로파일 파라미터
SIGMA_PROFILE_RANGE = 0.3    # σ_crit ± 0.3
SIGMA_PROFILE_POINTS = 51    # 격자 점수 (간격 ≈ 0.012)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 로깅
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

lines = []

def log(msg=""):
    print(msg, flush=True)
    lines.append(str(msg))

def flush_to_file():
    with open(OUTFILE, "w") as f:
        f.write("\n".join(lines) + "\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11a1 AFE (#52에서 그대로 복사, dps=60, N_MAX=100으로 변경)
# Λ(s) = Σ aₙ [xₙ^{-s} Γ(s, xₙ) + ε xₙ^{-(2-s)} Γ(2-s, xₙ)]
# ★★★ AFE 필수: 직접합은 임계선에서 조건부 수렴만
# ★★★ aₚ: bad prime p=11에서 별도 처리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_an_11 = None
_precomp_11 = None


def _init_11a1():
    """11a1 aₙ 계수 초기화 + AFE 전처리"""
    global _an_11, _precomp_11
    if _an_11 is not None:
        return

    def ap(p):
        """a_p for 11a1: y² + y = x³ - x² - 10x - 20"""
        if p == 11:
            return 1  # bad prime (★★★)
        if p == 2:
            count = 1
            for x in range(2):
                for y in range(2):
                    if (y * y + y - x * x * x + x * x + 10 * x + 20) % 2 == 0:
                        count += 1
            return p + 1 - count
        aff = 0
        for x in range(p):
            d = (4 * x * x * x - 4 * x * x - 40 * x - 79) % p
            if d == 0:
                aff += 1
            elif pow(d, (p - 1) // 2, p) == 1:
                aff += 2
        return p - aff

    # 소수 체
    sieve = [True] * (N_MAX + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(N_MAX**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, N_MAX + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, N_MAX + 1) if sieve[i]]

    ap_dict = {p: ap(p) for p in primes}

    # aₚᵏ (Hecke multiplicativity)
    apk = {}
    for p in primes:
        apk[(p, 0)] = 1
        apk[(p, 1)] = ap_dict[p]
        pk = p
        k = 1
        while pk * p <= N_MAX:
            pk *= p
            k += 1
            if p == 11:
                apk[(p, k)] = ap_dict[p] ** k
            else:
                apk[(p, k)] = ap_dict[p] * apk[(p, k - 1)] - p * apk[(p, k - 2)]

    # aₙ (multiplicative)
    an = [0] * (N_MAX + 1)
    an[1] = 1
    for n in range(2, N_MAX + 1):
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
            result *= ap_dict[temp]
        an[n] = result

    # AFE 전처리: xₙ = 2πn/√N (높은 dps로 전처리 → 모든 t에서 재사용)
    saved_dps = mpmath.mp.dps
    mpmath.mp.dps = DPS_PRECOMP
    sqrt_N = mpmath.sqrt(mpmath.mpf(N_COND))
    two_pi = 2 * mpmath.pi
    _precomp_11 = [(mpmath.mpf(an[n]), two_pi * n / sqrt_N)
                   for n in range(1, N_MAX + 1) if an[n] != 0]
    mpmath.mp.dps = saved_dps
    _an_11 = an

    # 검증: 알려진 a_p 값
    known = {2: -2, 3: -1, 5: 1, 7: -2, 11: 1, 13: 4}
    ok = True
    for p, exp in known.items():
        if p <= N_MAX and an[p] != exp:
            log(f"  ⚠️ a_{p} = {an[p]}, expected {exp}!")
            ok = False
    if ok:
        log(f"  [11a1 초기화] 검증 OK: a₂={an[2]}, a₃={an[3]}, a₅={an[5]}, "
            f"a₇={an[7]}, a₁₁={an[11]}")
    log(f"  [11a1 초기화] 비영 항: {len(_precomp_11)}개 (N_MAX={N_MAX}), dps={DPS_BASE}–{DPS_PRECOMP} (적응적)")


def Lambda_11a1(s):
    """
    Λ(11a1, s) via AFE — #52에서 그대로 복사.
    Λ(s) = Σ aₙ [xₙ^{-s} Γ(s, xₙ) + ε xₙ^{-(2-s)} Γ(2-s, xₙ)]
    σ_crit = 1, ε = +1, N = 11

    ★★ 적응적 DPS: t에 따라 자동 조절 (dps=60 → 고 t에서 cancellation 실패 방지)
    """
    saved_dps = mpmath.mp.dps
    t_val = float(abs(mpmath.im(s)))
    mpmath.mp.dps = get_dps(t_val)
    try:
        _init_11a1()
        s_mp = mpmath.mpc(s)
        s_conj = 2 - s_mp  # GL(2) 함수방정식: s → 2-s
        eps = mpmath.mpf(EPS_11)
        result = mpmath.mpc(0)
        for an_val, x_n in _precomp_11:
            term1 = mpmath.power(x_n, -s_mp) * mpmath.gammainc(s_mp, x_n)
            term2 = eps * mpmath.power(x_n, -s_conj) * mpmath.gammainc(s_conj, x_n)
            result += an_val * (term1 + term2)
        return result
    finally:
        mpmath.mp.dps = saved_dps


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 측정 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def curvature(sigma, t, h=1e-6):
    """
    κ(σ+it) = |Λ'/Λ|² via 중앙차분 비율.

    ★★ 고 t 대응:
    - |Λ(s)| ~ e^{-πt/2}로 극소하지만, mpmath dps=60으로 비율 계산은 안전.
    - 절대 임계값 금지 (|Λ| < 10^{-45} 같은 고정값 사용 안 함).
    - log-derivative 금지 (branch cut 문제 — arg(Λ) ≈ ±π일 때 2π 점프).
    - 원래 비율 방식 (Lp-Lm)/(2h·L0) 사용: L0이 정확히 0이 아닌 한 안전.
    """
    s = mpmath.mpc(sigma, t)
    h_mp = mpmath.mpf(str(h))
    try:
        L0 = Lambda_11a1(s)
        if L0 == 0:
            return 1e12  # 진짜 영점 (정확히 0인 경우만)
        Lp = Lambda_11a1(s + h_mp)
        Lm = Lambda_11a1(s - h_mp)
        conn = (Lp - Lm) / (2 * h_mp * L0)
        k = float(abs(conn) ** 2)
        return k if np.isfinite(k) else 1e12
    except Exception as e:
        log(f"    WARNING curvature σ={sigma:.3f} t={t:.4f}: {e}")
        return 0.0


def measure_monodromy(t_center, radius=MONO_RADIUS, n_steps=MONO_STEPS):
    """
    Λ(11a1, s) 주위 폐곡선 모노드로미 (arg 누적).
    중심: (σ_crit, t_center), 반지름 radius.
    Returns: |mono|/π (영점이면 ≈2.0, 아니면 ≈0.0). None이면 실패.
    """
    center = mpmath.mpc(SIGMA_CRIT, t_center)
    phase_accum = mpmath.mpf(0)
    prev_val = None

    for j in range(n_steps + 1):
        theta = 2 * mpmath.pi * j / n_steps
        s = center + radius * mpmath.exp(1j * theta)
        try:
            val = Lambda_11a1(s)
            if val == 0:  # 진짜 영점 경유 (절대 임계값 금지 — 고 t에서 |Λ|~e^{-πt/2})
                return None
        except Exception as e:
            log(f"    WARNING monodromy t={t_center:.4f}: {e}")
            return None

        if prev_val is not None:
            ratio = val / prev_val
            phase_accum += mpmath.im(mpmath.log(ratio))
        prev_val = val

    return float(abs(phase_accum)) / np.pi


def measure_sigma_profile(t_zero, sigma_center=SIGMA_CRIT,
                          half_range=SIGMA_PROFILE_RANGE,
                          n_points=SIGMA_PROFILE_POINTS):
    """
    σ-프로파일: 고정 t에서 σ를 스윕하여 κ(σ+it) 피크 위치와 FWHM 측정.
    Returns: (peak_sigma, fwhm, peak_kappa)
    """
    sigmas = np.linspace(sigma_center - half_range, sigma_center + half_range, n_points)
    kappas = np.array([curvature(sig, t_zero) for sig in sigmas])

    # peak_σ
    peak_idx = np.argmax(kappas)
    peak_sigma = sigmas[peak_idx]
    peak_kappa = kappas[peak_idx]

    # FWHM
    half_max = peak_kappa / 2.0
    above = kappas >= half_max
    if above.any():
        indices = np.where(above)[0]
        first = indices[0]
        last = indices[-1]
        fwhm = sigmas[last] - sigmas[first]
    else:
        fwhm = float('nan')

    return peak_sigma, fwhm, peak_kappa


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 영점 탐색 (κ 스윕 → findroot 정밀화)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_zeros_in_zone(t_min, t_max, dt):
    """
    구간 [t_min, t_max]에서 κ 스윕 → 피크 → findroot 정밀화.
    Returns: list of (t_zero, kappa_peak, sigma_zero) tuples
    """
    sigma_sweep = SIGMA_CRIT + DELTA
    ts = np.arange(t_min, t_max + dt / 2, dt)
    kappas = []

    log(f"\n  κ 스윕: σ={sigma_sweep:.3f}, t ∈ [{t_min}, {t_max}], Δt={dt}, {len(ts)}점")
    t0 = time.time()

    for t in ts:
        k = curvature(sigma_sweep, t)
        kappas.append(k)

    kappas = np.array(kappas)
    elapsed = time.time() - t0
    log(f"  스윕 완료: {elapsed:.1f}초, κ median={np.median(kappas):.2f}, max={kappas.max():.2f}")

    # 피크 검출
    log_k = np.log1p(kappas)
    if SCIPY_OK:
        median_logk = np.median(log_k)
        prom = max(0.3, median_logk * 0.3)
        peaks_idx, _ = scipy_find_peaks(log_k, prominence=prom, distance=2)
        log(f"  scipy.find_peaks: prominence≥{prom:.3f}, {len(peaks_idx)}개 피크")
    else:
        peaks_list = []
        for i in range(1, len(kappas) - 1):
            if kappas[i] > kappas[i - 1] and kappas[i] > kappas[i + 1]:
                peaks_list.append(i)
        peaks_idx = np.array(peaks_list, dtype=int)
        log(f"  수동 극대점: {len(peaks_idx)}개")

    # findroot으로 정밀화
    zeros = []
    fail_count = 0
    for idx in peaks_idx:
        t_approx = ts[idx]
        k_peak = kappas[idx]

        try:
            mpmath.mp.dps = get_dps(t_approx)
            # 복소 시작점으로 findroot → σ도 변동 허용
            s_approx = mpmath.mpc(SIGMA_CRIT, t_approx)
            s_root = mpmath.findroot(Lambda_11a1, s_approx)

            t_ref = float(mpmath.im(s_root))
            sigma_ref = float(mpmath.re(s_root))

            # 검증: σ ≈ σ_crit (임계선 위) + |Λ|가 인근 배경보다 훨씬 작은지
            val_at_root = Lambda_11a1(s_root)
            abs_val = float(abs(val_at_root))

            # 배경 |Λ| 추정: σ 약간 떨어진 지점 (비영점)
            s_bg = mpmath.mpc(SIGMA_CRIT + 0.5, t_ref)
            bg_val = float(abs(Lambda_11a1(s_bg)))
            ratio = abs_val / bg_val if bg_val > 0 else float('inf')

            on_crit = abs(sigma_ref - SIGMA_CRIT) < 0.01

            if on_crit and (ratio < 1e-5 or abs_val == 0):
                zeros.append((t_ref, k_peak, sigma_ref))
                log(f"    영점: t≈{t_approx:.2f} → t={t_ref:.8f}, σ={sigma_ref:.6f}, "
                    f"|Λ|={abs_val:.2e}, |Λ|/bg={ratio:.2e}")
            elif on_crit and ratio < 1e-2:
                zeros.append((t_ref, k_peak, sigma_ref))
                log(f"    영점 (약): t≈{t_approx:.2f} → t={t_ref:.8f}, σ={sigma_ref:.6f}, "
                    f"|Λ|={abs_val:.2e}, |Λ|/bg={ratio:.2e}")
            else:
                fail_count += 1
                log(f"    findroot 미수렴: t≈{t_approx:.2f} → t={t_ref:.4f}, σ={sigma_ref:.4f}, "
                    f"|Λ|/bg={ratio:.2e}")
        except Exception as e:
            fail_count += 1
            log(f"    findroot 실패: t≈{t_approx:.2f}: {e}")

    if fail_count > 0:
        log(f"  findroot 실패: {fail_count}/{len(peaks_idx)}건")

    return zeros


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log("=" * 72)
log("결과 #59 — GL(2) 11a1 고 t 스케일링 검증 (t ≈ 50–150)")
log("=" * 72)
log(f"시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"σ_crit={SIGMA_CRIT} (GL(2)!), δ={DELTA}, dps={DPS_BASE}(adaptive), N_MAX={N_MAX}")
log(f"모노드로미: r={MONO_RADIUS}, n={MONO_STEPS}")
log(f"σ-프로파일: σ_crit±{SIGMA_PROFILE_RANGE}, {SIGMA_PROFILE_POINTS}점")
log(f"탐색 구간: {SEARCH_ZONES}")
log(f"저 t 기준선: {LOW_T_REFS}")
log()

t_total = time.time()

# ════════════════════════════════════════════════════════════════════════════
# Step 0: AFE 초기화 + 함수방정식 검증
# ════════════════════════════════════════════════════════════════════════════
log("=" * 72)
log("[Step 0] AFE 초기화 + 함수방정식 검증")
log("=" * 72)

_init_11a1()

# 함수방정식: Λ(s) = ε·Λ(2-s) at various s including high t
log("\n  함수방정식 검증: Λ(s) = ε·Λ(2-s) [ε=+1]")
fe_points = [
    (0.7, 5.0),
    (1.3, 20.0),
    (0.5, 50.0),
    (1.5, 100.0),
    (0.8, 150.0),
]
fe_ok = True
for sig_test, t_test in fe_points:
    s_test = mpmath.mpc(sig_test, t_test)
    L_s = Lambda_11a1(s_test)
    L_2ms = Lambda_11a1(2 - s_test)
    abs_Ls = abs(L_s)
    abs_diff = abs(L_s - EPS_11 * L_2ms)
    # 고 t에서 |Λ| ~ e^{-πt/2}이므로 절대값이 극소. log-scale 비교 사용.
    if abs_Ls > 0 and abs_diff > 0:
        rel_err = float(abs_diff / abs_Ls)
    elif abs_Ls == 0 and abs_diff == 0:
        rel_err = 0.0  # 양쪽 다 0
    else:
        # |Λ|=0이지만 차이≠0, 또는 그 반대: 숫자 노이즈
        rel_err = float('nan')
    status = "✅" if rel_err < 1e-10 else "⚠️"
    if not np.isnan(rel_err) and rel_err >= 1e-5:
        fe_ok = False
        status = "❌"
    log(f"    {status} s={sig_test}+{t_test}i: rel_err={rel_err:.2e}, |Λ|={float(abs_Ls):.2e}")

log(f"\n  함수방정식 검증 결과: {'✅ 통과' if fe_ok else '❌ 실패'}")
log()
flush_to_file()

# ════════════════════════════════════════════════════════════════════════════
# Step 0b: LMFDB 교차검증 (저 t 영점으로 AFE 정확성 확인)
# ════════════════════════════════════════════════════════════════════════════
log("=" * 72)
log("[Step 0b] LMFDB 교차검증: 저 t 영점에서 |Λ(σ_crit + it)| 확인")
log("=" * 72)

lmfdb_err_max = 0.0
for i, t_lmfdb in enumerate(LMFDB_ZEROS_LOW):
    s_zero = mpmath.mpc(SIGMA_CRIT, t_lmfdb)
    val = Lambda_11a1(s_zero)
    abs_val = float(abs(val))
    # findroot으로 정밀 영점 위치 확인
    try:
        s_ref = mpmath.findroot(Lambda_11a1, s_zero)
        t_ref = float(mpmath.im(s_ref))
        loc_err = abs(t_ref - t_lmfdb)
        lmfdb_err_max = max(lmfdb_err_max, loc_err)
        log(f"  γ_{i+1:2d} = {t_lmfdb:.8f} → findroot: {t_ref:.8f}, "
            f"오차={loc_err:.6f}, |Λ|={abs_val:.2e}")
    except Exception as e:
        log(f"  γ_{i+1:2d} = {t_lmfdb:.8f}: findroot 실패 ({e})")

log(f"\n  LMFDB 17개 영점 교차검증: max_err={lmfdb_err_max:.6f}")
log(f"  기준 (오차 < 0.1): {'✅ PASS' if lmfdb_err_max < 0.1 else '❌ FAIL'}")
log()
flush_to_file()

# ════════════════════════════════════════════════════════════════════════════
# Step 1: 고 t 영점 탐색 (3구간)
# ════════════════════════════════════════════════════════════════════════════
log("=" * 72)
log("[Step 1] 고 t 영점 탐색")
log("=" * 72)

all_high_t_zeros = {}  # zone_label → [(t, kappa, sigma), ...]
for zone_idx, (t_min, t_max, dt) in enumerate(SEARCH_ZONES):
    zone_label = f"t≈{(t_min+t_max)//2}"
    log(f"\n--- 구간 {zone_idx+1}: {zone_label} (t ∈ [{t_min}, {t_max}]) ---")
    zeros = find_zeros_in_zone(t_min, t_max, dt)
    all_high_t_zeros[zone_label] = zeros
    log(f"  → {len(zeros)}개 영점 확인")
    flush_to_file()

# 각 구간에서 대표 영점 1개 선택 (가장 높은 κ)
selected_zeros = []
for zone_label, zeros in all_high_t_zeros.items():
    if zeros:
        best = max(zeros, key=lambda x: x[1])
        selected_zeros.append({
            'zone': zone_label,
            't': best[0],
            'kappa_peak': best[1],
            'sigma': best[2]
        })
        log(f"\n  {zone_label} 대표: t={best[0]:.8f}, σ={best[2]:.6f}")

# 구간별 영점 없으면 추가 보완
if len(selected_zeros) < 3:
    log(f"\n  ⚠️ 대표 영점 {len(selected_zeros)}개 < 3개")
    # 기존 구간에서 추가 영점 선택
    for zone_label, zeros in all_high_t_zeros.items():
        for z in zeros:
            already = any(abs(z[0] - sel['t']) < 1.0 for sel in selected_zeros)
            if not already:
                selected_zeros.append({
                    'zone': zone_label,
                    't': z[0],
                    'kappa_peak': z[1],
                    'sigma': z[2]
                })
            if len(selected_zeros) >= 3:
                break
        if len(selected_zeros) >= 3:
            break

log(f"\n  선택된 고 t 영점: {len(selected_zeros)}개")
for sel in selected_zeros:
    log(f"    {sel['zone']}: t={sel['t']:.8f}")

flush_to_file()

# ════════════════════════════════════════════════════════════════════════════
# Step 2: 3종 측정 (저 t 기준선 + 고 t)
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 72)
log("[Step 2] 3종 측정 (κ, 모노드로미, σ-프로파일)")
log("=" * 72)

measurements = []

# --- 저 t 기준선 ---
log("\n--- 저 t 기준선 (t<30, #52 범위) ---")
for t_ref in LOW_T_REFS:
    log(f"\n  ◆ 영점 t={t_ref:.8f} (저 t)")
    t_start = time.time()

    # (a) κ at σ=σ_crit+δ
    k_val = curvature(SIGMA_CRIT + DELTA, t_ref)
    log(f"    (a) κ(σ={SIGMA_CRIT+DELTA:.2f}, t={t_ref:.4f}) = {k_val:.4f}")

    # (b) 모노드로미
    mono_val = measure_monodromy(t_ref)
    mono_str = f"{mono_val:.4f}" if mono_val is not None else "FAIL"
    log(f"    (b) mono/π = {mono_str}")

    # (c) σ-프로파일
    p_sigma, p_fwhm, p_kmax = measure_sigma_profile(t_ref)
    log(f"    (c) peak_σ = {p_sigma:.4f}, FWHM = {p_fwhm:.4f}, κ_max = {p_kmax:.2f}")

    elapsed = time.time() - t_start
    log(f"    소요: {elapsed:.1f}초")

    measurements.append({
        'label': f'저t (γ, t={t_ref:.2f})',
        't': t_ref,
        'kappa': k_val,
        'mono_pi': mono_val,
        'peak_sigma': p_sigma,
        'fwhm': p_fwhm,
        'kappa_max': p_kmax,
        'category': 'low_t'
    })
    flush_to_file()

# --- 고 t 영점 ---
log("\n--- 고 t 영점 ---")
for sel in selected_zeros:
    t_z = sel['t']
    zone = sel['zone']
    log(f"\n  ★ 영점 t={t_z:.8f} ({zone})")
    t_start = time.time()

    # (a) κ
    k_val = curvature(SIGMA_CRIT + DELTA, t_z)
    log(f"    (a) κ(σ={SIGMA_CRIT+DELTA:.2f}, t={t_z:.4f}) = {k_val:.4f}")

    # (b) 모노드로미
    mono_val = measure_monodromy(t_z)
    mono_str = f"{mono_val:.4f}" if mono_val is not None else "FAIL"
    log(f"    (b) mono/π = {mono_str}")

    # (c) σ-프로파일
    p_sigma, p_fwhm, p_kmax = measure_sigma_profile(t_z)
    log(f"    (c) peak_σ = {p_sigma:.4f}, FWHM = {p_fwhm:.4f}, κ_max = {p_kmax:.2f}")

    elapsed = time.time() - t_start
    log(f"    소요: {elapsed:.1f}초")

    measurements.append({
        'label': f'고t ({zone})',
        't': t_z,
        'kappa': k_val,
        'mono_pi': mono_val,
        'peak_sigma': p_sigma,
        'fwhm': p_fwhm,
        'kappa_max': p_kmax,
        'category': 'high_t'
    })
    flush_to_file()

# ════════════════════════════════════════════════════════════════════════════
# Step 3: 비교 보고
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 72)
log("[Step 3] 비교 보고")
log("=" * 72)

low_t_ms = [m for m in measurements if m['category'] == 'low_t']
high_t_ms = [m for m in measurements if m['category'] == 'high_t']

# 기준값: 저 t 평균
base_kappa = np.mean([m['kappa'] for m in low_t_ms]) if low_t_ms else float('nan')
base_mono = np.mean([m['mono_pi'] for m in low_t_ms if m['mono_pi'] is not None]) \
    if low_t_ms else float('nan')
base_peak_sigma = np.mean([m['peak_sigma'] for m in low_t_ms]) if low_t_ms else float('nan')
base_fwhm = np.mean([m['fwhm'] for m in low_t_ms]) if low_t_ms else float('nan')

log(f"\n  저 t 기준값 ({len(low_t_ms)}점 평균):")
log(f"    κ = {base_kappa:.4f}")
log(f"    mono/π = {base_mono:.4f}")
log(f"    peak_σ = {base_peak_sigma:.4f}")
log(f"    FWHM = {base_fwhm:.4f}")
log()

# #31 GL(1) 기준값
log("  GL(1) #31 기준값 (t=1000/5000/10000):")
log("    κ ≈ 1116–1120, mono/π = 2.0000, peak_σ ≈ 0.494")
log()

# 결과 표
log("  " + "=" * 78)
log(f"  {'':>3} {'t':>10} {'κ':>10} {'κ비율':>8} {'mono/π':>8} "
    f"{'peak_σ':>8} {'FWHM':>8} {'판정':>10}")
log("  " + "-" * 78)

judgments = []
for m in measurements:
    t_val = m['t']
    k_val = m['kappa']
    k_ratio = k_val / base_kappa if base_kappa > 0 else float('nan')
    mono = m['mono_pi'] if m['mono_pi'] is not None else float('nan')
    p_sig = m['peak_sigma']
    fwhm = m['fwhm']

    # 판정 기준
    ok_kappa = abs(k_ratio - 1.0) < 0.30   # κ 30% 이내
    ok_mono = abs(mono - 2.0) < 0.01        # mono/π = 2.000 ± 0.01
    ok_sigma = abs(p_sig - SIGMA_CRIT) < 0.05  # peak_σ ≈ 1.0

    if ok_kappa and ok_mono and ok_sigma:
        verdict = "✅ PASS"
    elif ok_mono and ok_sigma:
        verdict = "⚠️ κ편차"
    elif ok_mono:
        verdict = "⚠️ σ편차"
    else:
        verdict = "❌ FAIL"

    cat = "★" if m['category'] == 'high_t' else " "
    log(f"  {cat} {t_val:>10.4f} {k_val:>10.2f} {k_ratio:>7.2f}× "
        f"{mono:>8.4f} {p_sig:>8.4f} {fwhm:>8.4f}  {verdict}")

    if m['category'] == 'high_t':
        judgments.append({
            't': t_val,
            'kappa_ratio': k_ratio,
            'mono_pi': mono,
            'peak_sigma': p_sig,
            'ok_kappa': ok_kappa,
            'ok_mono': ok_mono,
            'ok_sigma': ok_sigma,
        })

log("  " + "=" * 78)
flush_to_file()

# ════════════════════════════════════════════════════════════════════════════
# Step 4: 최종 판정
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 72)
log("[Step 4] 최종 판정")
log("=" * 72)

n_high = len(high_t_ms)
log(f"\n  고 t 영점 수: {n_high}개")

log("\n[ 성공 기준 판정 ]")

# [필수] 3개 이상 고 t 영점 측정
r1 = n_high >= 3
log(f"  {'✅' if r1 else '❌'} [필수] 3개 이상 고 t 영점 측정: {n_high}개 "
    f"{'PASS' if r1 else 'FAIL'}")

# [필수] LMFDB 교차검증 (오차 < 0.1)
r2 = lmfdb_err_max < 0.1
log(f"  {'✅' if r2 else '❌'} [필수] LMFDB 교차검증 (오차 < 0.1): "
    f"max_err={lmfdb_err_max:.6f} {'PASS' if r2 else 'FAIL'}")

# [양성] κ 비율 30% 이내
if judgments:
    kappa_ok = sum(1 for j in judgments if j['ok_kappa'])
    r3 = kappa_ok == len(judgments)
    kappa_ratios = [j['kappa_ratio'] for j in judgments]
    log(f"  {'✅' if r3 else '❌'} [양성] κ 비율 30% 이내: {kappa_ok}/{len(judgments)} "
        f"(비율: {', '.join(f'{r:.2f}×' for r in kappa_ratios)}) {'PASS' if r3 else 'FAIL'}")
else:
    r3 = False
    log(f"  ❌ [양성] κ 비율: 측정 없음")

# [양성] mono/π = 2.000 ± 0.01
if judgments:
    mono_ok = sum(1 for j in judgments if j['ok_mono'])
    r4 = mono_ok == len(judgments)
    mono_vals = [j['mono_pi'] for j in judgments]
    log(f"  {'✅' if r4 else '❌'} [양성] mono/π = 2.000 ± 0.01: {mono_ok}/{len(judgments)} "
        f"(값: {', '.join(f'{v:.4f}' for v in mono_vals)}) {'PASS' if r4 else 'FAIL'}")
else:
    r4 = False
    log(f"  ❌ [양성] mono/π: 측정 없음")

# [양성] peak_σ → 1.0
if judgments:
    sigma_ok = sum(1 for j in judgments if j['ok_sigma'])
    r5 = sigma_ok == len(judgments)
    sigma_vals = [j['peak_sigma'] for j in judgments]
    log(f"  {'✅' if r5 else '❌'} [양성] peak_σ → 1.0: {sigma_ok}/{len(judgments)} "
        f"(값: {', '.join(f'{v:.4f}' for v in sigma_vals)}) {'PASS' if r5 else 'FAIL'}")
else:
    r5 = False
    log(f"  ❌ [양성] peak_σ: 측정 없음")

log()

# GL(1) #31과의 교차비교
log("[ GL(1) #31 vs GL(2) #59 교차비교 ]")
log(f"  GL(1) ζ: t=1000 κ=1115.9, t=5000 κ=1120.3, t=10000 κ=1119.6 → 안정")
log(f"  GL(2) 11a1: 저 t κ={base_kappa:.1f}")
if high_t_ms:
    for m in high_t_ms:
        log(f"               고 t t={m['t']:.1f} κ={m['kappa']:.1f} "
            f"(비율={m['kappa']/base_kappa:.2f}×)")
log(f"  결론: GL(1)과 GL(2) 모두 κ가 t에 대해 안정한지 비교")
log()

# 종합 판정
mandatory_ok = r1 and r2
optional_count = sum([r3, r4, r5])

if mandatory_ok and optional_count >= 3:
    final = "★★★ 완전 양성: 전 기준 통과 → GL(2) 고 t 스케일링 확립"
elif mandatory_ok and optional_count >= 2:
    final = "★★ 양성: 필수 통과, 대부분 양성"
elif mandatory_ok and optional_count >= 1:
    final = "★ 조건부 양성: 필수 통과, 일부 양성"
elif mandatory_ok:
    final = "⚠️ 약한 양성: 필수는 통과했으나 양성 기준 미달"
else:
    final = "❌ 미달: 필수 기준 미충족"

log(f"종합 판정: {final}")
log()
log(f"  통과: {sum([r1,r2,r3,r4,r5])}/5 (필수 {sum([r1,r2])}/2, 양성 {optional_count}/3)")
log()

# 각 구간 영점 전체 목록 (수학자 참조용)
log("[ 부록: 발견된 고 t 영점 전체 목록 ]")
for zone_label, zeros in all_high_t_zeros.items():
    log(f"\n  {zone_label}:")
    for t_z, k_z, sig_z in zeros:
        log(f"    t={t_z:.8f}, σ={sig_z:.6f}, κ_peak={k_z:.2f}")

log()
log(f"총 소요 시간: {time.time()-t_total:.1f}초")
log(f"완료 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")

flush_to_file()
log(f"\n결과 저장: {OUTFILE}")
