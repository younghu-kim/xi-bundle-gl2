"""
=============================================================================
[Project RDL] 결과 #56 — 라마누잔 Δ 블라인드 영점 예측 (weight 12, level 1)
=============================================================================

목표:
  - GL(2) 블라인드 3번째이자 최종 곡선
  - Ramanujan Δ: weight 12, level 1, ε=+1, σ_crit=6
  - κ 피크 + 모노드로미 필터로 영점 위치 예측 (블라인드)
  - LMFDB 알려진 영점은 검증용으로만 (예측 단계 사용 금지)

프로토콜:
  1. σ_crit + δ = 6.03 따라 t ∈ [5, 30], Δt=0.2 스윕 → 126점
  2. κ(6.03 + it) = |Λ'/Λ|² 계산
  3. scipy.signal.find_peaks 극대점 추출 → 후보 목록
  4. 각 후보에 모노드로미 측정 (radius=0.4, n_steps=32)
  5. 비교 기준:
     - κ-only: 피크 후보 전체
     - κ+mono: mono/π > 1.5 필터 후
  6. LMFDB 영점과 사후 비교 (검증)
  7. Precision, Recall, F1 측정 (두 기준 모두)
  8. 11a1(#52) vs 37a1(#54) vs Δ(#56) 비교표

성공 기준:
  - Recall ≥ 0.7  [필수]
  - 위치 오차 < 0.5  [필수]
  - Precision ≥ 0.5  [양성 근거]
  - F1 ≥ 0.6  [양성 근거]
  - κ+mono가 κ-only보다 Precision +10%p  [양성 근거]

주의:
  ★   임계선 σ=6 (weight 12, k/2=6)! GL(1) σ=1/2, 타원곡선 σ=1과 다름!
  ★★  AFE: #46 스크립트에서 정확히 복사. 재구현 금지.
  ★★★ dps=60 (수학자 지시, #54 표준)
  ★★★★ 블라인드 = LMFDB 영점 예측 단계에서 사용 금지
  ★★★★★ ε=+1 → Λ(Δ, 6+it) 실수 (37a1 ε=-1과 다름!)
  ★★★★★★ #54 블라인드 파이프라인 골격 유지, Δ AFE로 교체

결과 파일: results/ramanujan_delta_blind_56.txt
=============================================================================
"""

import sys
import os
import time
import numpy as np
from math import comb

sys.path.insert(0, os.path.expanduser('~/Desktop/gdl_unified'))
sys.path.insert(0, os.path.expanduser('~/Desktop/gdl_unified/scripts'))

import mpmath

try:
    from scipy.signal import find_peaks as scipy_find_peaks
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

OUTFILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "ramanujan_delta_blind_56.txt"
)
os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 파라미터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SIGMA_CRIT   = 6.0         # ⚠️ Δ 임계선 σ = k/2 = 12/2 = 6 (GL(2) weight 12!)
WEIGHT       = 12           # Weight
EPSILON      = 1            # Root number ε = +1
N_COND       = 1            # Level (conductor)
DELTA_OFFSET = 0.03         # κ 측정 오프셋 (영점 위 직접 측정 금지)
T_MIN        = 5.0          # 수학자 지시: t ∈ [5, 30]
T_MAX        = 30.0
DT           = 0.2          # 수학자 지시: Δt=0.2
MONO_RADIUS  = 0.4          # 모노드로미 반지름
MONO_STEPS   = 64           # 폐곡선 분할 수 (수학자 지시: 64단계)
MONO_THRESHOLD = 1.5        # mono/π > MONO_THRESHOLD → 영점 판정
MATCH_TOL    = 0.5          # 예측/실제 매칭 허용 오차

# DPS & AFE
DPS_DELTA    = 60           # ★ dps=60 (수학자 지시, #54 표준)
N_MAX_COEFF  = 80           # AFE 항 수 (#46 표준: x_n=2πn, n=13에서 e^{-82} ≈ 0)

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
# Ramanujan Δ L-함수: Lambda_Delta (★ #46에서 정확히 복사 ★)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Λ(Δ,s) = Σ τ(n) · [xₙ^{-s} Γ(s, xₙ) + ε · xₙ^{-(12-s)} Γ(12-s, xₙ)]
# where xₙ = 2πn/√N = 2πn (N=1), Γ(s,x) = upper incomplete gamma
# σ_crit=6, ε=+1, N=1

_tau_cache = None
_precomp_cache = None


def compute_tau_qexpansion(n_max):
    """
    τ(n) via q-expansion: Δ(q) = q · ∏_{m≥1}(1-q^m)²⁴
    정수 연산 (정확), n=1..n_max.
    ★ #46에서 그대로 복사
    """
    N = n_max
    coeffs = [0] * N
    coeffs[0] = 1

    for m in range(1, N):
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

    tau = [0] * (n_max + 1)
    for n in range(1, n_max + 1):
        tau[n] = coeffs[n - 1]

    return tau


def _init_tables():
    """τ(n) + 전처리 테이블 초기화 ★ #46에서 그대로 복사"""
    global _tau_cache, _precomp_cache
    if _tau_cache is not None:
        return

    print("  [Δ 초기화] τ(n) 계수 계산 (n ≤ %d)..." % N_MAX_COEFF, flush=True)
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
    ok = mismatch == 0
    print(f"  τ(n) LMFDB 검증 (n=1..20): {20 - mismatch}/20 일치 {'✅' if ok else '❌'}", flush=True)

    if not ok:
        print("  ⚠️⚠️ τ(n) 불일치! 실험 신뢰성 저하!", flush=True)

    # 전처리: xₙ 값 + 0이 아닌 항만 필터
    saved_dps = mpmath.mp.dps
    mpmath.mp.dps = DPS_DELTA
    sqrt_N = mpmath.mpf(1)  # √1 = 1
    two_pi = 2 * mpmath.pi
    _precomp_cache = []
    for n in range(1, N_MAX_COEFF + 1):
        if _tau_cache[n] == 0:
            continue
        x_n = two_pi * n / sqrt_N  # = 2πn
        _precomp_cache.append((mpmath.mpf(_tau_cache[n]), x_n))
    mpmath.mp.dps = saved_dps

    print(f"  [Δ 초기화] 완료 ({time.time()-t0:.1f}초, 비영 항 {len(_precomp_cache)}개)", flush=True)
    print(f"  τ(2)={_tau_cache[2]}, τ(3)={_tau_cache[3]}, τ(5)={_tau_cache[5]}, "
          f"τ(7)={_tau_cache[7]}, ε={EPSILON}", flush=True)


def Lambda_Delta(s):
    """
    Λ(Δ,s) via AFE ★ #46에서 그대로 복사
    Λ(Δ,s) = Σ τ(n) · [xₙ^{-s} Γ(s, xₙ) + ε · xₙ^{-(12-s)} Γ(12-s, xₙ)]
    """
    saved_dps = mpmath.mp.dps
    mpmath.mp.dps = DPS_DELTA
    try:
        _init_tables()
        s_mp = mpmath.mpc(s)
        s_conj = WEIGHT - s_mp  # 12 - s
        eps = mpmath.mpf(EPSILON)  # +1

        result = mpmath.mpc(0)
        for tau_val, x_n in _precomp_cache:
            term1 = mpmath.power(x_n, -s_mp) * mpmath.gammainc(s_mp, x_n)
            term2 = eps * mpmath.power(x_n, -s_conj) * mpmath.gammainc(s_conj, x_n)
            result += tau_val * (term1 + term2)
        return result
    finally:
        mpmath.mp.dps = saved_dps


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 공용 함수: κ 계산, 모노드로미 (★ #54 구조, Δ AFE로 교체 ★)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def curvature_delta(sigma, t, h=1e-6):
    """κ(σ+it) = |Λ'/Λ|² via 중앙차분"""
    s_mp = mpmath.mpc(sigma, t)
    h_mp = mpmath.mpf(str(h))
    try:
        L0 = Lambda_Delta(s_mp)
        if abs(L0) < mpmath.mpf(10)**(-DPS_DELTA + 20):
            return 1e12  # 영점 근처
        Lp = Lambda_Delta(s_mp + h_mp)
        Lm = Lambda_Delta(s_mp - h_mp)
        conn = (Lp - Lm) / (2 * h_mp * L0)
        k = float(abs(conn)**2)
        return k if np.isfinite(k) else 1e12
    except Exception as e:
        print(f"    WARNING curvature t={t:.4f}: {e}", flush=True)
        return 0.0


def monodromy_delta(t_center, radius=MONO_RADIUS, n_steps=MONO_STEPS):
    """
    Λ(Δ, s) 주위 폐곡선 모노드로미 (arg 누적 방식).
    중심: σ_crit + it_center
    Returns: |mono|/π  (영점이면 ≈2.0, 비영점이면 ≈0.0), None이면 계산 실패
    """
    saved_dps = mpmath.mp.dps
    mpmath.mp.dps = DPS_DELTA
    try:
        center = mpmath.mpc(SIGMA_CRIT, t_center)
        phase_accum = mpmath.mpf(0)
        prev_val = None

        for j in range(n_steps + 1):
            theta = 2 * mpmath.pi * j / n_steps
            s = center + radius * mpmath.exp(1j * theta)
            try:
                val = Lambda_Delta(s)
                if abs(val) < mpmath.mpf(10)**(-DPS_DELTA + 20):
                    return None  # 영점 경유 또는 underflow
            except Exception as e:
                print(f"    WARNING monodromy t={t_center:.4f}: {e}", flush=True)
                return None

            if prev_val is not None:
                ratio = val / prev_val
                phase_accum += mpmath.im(mpmath.log(ratio))
            prev_val = val

        return float(abs(phase_accum)) / np.pi
    finally:
        mpmath.mp.dps = saved_dps


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LMFDB 알려진 영점 (검증용으로만! 예측 단계에서 사용 금지)
# 라마누잔 Δ: σ=6 임계선, t ∈ [5, 30]
# 출처: #46 결과 (dps=80 계산) + LMFDB
# 참고: Δ의 첫 영점 γ₁ ≈ 9.2224 (t<9에는 영점 없음)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LMFDB_ZEROS_DELTA = np.array([
    9.22237940,    # γ₁
    13.90754986,   # γ₂
    17.44277698,   # γ₃
    19.65651314,   # γ₄
    22.33610364,   # γ₅
    25.27463655,   # γ₆
    26.80439116,   # γ₇
    28.83168262,   # γ₈
])
# → 이 변수는 evaluate_predictions()에서만 사용
# 주의: #46에서 dps=80으로 계산한 8개 영점. t ∈ [5,30] 범위 내.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: 곡률 스윕 (블라인드 — LMFDB 영점 미사용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sweep_curvature():
    """
    σ = SIGMA_CRIT + DELTA_OFFSET 따라 t 스윕.
    κ 배열 + t 배열 반환.
    """
    sigma_sweep = SIGMA_CRIT + DELTA_OFFSET
    ts = np.arange(T_MIN, T_MAX + DT / 2, DT)
    kappas = []

    log(f"\n[Step 1] κ 스윕: σ={sigma_sweep:.3f}, t ∈ [{T_MIN}, {T_MAX}], Δt={DT}")
    log(f"  총 {len(ts)}점 계산 예정")
    log(f"  Δ: N={N_COND}, ε={EPSILON}, weight={WEIGHT}, σ_crit={SIGMA_CRIT}")
    log(f"  dps={DPS_DELTA}, N_MAX_COEFF={N_MAX_COEFF}")

    t0 = time.time()
    for i, t in enumerate(ts):
        k = curvature_delta(sigma_sweep, t)
        kappas.append(k)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(ts) - i - 1)
            log(f"  ... {i+1}/{len(ts)} ({(i+1)/len(ts)*100:.0f}%), "
                f"경과={elapsed:.0f}초, 잔여≈{eta:.0f}초, "
                f"t={t:.2f}, κ={k:.2f}")
            flush_to_file()

    kappas = np.array(kappas)
    log(f"\n  스윕 완료: {time.time()-t0:.1f}초")
    log(f"  κ 통계: min={kappas.min():.4f}, max={kappas.max():.2f}, "
        f"median={np.median(kappas):.4f}, mean={kappas.mean():.4f}")

    return ts, kappas


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: κ 피크 추출 (블라인드)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_kappa_peaks(ts, kappas):
    """
    κ 극대점 추출. scipy.find_peaks 사용 (없으면 수동 극대).
    Returns: (peak_ts, peak_kappas)
    """
    log(f"\n[Step 2] κ 피크 추출")

    log_k = np.log1p(kappas)

    if SCIPY_OK:
        median_logk = np.median(log_k)
        prominence_thresh = max(0.3, median_logk * 0.3)
        peaks, props = scipy_find_peaks(log_k, prominence=prominence_thresh, distance=2)
        log(f"  scipy.find_peaks: prominence≥{prominence_thresh:.3f}, distance≥2")
        log(f"  → {len(peaks)}개 피크 발견")

        if len(peaks) < 5:
            prominence_thresh2 = max(0.1, median_logk * 0.1)
            peaks2, _ = scipy_find_peaks(log_k, prominence=prominence_thresh2, distance=2)
            log(f"  threshold 낮춤 (prominence≥{prominence_thresh2:.3f}): {len(peaks2)}개 피크")
            if len(peaks2) > len(peaks):
                peaks = peaks2

        peak_ts = ts[peaks]
        peak_kappas = kappas[peaks]
    else:
        log("  scipy 없음 — 수동 극대점 탐색")
        peaks_list = []
        for i in range(1, len(kappas) - 1):
            if kappas[i] > kappas[i - 1] and kappas[i] > kappas[i + 1]:
                peaks_list.append(i)
        peaks = np.array(peaks_list, dtype=int)
        peak_ts = ts[peaks]
        peak_kappas = kappas[peaks]
        log(f"  → {len(peaks)}개 극대점 발견")

    if len(peaks) == 0:
        log("  ⚠️ 피크 0개 — κ 분포 이상!")
        return np.array([]), np.array([])

    log(f"\n  피크 목록 (κ-only 후보):")
    for i, (tp, kp) in enumerate(zip(peak_ts, peak_kappas)):
        log(f"    후보 {i+1:2d}: t={tp:.4f}, κ={kp:.4f}")

    return peak_ts, peak_kappas


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: 모노드로미 필터 (블라인드)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_monodromy_filter(peak_ts, peak_kappas):
    """
    각 κ 피크 후보에 모노드로미 측정.
    mono/π > MONO_THRESHOLD → 영점 후보 (κ+mono 기준).
    Returns: (mono_values, filtered_ts, filtered_kappas)
    """
    log(f"\n[Step 3] 모노드로미 필터 (radius={MONO_RADIUS}, n_steps={MONO_STEPS})")
    log(f"  기준: mono/π > {MONO_THRESHOLD}")
    log(f"  대상: {len(peak_ts)}개 후보")

    mono_values = []
    filtered_ts = []
    filtered_kappas = []

    for i, (t_cand, k_cand) in enumerate(zip(peak_ts, peak_kappas)):
        # radius: 다른 후보와의 거리 고려 (겹침 방지)
        other_ts = [pt for j, pt in enumerate(peak_ts) if j != i]
        if other_ts:
            nearest = min(abs(t_cand - ot) for ot in other_ts)
            radius = min(MONO_RADIUS, nearest * 0.45)
            radius = max(radius, 0.1)
        else:
            radius = MONO_RADIUS

        mono_pi = monodromy_delta(t_cand, radius=radius)
        if mono_pi is None:
            log(f"    [{i+1:2d}] t={t_cand:.4f}, κ={k_cand:.4f}, mono=FAIL (영점 경유?)")
            mono_values.append(np.nan)
            continue

        mono_values.append(mono_pi)
        marker = " ← ★ 영점" if mono_pi > MONO_THRESHOLD else ""
        log(f"    [{i+1:2d}] t={t_cand:.4f}, κ={k_cand:.4f}, "
            f"r={radius:.3f}, mono/π={mono_pi:.4f}{marker}")

        if mono_pi > MONO_THRESHOLD:
            filtered_ts.append(t_cand)
            filtered_kappas.append(k_cand)

    mono_values = np.array(mono_values)
    filtered_ts = np.array(filtered_ts)
    filtered_kappas = np.array(filtered_kappas)

    log(f"\n  κ+mono 필터 결과: {len(filtered_ts)}개 후보 → 영점 예측")
    if len(filtered_ts) == 0:
        log("  ⚠️ 모노드로미 필터 후 0개 — threshold 조정 또는 실패")

    return mono_values, filtered_ts, filtered_kappas


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: 평가 (LMFDB와 비교)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_predictions(pred_ts, label, true_zeros=None):
    """
    예측 영점 목록 vs LMFDB 영점 비교.
    |t_pred - t_true| < MATCH_TOL → 매칭.
    Returns: (precision, recall, f1, matched_pairs)
    """
    if true_zeros is None:
        true_zeros = LMFDB_ZEROS_DELTA

    n_true = len(true_zeros)
    pred_ts = np.array(pred_ts) if len(pred_ts) > 0 else np.array([])

    log(f"\n[평가: {label}]")
    log(f"  예측 {len(pred_ts)}개 vs 실제 {n_true}개 (매칭 허용 오차: {MATCH_TOL})")

    if len(pred_ts) == 0:
        log(f"  예측 0개 → P=0, R=0, F1=0")
        return 0.0, 0.0, 0.0, []

    # 매칭 (greedy: 가장 가까운 쌍)
    true_matched = [False] * n_true
    pred_matched = [False] * len(pred_ts)
    matched_pairs = []
    errors = []

    dists = []
    for pi, pt in enumerate(pred_ts):
        for ti, tt in enumerate(true_zeros):
            dists.append((abs(pt - tt), pi, ti, pt, tt))
    dists.sort()

    for dist, pi, ti, pt, tt in dists:
        if pred_matched[pi] or true_matched[ti]:
            continue
        if dist < MATCH_TOL:
            pred_matched[pi] = True
            true_matched[ti] = True
            matched_pairs.append((pt, tt, dist))
            errors.append(dist)

    tp = len(matched_pairs)
    fp = len(pred_ts) - tp
    fn = n_true - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    log(f"  TP={tp}, FP={fp}, FN={fn}")
    log(f"  Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    if matched_pairs:
        log(f"  위치 오차: mean={np.mean(errors):.4f}, max={np.max(errors):.4f}")
        log(f"\n  매칭 상세:")
        for pt, tt, d in sorted(matched_pairs, key=lambda x: x[1]):
            log(f"    예측 {pt:.4f} ↔ 실제 {tt:.8f} (오차 {d:.4f})")
    else:
        log("  매칭 없음")

    # 미탐지 영점
    undetected = [true_zeros[i] for i in range(n_true) if not true_matched[i]]
    if undetected:
        log(f"\n  미탐지 실제 영점: {[f'{t:.4f}' for t in undetected]}")

    # 오탐 예측
    false_positives = [pred_ts[i] for i in range(len(pred_ts)) if not pred_matched[i]]
    if false_positives:
        log(f"  오탐 예측: {[f'{t:.4f}' for t in false_positives]}")

    return precision, recall, f1, matched_pairs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log("=" * 70)
log("결과 #56 — 라마누잔 Δ 블라인드 영점 예측 (weight 12, level 1)")
log("=" * 70)
log(f"시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Δ: weight={WEIGHT}, level={N_COND}, ε={EPSILON}")
log(f"SIGMA_CRIT={SIGMA_CRIT} (= k/2 = 12/2 = 6) ⚠️ NOT σ=1/2 or σ=1!")
log(f"t ∈ [{T_MIN}, {T_MAX}], Δt={DT} → {int((T_MAX - T_MIN)/DT + 1)}점")
log(f"DPS={DPS_DELTA}, N_MAX_COEFF={N_MAX_COEFF}")
log(f"LMFDB 알려진 영점: {len(LMFDB_ZEROS_DELTA)}개 (검증용으로만, t ∈ [{T_MIN},{T_MAX}])")
log(f"MONO_RADIUS={MONO_RADIUS}, MONO_STEPS={MONO_STEPS}, MONO_THRESHOLD={MONO_THRESHOLD}")
log(f"MATCH_TOL={MATCH_TOL}")
log(f"scipy 사용: {SCIPY_OK}")
log()

t_total_start = time.time()

# ── AFE 검증 (빠른 sanity check) ──
log("[사전 검증] AFE + 함수 방정식")
_init_tables()

# 함수 방정식: Λ(Δ, s) = Λ(Δ, 12-s) (ε=+1)
for sp_re, sp_im in [(7.3, 5.0), (5.5, 12.0)]:
    sp = mpmath.mpc(sp_re, sp_im)
    L1 = Lambda_Delta(sp)
    L2 = Lambda_Delta(WEIGHT - sp)
    denom = max(float(abs(L1)), 1e-50)
    rel_err = float(abs(L1 - EPSILON * L2)) / denom
    ok = "✅" if rel_err < 1e-10 else "❌"
    log(f"  Λ({sp_re}+{sp_im}i) vs εΛ(12-s): rel_err={rel_err:.2e} {ok}")

# Λ(Δ, 6+it) 실수 확인 (ε=+1)
for t_test in [7.0, 15.0]:
    s_test = mpmath.mpc(SIGMA_CRIT, t_test)
    val = Lambda_Delta(s_test)
    re_part = float(mpmath.re(val))
    im_part = float(mpmath.im(val))
    ratio = abs(im_part) / max(abs(re_part), 1e-50)
    ok = "✅" if ratio < 1e-10 else "❌"
    log(f"  Λ(6+{t_test}i): |Im/Re|={ratio:.2e} {ok}")

log()
flush_to_file()

# ── Step 1: κ 스윕 (블라인드) ──
ts, kappas = sweep_curvature()
flush_to_file()

# ── Step 2: κ 피크 추출 (블라인드) ──
peak_ts, peak_kappas = extract_kappa_peaks(ts, kappas)
flush_to_file()

if len(peak_ts) == 0:
    log("\n⚠️⚠️ κ 피크 0개 — 실험 중단!")
    flush_to_file()
    sys.exit(1)

# ── Step 3: 모노드로미 필터 (블라인드) ──
mono_values, filtered_ts, filtered_kappas = apply_monodromy_filter(peak_ts, peak_kappas)
flush_to_file()

# ── Step 4: 평가 (LMFDB 사용) ──
log("\n" + "=" * 70)
log("[Step 4] 평가 — LMFDB 영점과 비교 (검증 단계)")
log("=" * 70)
log(f"\nLMFDB Δ 영점 ({len(LMFDB_ZEROS_DELTA)}개, t ∈ [{T_MIN},{T_MAX}]):")
for i, tz in enumerate(LMFDB_ZEROS_DELTA):
    log(f"  γ_{i+1:2d} = {tz:.8f}")

# κ-only 평가
P_konly, R_konly, F1_konly, pairs_konly = evaluate_predictions(
    peak_ts, "κ-only (피크 전체)", LMFDB_ZEROS_DELTA
)
flush_to_file()

# κ+mono 평가
P_kmono, R_kmono, F1_kmono, pairs_kmono = evaluate_predictions(
    filtered_ts, "κ+mono (모노드로미 필터 후)", LMFDB_ZEROS_DELTA
)
flush_to_file()

# ── 최종 보고 ──
log("\n" + "=" * 70)
log("최종 결과 요약")
log("=" * 70)
log()

log("[ κ 스윕 ]")
log(f"  t 범위: [{T_MIN}, {T_MAX}], Δt={DT}, 총 {len(ts)}점")
log(f"  σ 오프셋: {SIGMA_CRIT} + {DELTA_OFFSET} = {SIGMA_CRIT + DELTA_OFFSET}")
log(f"  κ 중앙값: {np.median(kappas):.4f}, 최대: {kappas.max():.2f}")
log()

log("[ 후보 비교 ]")
log(f"  κ-only  후보: {len(peak_ts)}개")
log(f"  κ+mono  후보: {len(filtered_ts)}개")
log()

log("[ 성능 비교 ]")
log(f"{'기준':<15} {'Precision':>12} {'Recall':>10} {'F1':>8} {'TP':>5} {'FP':>5}")
log("-" * 55)
n_true = len(LMFDB_ZEROS_DELTA)
tp_konly  = len(pairs_konly)
fp_konly  = len(peak_ts) - tp_konly
tp_kmono  = len(pairs_kmono)
fp_kmono  = len(filtered_ts) - tp_kmono
log(f"{'κ-only':<15} {P_konly:>12.4f} {R_konly:>10.4f} {F1_konly:>8.4f} "
    f"{tp_konly:>5} {fp_konly:>5}")
log(f"{'κ+mono':<15} {P_kmono:>12.4f} {R_kmono:>10.4f} {F1_kmono:>8.4f} "
    f"{tp_kmono:>5} {fp_kmono:>5}")
log()

delta_P = P_kmono - P_konly
log(f"  Precision 개선 (κ+mono - κ-only): {delta_P:+.4f} ({delta_P*100:+.1f}%p)")
log()

log("[ 성공 기준 판정 ]")
criterion_results = []

r_recall = R_kmono >= 0.7
log(f"  {'✅' if r_recall else '❌'} [필수] Recall ≥ 0.7: "
    f"{R_kmono:.4f} {'PASS' if r_recall else 'FAIL'}")
criterion_results.append(('recall_필수', r_recall))

if pairs_kmono:
    max_err = max(d for _, _, d in pairs_kmono)
    r_pos = max_err < MATCH_TOL
else:
    max_err = float('nan')
    r_pos = False
log(f"  {'✅' if r_pos else '❌'} [필수] 위치 오차 < {MATCH_TOL}: "
    f"max={max_err:.4f} {'PASS' if r_pos else 'FAIL'}")
criterion_results.append(('position_필수', r_pos))

r_precision = P_kmono >= 0.5
log(f"  {'✅' if r_precision else '❌'} [양성] Precision ≥ 0.5: "
    f"{P_kmono:.4f} {'PASS' if r_precision else 'FAIL'}")
criterion_results.append(('precision_양성', r_precision))

r_f1 = F1_kmono >= 0.6
log(f"  {'✅' if r_f1 else '❌'} [양성] F1 ≥ 0.6: "
    f"{F1_kmono:.4f} {'PASS' if r_f1 else 'FAIL'}")
criterion_results.append(('f1_양성', r_f1))

r_improvement = delta_P > 0.10
log(f"  {'✅' if r_improvement else '❌'} [양성] κ+mono Precision 개선 > +10%p: "
    f"{delta_P*100:+.1f}%p {'PASS' if r_improvement else 'FAIL'}")
criterion_results.append(('improvement_양성', r_improvement))

log()
n_pass = sum(v for _, v in criterion_results)
n_total = len(criterion_results)
n_mandatory = sum(1 for k, v in criterion_results if '필수' in k)
n_mandatory_pass = sum(v for k, v in criterion_results if '필수' in k)
log(f"  통과: {n_pass}/{n_total} (필수 {n_mandatory_pass}/{n_mandatory})")
log()

mandatory_ok = n_mandatory_pass == n_mandatory
if mandatory_ok and n_pass >= 4:
    verdict = "★★★ 완전 양성"
elif mandatory_ok and n_pass >= 3:
    verdict = "★★ 양성"
elif mandatory_ok:
    verdict = "★ 조건부 양성 (필수 기준 충족, 양성 기준 부족)"
else:
    verdict = "❌ 음성 (필수 기준 미충족)"

log(f"최종 판정: {verdict}")
log()

# ── GL(2) 블라인드 3곡선 비교표 ──
log("=" * 70)
log("[ GL(2) 블라인드 예측 비교: 11a1 (#52) vs 37a1 (#54) vs Δ (#56) ]")
log("=" * 70)
log()
log(f"{'항목':<30} {'11a1 (#52)':>15} {'37a1 (#54)':>15} {'Δ (#56)':>15}")
log("-" * 75)
log(f"{'L-함수':<30} {'타원곡선':>15} {'타원곡선':>15} {'모듈러 형식':>15}")
log(f"{'weight':<30} {'2':>15} {'2':>15} {'12':>15}")
log(f"{'conductor (level)':<30} {'11':>15} {'37':>15} {'1':>15}")
log(f"{'rank':<30} {'0':>15} {'1':>15} {'N/A':>15}")
log(f"{'root number ε':<30} {'+1':>15} {'-1':>15} {'+1':>15}")
log(f"{'σ_crit':<30} {'1':>15} {'1':>15} {'6':>15}")
log(f"{'스윕 범위':<30} {'[5, 30]':>15} {'[3, 30]':>15} {f'[{T_MIN:.0f}, {T_MAX:.0f}]':>15}")
log(f"{'Δt':<30} {'0.1':>15} {'0.2':>15} {str(DT):>15}")
log(f"{'DPS':<30} {'45':>15} {'60':>15} {str(DPS_DELTA):>15}")
log(f"{'LMFDB 영점 수':<30} {'17':>15} {'23':>15} {str(len(LMFDB_ZEROS_DELTA)):>15}")
log(f"{'κ 피크 수':<30} {'17':>15} {'22':>15} {str(len(peak_ts)):>15}")
log(f"{'κ-only TP':<30} {'16':>15} {'22':>15} {str(tp_konly):>15}")
log(f"{'κ-only FP':<30} {'1':>15} {'0':>15} {str(fp_konly):>15}")
log(f"{'κ-only Precision':<30} {'0.9412':>15} {'1.0000':>15} {P_konly:>15.4f}")
log(f"{'κ-only Recall':<30} {'0.9412':>15} {'0.9565':>15} {R_konly:>15.4f}")
log(f"{'κ-only F1':<30} {'0.9412':>15} {'0.9778':>15} {F1_konly:>15.4f}")
log(f"{'κ+mono TP':<30} {'16':>15} {'22':>15} {str(tp_kmono):>15}")
log(f"{'κ+mono FP':<30} {'1':>15} {'0':>15} {str(fp_kmono):>15}")
log(f"{'κ+mono Precision':<30} {'0.9412':>15} {'1.0000':>15} {P_kmono:>15.4f}")
log(f"{'κ+mono Recall':<30} {'0.9412':>15} {'0.9565':>15} {R_kmono:>15.4f}")
log(f"{'κ+mono F1':<30} {'0.9412':>15} {'0.9778':>15} {F1_kmono:>15.4f}")
log()

# ── κ near/far 원시값 기록 (검토자 피드백 반영) ──
log("[ κ near/far 원시값 ]")
if pairs_kmono:
    near_kappas = []
    for pt, tt, d in pairs_kmono:
        idx = np.argmin(np.abs(ts - pt))
        near_kappas.append(kappas[idx])
    log(f"  near (TP 위치): {[f'{k:.1f}' for k in near_kappas]}")
    log(f"  near median: {np.median(near_kappas):.1f}")

    # far: 영점에서 1.0 이상 떨어진 점들의 κ
    matched_ts = np.array([pt for pt, _, _ in pairs_kmono])
    far_mask = np.array([np.min(np.abs(matched_ts - t)) > 1.0 if len(matched_ts) > 0 else True
                         for t in ts])
    far_kappas = kappas[far_mask]
    if len(far_kappas) > 0:
        log(f"  far median: {np.median(far_kappas):.4f}")
        log(f"  near/far ratio: {np.median(near_kappas)/np.median(far_kappas):.1f}×")
log()

log(f"총 소요 시간: {time.time()-t_total_start:.1f}초")
log(f"완료 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")

flush_to_file()
print(f"\n결과 저장: {OUTFILE}", flush=True)
