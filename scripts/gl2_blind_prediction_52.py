"""
=============================================================================
[Project RDL] 결과 #52 — GL(2) 블라인드 영점 예측 (11a1 L-함수)
=============================================================================

목표:
  - GL(1) 블라인드 7/7 (#2)의 GL(2) 확장 검증
  - 11a1 L-함수에서 κ 피크 + 모노드로미 필터로 영점 위치 예측
  - LMFDB 알려진 영점은 검증용으로만 (예측 단계에서 사용 금지)

프로토콜:
  1. σ_crit + δ = 1.03 따라 t ∈ [5, 30], Δt=0.1 스윕 → 250점
  2. κ(1.03 + it) = |Λ'/Λ|² 계산
  3. scipy.signal.find_peaks 또는 수동 극대점 추출 → 후보 목록
  4. 각 후보에 모노드로미 측정 (radius=0.4, n_steps=32)
  5. 비교 기준:
     - κ-only: 피크 후보 전체
     - κ+mono: mono/π > 1.5 필터 후
  6. LMFDB 17개 영점과 비교 (|t_pred - t_true| < 0.5 매칭)
  7. Precision, Recall, F1 측정 (두 기준 모두)

성공 기준:
  - Recall ≥ 0.7 (17개 중 ≥12개 탐지)   [필수]
  - 위치 오차 < 0.5                         [필수]
  - Precision ≥ 0.5 (FP < TP)             [양성 근거]
  - F1 ≥ 0.6                               [양성 근거]
  - κ+mono가 κ-only보다 Precision +10%p   [양성 근거]

주의:
  - GL(2) 임계선 σ=1.0 (GL(1)의 σ=0.5와 다름!)
  - AFE 필수 (Dirichlet series 직접합은 조건부 수렴만)
  - Γ(s) 인자 (GL(1)의 Γ(s/2)와 다름)
  - LMFDB 영점: 예측 단계에서 절대 사용 금지

결과 파일: results/gl2_blind_prediction_52.txt
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
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "gl2_blind_prediction_52.txt"
)
os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 파라미터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SIGMA_CRIT = 1.0       # 11a1 임계선 (GL(2)!)
DELTA_OFFSET = 0.03    # κ 측정 오프셋 (영점 위 직접 측정 금지)
T_MIN = 5.0
T_MAX = 30.0
DT = 0.1               # 스윕 간격
MONO_RADIUS = 0.4      # 모노드로미 반지름
MONO_STEPS = 32        # 폐곡선 분할 수
MONO_THRESHOLD = 1.5   # mono/π > MONO_THRESHOLD → 영점으로 판정
MATCH_TOL = 0.5        # 예측/실제 매칭 허용 오차 (|t_pred - t_true| < 0.5)

# DPS
DPS_11 = 45            # 11a1 AFE 정밀도 (t<24에서 안정)
N_MAX_11 = 60          # AFE 항 수

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
# 11a1 L-함수: Lambda_11a1 (#51에서 그대로 복사)
# aₙ 계수 + AFE (Approximate Functional Equation)
# Λ(s) = Σ aₙ [xₙ^{-s} Γ(s,xₙ) + ε xₙ^{-(2-s)} Γ(2-s,xₙ)]
# σ_crit = 1, ε = +1, N = 11
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

N_COND_11 = 11
EPS_11 = 1
_an_11 = None
_precomp_11 = None


def _init_11a1():
    global _an_11, _precomp_11
    if _an_11 is not None:
        return
    def ap(p):
        if p == 11: return 1
        if p == 2:
            count = 1
            for x in range(2):
                for y in range(2):
                    if (y*y + y - x*x*x + x*x + 10*x + 20) % 2 == 0: count += 1
            return p + 1 - count
        aff = 0
        for x in range(p):
            d = (4*x*x*x - 4*x*x - 40*x - 79) % p
            if d == 0: aff += 1
            elif pow(d, (p-1)//2, p) == 1: aff += 2
        return p - aff
    sieve = [True] * (N_MAX_11 + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(N_MAX_11**0.5)+1):
        if sieve[i]:
            for j in range(i*i, N_MAX_11+1, i): sieve[j] = False
    primes = [i for i in range(2, N_MAX_11+1) if sieve[i]]
    ap_dict = {p: ap(p) for p in primes}
    apk = {}
    for p in primes:
        apk[(p,0)] = 1; apk[(p,1)] = ap_dict[p]
        pk = p; k = 1
        while pk*p <= N_MAX_11:
            pk *= p; k += 1
            if p == 11: apk[(p,k)] = ap_dict[p]**k
            else: apk[(p,k)] = ap_dict[p]*apk[(p,k-1)] - p*apk[(p,k-2)]
    an = [0] * (N_MAX_11+1); an[1] = 1
    for n in range(2, N_MAX_11+1):
        temp = n; result = 1
        for p in primes:
            if p*p > temp: break
            if temp % p == 0:
                k = 0
                while temp % p == 0: k += 1; temp //= p
                result *= apk[(p,k)]
        if temp > 1: result *= ap_dict[temp]
        an[n] = result
    # 전처리: xₙ = 2πn/√N
    saved_dps = mpmath.mp.dps
    mpmath.mp.dps = DPS_11
    sqrt_N = mpmath.sqrt(mpmath.mpf(N_COND_11))
    two_pi = 2 * mpmath.pi
    _precomp_11 = [(mpmath.mpf(an[n]), two_pi*n/sqrt_N)
                   for n in range(1, N_MAX_11+1) if an[n] != 0]
    mpmath.mp.dps = saved_dps
    _an_11 = an
    # 검증
    known = {2: -2, 3: -1, 5: 1, 7: -2, 11: 1, 13: 4}
    ok = True
    for p, exp in known.items():
        if p <= N_MAX_11 and an[p] != exp:
            print(f"  ⚠️ a_{p} = {an[p]}, expected {exp}!", flush=True)
            ok = False
    if ok:
        print(f"  [11a1 초기화] 검증 OK: a₂={an[2]}, a₃={an[3]}, a₅={an[5]}, "
              f"a₇={an[7]}, a₁₁={an[11]}", flush=True)
    print(f"  [11a1 초기화] 비영 항: {len(_precomp_11)}개 (N_MAX={N_MAX_11})", flush=True)


def Lambda_11a1(s):
    """Λ(11a1, s) via AFE"""
    saved_dps = mpmath.mp.dps
    mpmath.mp.dps = DPS_11
    try:
        _init_11a1()
        s_mp = mpmath.mpc(s)
        s_conj = 2 - s_mp        # GL(2) 함수방정식: s → 2-s
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
# 공용 함수: κ 계산, 모노드로미
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def curvature_11a1(sigma, t, h=1e-6):
    """κ(σ+it) = |Λ'/Λ|² via 중앙차분"""
    s_mp = mpmath.mpc(sigma, t)
    h_mp = mpmath.mpf(str(h))
    try:
        L0 = Lambda_11a1(s_mp)
        if abs(L0) < mpmath.mpf(10)**(-DPS_11 + 15):
            return 1e12  # 영점 근처
        Lp = Lambda_11a1(s_mp + h_mp)
        Lm = Lambda_11a1(s_mp - h_mp)
        conn = (Lp - Lm) / (2 * h_mp * L0)
        k = float(abs(conn)**2)
        return k if np.isfinite(k) else 1e12
    except Exception as e:
        print(f"    WARNING curvature t={t:.4f}: {e}", flush=True)
        return 0.0


def monodromy_11a1(t_center, radius=MONO_RADIUS, n_steps=MONO_STEPS):
    """
    Λ(11a1, s) 주위 폐곡선 모노드로미 (arg 누적 방식).
    중심: σ_crit + it_center
    Returns: |mono|/π  (영점이면 ≈2.0, 비영점이면 ≈0.0), None이면 계산 실패
    """
    center = mpmath.mpc(SIGMA_CRIT, t_center)
    phase_accum = mpmath.mpf(0)
    prev_val = None

    for j in range(n_steps + 1):
        theta = 2 * mpmath.pi * j / n_steps
        s = center + radius * mpmath.exp(1j * theta)
        try:
            val = Lambda_11a1(s)
            if abs(val) < mpmath.mpf(10)**(-DPS_11 + 15):
                return None  # 영점 경유 또는 underflow
        except Exception as e:
            print(f"    WARNING monodromy t={t_center:.4f}: {e}", flush=True)
            return None

        if prev_val is not None:
            ratio = val / prev_val
            phase_accum += mpmath.im(mpmath.log(ratio))
        prev_val = val

    return float(abs(phase_accum)) / np.pi


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LMFDB 알려진 영점 (검증용으로만! 예측 단계에서 사용 금지)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LMFDB_ZEROS_11A1 = np.array([
    6.36261389, 8.60353962, 10.03550910, 11.45125861, 13.56863906,
    15.91407260, 17.03361032, 17.94143357, 19.18572497, 20.37926046,
    22.17249029, 23.30141550, 25.20986842, 25.87640308, 27.06763523,
    28.68390988, 29.97485995
])
# → 이 변수는 evaluate_predictions()에서만 사용


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: 곡률 스윕 (블라인드 — LMFDB 영점 미사용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sweep_curvature():
    """
    σ = SIGMA_CRIT + DELTA_OFFSET 따라 t 스윕.
    κ 배열 + t 배열 반환.
    """
    sigma_sweep = SIGMA_CRIT + DELTA_OFFSET
    ts = np.arange(T_MIN, T_MAX + DT/2, DT)
    kappas = []

    log(f"\n[Step 1] κ 스윕: σ={sigma_sweep:.3f}, t ∈ [{T_MIN}, {T_MAX}], Δt={DT}")
    log(f"  총 {len(ts)}점 계산 예정")

    t0 = time.time()
    for i, t in enumerate(ts):
        k = curvature_11a1(sigma_sweep, t)
        kappas.append(k)
        if (i+1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (len(ts) - i - 1)
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
    Returns: (peak_ts, peak_kappas) — 후보 (t, κ) 쌍
    """
    log(f"\n[Step 2] κ 피크 추출")

    # 로그 스케일로 피크 찾기 (큰 κ 값에 왜곡 방지)
    log_k = np.log1p(kappas)

    if SCIPY_OK:
        # prominence = 전역 중앙값의 0.5배 이상 돌출
        median_logk = np.median(log_k)
        prominence_thresh = max(0.3, median_logk * 0.3)
        peaks, props = scipy_find_peaks(log_k, prominence=prominence_thresh, distance=2)
        log(f"  scipy.find_peaks: prominence≥{prominence_thresh:.3f}, distance≥2")
        log(f"  → {len(peaks)}개 피크 발견")

        # 피크가 너무 적으면 threshold 낮추기
        if len(peaks) < 8:
            prominence_thresh2 = max(0.1, median_logk * 0.1)
            peaks2, _ = scipy_find_peaks(log_k, prominence=prominence_thresh2, distance=2)
            log(f"  threshold 낮춤 (prominence≥{prominence_thresh2:.3f}): {len(peaks2)}개 피크")
            if len(peaks2) > len(peaks):
                peaks = peaks2

        peak_ts = ts[peaks]
        peak_kappas = kappas[peaks]
    else:
        # 수동 극대점 탐색
        log("  scipy 없음 — 수동 극대점 탐색")
        peaks_list = []
        for i in range(1, len(kappas)-1):
            if kappas[i] > kappas[i-1] and kappas[i] > kappas[i+1]:
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
        # radius: 다른 후보와의 거리 고려
        other_ts = [pt for j, pt in enumerate(peak_ts) if j != i]
        if other_ts:
            nearest = min(abs(t_cand - ot) for ot in other_ts)
            radius = min(MONO_RADIUS, nearest * 0.45)
            radius = max(radius, 0.1)
        else:
            radius = MONO_RADIUS

        mono_pi = monodromy_11a1(t_cand, radius=radius)
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
        true_zeros = LMFDB_ZEROS_11A1

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

    # 모든 (pred, true) 쌍 거리 계산
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
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

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
log("결과 #52 — GL(2) 블라인드 영점 예측 (11a1 L-함수)")
log("=" * 70)
log(f"시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"SIGMA_CRIT={SIGMA_CRIT} (GL(2)!), DELTA_OFFSET={DELTA_OFFSET}")
log(f"t ∈ [{T_MIN}, {T_MAX}], Δt={DT} → {int((T_MAX - T_MIN)/DT + 1)}점")
log(f"LMFDB 알려진 영점: {len(LMFDB_ZEROS_11A1)}개 (검증용으로만)")
log(f"MONO_RADIUS={MONO_RADIUS}, MONO_STEPS={MONO_STEPS}, MONO_THRESHOLD={MONO_THRESHOLD}")
log(f"MATCH_TOL={MATCH_TOL}")
log(f"scipy 사용: {SCIPY_OK}")
log()

t_total_start = time.time()

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
log(f"\nLMFDB 11a1 영점 ({len(LMFDB_ZEROS_11A1)}개):")
for i, tz in enumerate(LMFDB_ZEROS_11A1):
    log(f"  γ_{i+1:2d} = {tz:.8f}")

# κ-only 평가
P_konly, R_konly, F1_konly, pairs_konly = evaluate_predictions(
    peak_ts, "κ-only (피크 전체)", LMFDB_ZEROS_11A1
)
flush_to_file()

# κ+mono 평가
P_kmono, R_kmono, F1_kmono, pairs_kmono = evaluate_predictions(
    filtered_ts, "κ+mono (모노드로미 필터 후)", LMFDB_ZEROS_11A1
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
n_true = len(LMFDB_ZEROS_11A1)
tp_konly = len(pairs_konly)
fp_konly = len(peak_ts) - tp_konly
tp_kmono = len(pairs_kmono)
fp_kmono = len(filtered_ts) - tp_kmono
log(f"{'κ-only':<15} {P_konly:>12.4f} {R_konly:>10.4f} {F1_konly:>8.4f} "
    f"{tp_konly:>5} {fp_konly:>5}")
log(f"{'κ+mono':<15} {P_kmono:>12.4f} {R_kmono:>10.4f} {F1_kmono:>8.4f} "
    f"{tp_kmono:>5} {fp_kmono:>5}")
log()

# Precision 개선량
delta_P = P_kmono - P_konly
log(f"  Precision 개선 (κ+mono - κ-only): {delta_P:+.4f} ({delta_P*100:+.1f}%p)")
log()

log("[ 성공 기준 판정 ]")
criterion_results = []

# 필수 기준
r_recall = R_kmono >= 0.7
log(f"  {'✅' if r_recall else '❌'} [필수] Recall ≥ 0.7: "
    f"{R_kmono:.4f} {'PASS' if r_recall else 'FAIL'}")
criterion_results.append(('recall_필수', r_recall))

# 위치 오차 (매칭된 쌍들의 최대 오차)
if pairs_kmono:
    max_err = max(d for _, _, d in pairs_kmono)
    r_pos = max_err < MATCH_TOL
else:
    max_err = float('nan')
    r_pos = False
log(f"  {'✅' if r_pos else '❌'} [필수] 위치 오차 < {MATCH_TOL}: "
    f"max={max_err:.4f} {'PASS' if r_pos else 'FAIL'}")
criterion_results.append(('position_필수', r_pos))

# 양성 근거 기준
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

# 최종 판정
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

log(f"총 소요 시간: {time.time()-t_total_start:.1f}초")
log(f"완료 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")

flush_to_file()
print(f"\n결과 저장: {OUTFILE}", flush=True)
