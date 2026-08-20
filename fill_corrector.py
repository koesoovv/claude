# fill_corrector.py ── 급수 정확도 적응형 보정 모듈 (AI)
#
# [역할]
#   기존 음향 기반 채움률 추정기(smart_water_core)는 정지 시점에 체계적 편향을 가진다.
#   499회 실험 분석 결과 내부 추정값이 실측보다 평균 +3.07%p 높게 나와(= 일찍 멈춤)
#   최종 채움 오차가 평균 -1.23%p로 치우쳐 있었다.
#
#   이 모듈은 제어 로직을 바꾸지 않고, "지금 멈추면 최종 오차가 얼마일지"를
#   학습된 회귀 모델로 예측해 정지 임계값을 이동시킨다.
#
#       기존:  fill_pct >= TARGET - (STOP_ADVANCE + zone_adv + bridge_adv)
#       개선:  fill_pct >= TARGET - (STOP_ADVANCE + zone_adv + bridge_adv + e_hat)
#
#   e_hat < 0 (덜 채울 것으로 예측) → 임계값이 올라가 더 늦게 정지
#   e_hat > 0 (넘칠 것으로 예측)   → 임계값이 내려가 더 일찍 정지
#
# [학습 방식]
#   1) 오프라인 사전학습 : trial_summary.csv 누적분으로 Ridge 회귀 초기 가중치 생성
#   2) 온라인 적응 학습  : 급수 1회가 끝나고 실측 채움률이 입력될 때마다
#                          망각계수 기반 재귀최소자승(RLS)으로 가중치를 갱신한다.
#                          → 설치된 개체의 수압·마이크 특성·주변 소음에 스스로 맞춰간다.
#
# [안전 장치]
#   - 보정량 ±MAX_CORRECTION_PCT 로 클리핑 (모델이 폭주해도 제어가 망가지지 않음)
#   - 학습 표본 MIN_SAMPLES 미만이면 보정 비활성 (콜드 스타트 보호)
#   - 학습 분포를 크게 벗어난 입력(처음 보는 형태의 용기)은 보정량을 축소
#   - 특징값에 NaN/Inf 또는 미확정(0) 값이 있으면 보정 0 반환
#
# [의존성] numpy 만 사용. 학습·추론 모두 표준 라이브러리 + numpy 로 동작하며
#          라즈베리파이에서 추론 1회 < 0.05ms, 갱신 1회 < 1ms.

from __future__ import annotations

import json                                                        # 모델 가중치 영속화
import math                                                        # 유한값 검사
import os                                                          # 파일 존재 확인
import threading                                                   # 코어 스레드와 동시 접근 보호
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ════════════════════════════════════════════════════════════════════
#  설정값
# ════════════════════════════════════════════════════════════════════
MODEL_FILE          = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'fill_corrector_model.json')     # 가중치 저장 경로

MAX_CORRECTION_PCT  = 6.0      # 보정량 클리핑 한계(%p). 검증에서 ±6이 최적
RIDGE_ALPHA         = 30.0     # 정규화 강도. 미지 용기 일반화와 적합도의 균형점
FORGET_FACTOR       = 0.97     # 온라인 망각계수. 유효 기억 ≈ 최근 30여 회
MIN_SAMPLES         = 30       # 이 표본 수 미만이면 보정하지 않음
Z_GUARD             = 4.0      # 표준화 특징 |z|가 이 값을 넘으면 분포 이탈로 간주
OOD_SHRINK_MIN      = 0.3      # 분포 이탈 시 보정량 축소 하한 배율
BIAS_RIDGE          = 1e-6     # 편향항은 정규화하지 않는다(수치 안정용 극소값만)


def _ridge_matrix(dim: int) -> np.ndarray:
    """정규화 대각행렬. 마지막 성분(편향항)만 페널티에서 제외한다."""
    diag = np.full(dim, RIDGE_ALPHA)
    diag[-1] = BIAS_RIDGE                                          # 평균 편향 보정을 죽이지 않기 위함
    return np.diag(diag)

# 특징 순서는 학습·추론이 반드시 동일해야 하므로 여기서 한 번만 정의한다.
FEATURE_NAMES = [
    'Decision_Fill_Pct',        # 정지 판단 시점 내부 추정 채움률(%)
    'Estimated_Stop_Air_cm',    # 정지 판단 시점 추정 공기층(cm)
    'Stop_Freq_Hz',             # 정지 판단 시점 안정화 주파수(Hz)
    'Stop_Elapsed_Sec',         # 소음 보정 이후 경과 시간(s)
    'Init_Air_Locked_cm',       # 확정된 초기 공기층 H(cm)
    'Diameter_cm',              # 추정 지름 D(cm)
    'Reg_R2',                   # 이론 곡선 회귀 적합도
    'Target_Fill_Pct',          # 목표 채움률(%)
    'Dynamic_Q',                # 동적 유량 추정(ml/s)
    'Bridge_Used',              # 시간/유량 브릿지 사용 여부(0/1)
    'Initial_Stable_Freq_Hz',   # 초기 안정 주파수(Hz)
    'Max_Freq_Seen_Hz',         # 관측 최대 주파수(Hz)
    'Noise_Floor_Level',        # 노이즈 플로어
    'D_Candidate_MAD_cm',       # D 후보 산포(cm) — 지름 추정 신뢰도
    'Fill_By_Volume_Pct',       # 시간×유량 부피적분 기반 채움률(%)
    'Fill_Gap_Pct',             # 음향 추정 - 부피적분 (두 센서의 불일치량)
    'Air_Ratio',                # 잔여 공기층 / 초기 공기층
]
N_FEATURES = len(FEATURE_NAMES)


# ════════════════════════════════════════════════════════════════════
#  특징 생성
# ════════════════════════════════════════════════════════════════════
def _f(value: Any, default: float = 0.0) -> float:
    """어떤 값이든 유한한 float으로 변환한다."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def build_features(
    *,
    decision_fill_pct: float,
    current_air_cm: float,
    stop_freq_hz: float,
    elapsed_sec: float,
    init_air_cm: float,
    diameter_cm: float,
    reg_r2: float,
    target_fill_pct: float,
    dynamic_q: float,
    bridge_used: bool,
    initial_stable_freq_hz: float,
    max_freq_seen_hz: float,
    noise_floor_level: float,
    d_candidate_mad_cm: float,
    flow_rate_ml_s: float,
) -> Optional[np.ndarray]:
    """
    코어의 정지 판단 시점 상태로부터 특징 벡터를 만든다.
    모델이 성립할 수 없는 상태(초기 공기층/지름 미확정)면 None을 반환한다.
    """
    init_air = _f(init_air_cm)
    diameter = _f(diameter_cm)
    if init_air <= 0.0 or diameter <= 0.0:                         # H·D 미확정이면 보정 불가
        return None

    elapsed = max(0.0, _f(elapsed_sec))
    area_cm2 = math.pi * (diameter * 0.5) ** 2                     # 용기 단면적(cm^2)
    risen_cm = _f(flow_rate_ml_s) * elapsed / area_cm2             # 흘린 부피로 계산한 상승 수위(cm)
    fill_by_volume = 100.0 * risen_cm / init_air                   # 부피적분 기반 채움률(%)

    decision_fill = _f(decision_fill_pct)

    vec = np.array([
        decision_fill,
        _f(current_air_cm),
        _f(stop_freq_hz),
        elapsed,
        init_air,
        diameter,
        _f(reg_r2),
        _f(target_fill_pct),
        _f(dynamic_q),
        1.0 if bridge_used else 0.0,
        _f(initial_stable_freq_hz),
        _f(max_freq_seen_hz),
        _f(noise_floor_level),
        _f(d_candidate_mad_cm),
        fill_by_volume,
        decision_fill - fill_by_volume,                            # 음향 vs 부피 불일치
        _f(current_air_cm) / init_air,                             # 잔여 공기층 비율
    ], dtype=float)

    if not np.all(np.isfinite(vec)):                               # 하나라도 비정상이면 포기
        return None
    return vec


def features_from_core(core: Any, *, fill_pct: float, current_air: float,
                       stable_f: float, elapsed: float,
                       diameter: float, init_air: float,
                       bridge_used: bool) -> Optional[np.ndarray]:
    """smart_water_core 모듈 객체에서 직접 특징을 만든다 (코어 내부 호출용)."""
    return build_features(
        decision_fill_pct=fill_pct,
        current_air_cm=current_air,
        stop_freq_hz=stable_f,
        elapsed_sec=elapsed,
        init_air_cm=init_air,
        diameter_cm=diameter,
        reg_r2=getattr(core, 'last_r2', 0.0),
        target_fill_pct=getattr(core, 'TARGET_FILL_PCT', 0.0),
        dynamic_q=getattr(core, 'dynamic_Q', 0.0),
        bridge_used=bridge_used,
        initial_stable_freq_hz=getattr(core, 'initial_stable_freq', 0.0),
        max_freq_seen_hz=getattr(core, 'max_freq_seen', 0.0),
        noise_floor_level=getattr(core, 'noise_floor_level', 0.0),
        d_candidate_mad_cm=getattr(core, 'last_d_candidate_mad', 0.0),
        flow_rate_ml_s=getattr(core, 'FLOW_RATE', 0.0),
    )


# ════════════════════════════════════════════════════════════════════
#  적응형 보정 모델
# ════════════════════════════════════════════════════════════════════
class FillCorrector:
    """
    Ridge 회귀 + 망각계수 RLS 온라인 학습기.

    내부적으로 표준화 공간에서 정규방정식의 충분통계량만 들고 있는다.

        A = Σ λ^(n-i) · x_i x_iᵀ + αI      (D+1 × D+1, 편향항 포함)
        b = Σ λ^(n-i) · x_i y_i
        w = A⁻¹ b

    표본 하나를 통째로 저장하지 않으므로 메모리가 시행 수와 무관하게 일정하다.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.mean = np.zeros(N_FEATURES)                           # 표준화 평균(사전학습에서 고정)
        self.scale = np.ones(N_FEATURES)                           # 표준화 표준편차
        self.dim = N_FEATURES + 1                                  # 편향항 포함 차원
        self.A = _ridge_matrix(self.dim)                           # 정규방정식 좌변
        self.b = np.zeros(self.dim)                                # 정규방정식 우변
        self.w = np.zeros(self.dim)                                # 가중치
        self.n_samples = 0                                         # 누적 학습 표본 수
        self.n_online = 0                                          # 그중 온라인으로 학습한 수
        self.enabled = True                                        # 보정 on/off
        self.trained_at = ''                                       # 사전학습 시각

    # ── 표준화 ───────────────────────────────────────────────────
    def _standardize(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.scale
        return np.append(z, 1.0)                                   # 편향항 추가

    def set_normalizer(self, mean: np.ndarray, scale: np.ndarray) -> None:
        """사전학습 데이터의 평균/표준편차를 표준화 기준으로 고정한다."""
        scale = np.asarray(scale, dtype=float).copy()
        scale[~np.isfinite(scale) | (scale <= 1e-9)] = 1.0         # 0 분산 특징 보호
        self.mean = np.asarray(mean, dtype=float).copy()
        self.scale = scale

    def _solve(self) -> None:
        """현재 충분통계량으로 가중치를 다시 푼다."""
        try:
            self.w = np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:                              # 특이행렬이면 유사역행렬로 대체
            self.w = np.linalg.lstsq(self.A, self.b, rcond=None)[0]
        if not np.all(np.isfinite(self.w)):                        # 수치 파탄 시 보정 포기
            self.w = np.zeros(self.dim)

    # ── 학습 ─────────────────────────────────────────────────────
    def fit_batch(self, X: np.ndarray, y: np.ndarray) -> None:
        """오프라인 사전학습. 누적 실험 데이터로 초기 가중치를 만든다."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.set_normalizer(X.mean(axis=0), X.std(axis=0))
        Z = np.hstack([(X - self.mean) / self.scale, np.ones((len(X), 1))])
        with self._lock:
            self.A = Z.T @ Z + _ridge_matrix(self.dim)
            self.b = Z.T @ y
            self.n_samples = len(X)
            self.n_online = 0
            self._solve()

    def update(self, x: np.ndarray, fill_error_pct: float) -> None:
        """
        온라인 학습 1스텝. 급수가 끝나고 실측 채움률이 확정될 때 호출한다.

        x              : 그 시행의 정지 시점 특징 벡터
        fill_error_pct : 실측 채움률 - 목표 채움률 (= 그 시행의 최종 오차)
        """
        x = np.asarray(x, dtype=float)
        y = _f(fill_error_pct)
        if x.shape != (N_FEATURES,) or not np.all(np.isfinite(x)):
            return
        if abs(y) > 60.0:                                          # 실측 입력 오타 등 이상치 배제
            return

        z = self._standardize(x)
        with self._lock:
            self.A *= FORGET_FACTOR                                # 오래된 통계를 서서히 잊는다
            self.b *= FORGET_FACTOR
            self.A += np.outer(z, z)
            self.b += z * y
            self.n_samples += 1
            self.n_online += 1
            self._solve()

    # ── 추론 ─────────────────────────────────────────────────────
    def predict(self, x: Optional[np.ndarray]) -> Tuple[float, Dict[str, Any]]:
        """
        지금 멈추면 발생할 최종 채움 오차(%p)를 예측한다.
        반환: (보정량, 진단정보). 보정 불가 상황이면 0.0을 반환한다.
        """
        info: Dict[str, Any] = {'active': False, 'raw': 0.0, 'shrink': 1.0,
                                'reason': 'ok', 'n_samples': self.n_samples}

        if not self.enabled:
            info['reason'] = 'disabled'
            return 0.0, info
        if x is None:
            info['reason'] = 'no_features'
            return 0.0, info
        if self.n_samples < MIN_SAMPLES:
            info['reason'] = 'insufficient_data'
            return 0.0, info

        with self._lock:
            w = self.w.copy()
        z = self._standardize(np.asarray(x, dtype=float))
        if not np.all(np.isfinite(z)):
            info['reason'] = 'bad_features'
            return 0.0, info

        raw = float(z @ w)
        if not math.isfinite(raw):
            info['reason'] = 'nan_output'
            return 0.0, info

        # 학습 분포 이탈 정도에 따라 보정량을 줄인다 (처음 보는 용기 보호).
        max_z = float(np.max(np.abs(z[:-1])))
        if max_z > Z_GUARD:
            shrink = max(OOD_SHRINK_MIN, Z_GUARD / max_z)
            info['reason'] = 'out_of_distribution'
        else:
            shrink = 1.0
        corrected = float(np.clip(raw * shrink, -MAX_CORRECTION_PCT, MAX_CORRECTION_PCT))

        info.update(active=True, raw=raw, shrink=shrink, max_z=max_z, value=corrected)
        return corrected, info

    # ── 영속화 ───────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'version': 1,
                'feature_names': FEATURE_NAMES,
                'ridge_alpha': RIDGE_ALPHA,
                'forget_factor': FORGET_FACTOR,
                'mean': self.mean.tolist(),
                'scale': self.scale.tolist(),
                'A': self.A.tolist(),
                'b': self.b.tolist(),
                'w': self.w.tolist(),
                'n_samples': self.n_samples,
                'n_online': self.n_online,
                'trained_at': self.trained_at,
            }

    def load_dict(self, data: Dict[str, Any]) -> bool:
        """저장된 가중치를 복원한다. 특징 구성이 바뀌었으면 거부한다."""
        if list(data.get('feature_names', [])) != FEATURE_NAMES:
            return False
        try:
            mean = np.array(data['mean'], dtype=float)
            scale = np.array(data['scale'], dtype=float)
            A = np.array(data['A'], dtype=float)
            b = np.array(data['b'], dtype=float)
        except (KeyError, TypeError, ValueError):
            return False
        if A.shape != (self.dim, self.dim) or b.shape != (self.dim,):
            return False
        with self._lock:
            self.set_normalizer(mean, scale)
            self.A, self.b = A, b
            self.n_samples = int(data.get('n_samples', 0))
            self.n_online = int(data.get('n_online', 0))
            self.trained_at = str(data.get('trained_at', ''))
            self._solve()
        return True

    def save(self, path: str = MODEL_FILE) -> bool:
        try:
            tmp = path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:             # 원자적 교체로 손상 방지
                json.dump(self.to_dict(), f, ensure_ascii=False)
            os.replace(tmp, path)
            return True
        except OSError:
            return False

    def load(self, path: str = MODEL_FILE) -> bool:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return self.load_dict(json.load(f))
        except (OSError, ValueError):
            return False


# ════════════════════════════════════════════════════════════════════
#  모듈 전역 인스턴스 — 코어/기록기가 공유한다
# ════════════════════════════════════════════════════════════════════
_model = FillCorrector()
_loaded = _model.load()                                            # 저장된 가중치가 있으면 자동 복원
_last_features: Optional[np.ndarray] = None                        # 직전 정지 시점 특징(온라인 학습용)
_features_lock = threading.Lock()


def is_ready() -> bool:
    """보정을 실제로 적용할 수 있는 상태인지."""
    return _model.enabled and _model.n_samples >= MIN_SAMPLES


def get_model() -> FillCorrector:
    return _model


def set_enabled(flag: bool) -> None:
    _model.enabled = bool(flag)


def predict_correction(x: Optional[np.ndarray]) -> Tuple[float, Dict[str, Any]]:
    """정지 임계값에 더할 보정량(%p)을 구한다."""
    return _model.predict(x)


def remember_features(x: Optional[np.ndarray]) -> None:
    """정지 순간의 특징을 기억해 둔다. 나중에 실측값이 들어오면 학습에 쓴다."""
    global _last_features
    with _features_lock:
        _last_features = None if x is None else np.asarray(x, dtype=float).copy()


def take_features() -> Optional[np.ndarray]:
    """기억해 둔 특징을 꺼내고 비운다(1회성)."""
    global _last_features
    with _features_lock:
        x, _last_features = _last_features, None
    return x


def observe(fill_error_pct: float, *, features: Optional[np.ndarray] = None,
            save: bool = True) -> bool:
    """
    실측 채움률이 확정됐을 때 호출하는 온라인 학습 진입점.

    fill_error_pct : 실측 채움률 - 목표 채움률
    features       : 생략하면 remember_features()로 저장해 둔 값을 사용
    반환           : 학습이 수행됐으면 True
    """
    x = features if features is not None else take_features()
    if x is None:
        return False
    before = _model.n_samples
    _model.update(x, fill_error_pct)
    if _model.n_samples == before:                                 # 이상치로 거부된 경우
        return False
    if save:
        _model.save()
    return True


def status() -> Dict[str, Any]:
    """GUI/로그 표시용 요약."""
    return {
        'enabled': _model.enabled,
        'ready': is_ready(),
        'n_samples': _model.n_samples,
        'n_online': _model.n_online,
        'trained_at': _model.trained_at,
        'model_loaded': _loaded,
        'max_correction_pct': MAX_CORRECTION_PCT,
    }


# ════════════════════════════════════════════════════════════════════
#  [v5.36-SHAPE] 실측 정합 용기 치수 추정
# ════════════════════════════════════════════════════════════════════
#  코어의 D 추정값은 '실제 치수'가 아니라 이 음향 모델에서 맞아떨어지는
#  유효 파라미터다. 499회 분석에서 실제 지름 대비 평균 +1.00cm 과대였고,
#  실제 치수로 바꿔 넣으면 채움률 정확도가 오히려 나빠졌다 (MAE 4.27 -> 7.92).
#
#  그래서 제어에는 손대지 않고, 표시/기록용 '실측 정합 지름'만 따로 추정한다.
#
#      제어용 D  : 기존 그대로 (정확도 유지)
#      표시용 D  : 아래 추정기 (실측 치수에 맞춤)
#
#  검증 (499회 교차검증, 실제 컵 치수 대비)
#      코드 추정 D     MAE 1.13cm  편향 +1.00
#      학습 추정 D     MAE 0.29cm  편향  0.00   (처음 보는 컵 0.49cm)
#
#  높이 H 는 학습해도 처음 보는 컵에서 오히려 나빠져(0.70 -> 1.00cm) 넣지 않았다.
#  H 는 코어의 기존 추정(MAE 0.70cm, 상대 6.2%)이 이미 충분하다.

SHAPE_MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'shape_model.json')

SHAPE_FEATURES = [
    'Initial_Stable_Freq_Hz',   # 초기 안정 주파수(Hz)
    'Theory_Slope',             # 이론 곡선 기울기
    'Theory_Intercept',         # 이론 곡선 절편
    'Reg_R2',                   # 회귀 적합도
    'Dynamic_Q',                # 동적 유량(ml/s)
    'Diameter_cm',              # 코어의 유효 지름 추정
    'D_Candidate_Median_cm',    # D 후보 중앙값
    'D_Candidate_MAD_cm',       # D 후보 산포
    'Init_Air_Locked_cm',       # 확정 초기 공기층
    'Noise_Floor_Level',        # 노이즈 플로어
    'Initial_Stable_Time_Sec',  # 초기 주파수 확정 시각
    'L_Init_cm',                # C/(4·init_f) — 음향이 직접 주는 길이
    'Inv_Slope',                # 1/|기울기| — D²에 비례하는 양
]
N_SHAPE = len(SHAPE_FEATURES)
SHAPE_RIDGE_ALPHA = 10.0
SHAPE_D_MIN_CM = 4.0        # 출력 클리핑 하한
SHAPE_D_MAX_CM = 12.0       # 출력 클리핑 상한


def build_shape_features(*, initial_stable_freq_hz: float, theory_slope: float,
                         theory_intercept: float, reg_r2: float, dynamic_q: float,
                         diameter_cm: float, d_candidate_median_cm: float,
                         d_candidate_mad_cm: float, init_air_cm: float,
                         noise_floor_level: float, initial_stable_time_sec: float,
                         sound_speed: float = 34300.0) -> Optional[np.ndarray]:
    """치수 추정용 특징 벡터. 확정(락) 이후에만 의미가 있다."""
    init_f = _f(initial_stable_freq_hz)
    slope = _f(theory_slope)
    if init_f <= 0:                                                # 초기 주파수 미확정
        return None

    l_init = sound_speed / (4.0 * init_f)
    inv_slope = 1.0 / abs(slope) if abs(slope) > 1e-12 else 0.0

    vec = np.array([
        init_f, slope, _f(theory_intercept), _f(reg_r2), _f(dynamic_q),
        _f(diameter_cm), _f(d_candidate_median_cm), _f(d_candidate_mad_cm),
        _f(init_air_cm), _f(noise_floor_level), _f(initial_stable_time_sec),
        l_init, inv_slope,
    ], dtype=float)
    return vec if np.all(np.isfinite(vec)) else None


class ShapeEstimator:
    """실제 용기 지름을 맞추는 Ridge 회귀. 온라인 학습은 하지 않는다(형상은 안 변한다)."""

    def __init__(self) -> None:
        self.mean = np.zeros(N_SHAPE)
        self.scale = np.ones(N_SHAPE)
        self.w = np.zeros(N_SHAPE + 1)
        self.n_samples = 0
        self.trained_at = ''

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[~np.isfinite(scale) | (scale <= 1e-9)] = 1.0
        self.scale = scale
        Z = np.hstack([(X - self.mean) / self.scale, np.ones((len(X), 1))])
        reg = np.full(N_SHAPE + 1, SHAPE_RIDGE_ALPHA)
        reg[-1] = 1e-6                                             # 편향항은 정규화 제외
        self.w = np.linalg.solve(Z.T @ Z + np.diag(reg), Z.T @ y)
        self.n_samples = len(X)

    def predict(self, x: Optional[np.ndarray]) -> Optional[float]:
        """실측 정합 지름(cm). 추정 불가면 None."""
        if x is None or self.n_samples <= 0:
            return None
        x = np.asarray(x, dtype=float)
        if x.shape != (N_SHAPE,) or not np.all(np.isfinite(x)):
            return None
        z = np.append((x - self.mean) / self.scale, 1.0)
        out = float(z @ self.w)
        if not math.isfinite(out):
            return None
        return float(np.clip(out, SHAPE_D_MIN_CM, SHAPE_D_MAX_CM))

    def to_dict(self) -> Dict[str, Any]:
        return {'version': 1, 'feature_names': SHAPE_FEATURES,
                'mean': self.mean.tolist(), 'scale': self.scale.tolist(),
                'w': self.w.tolist(), 'n_samples': self.n_samples,
                'trained_at': self.trained_at}

    def load_dict(self, data: Dict[str, Any]) -> bool:
        if list(data.get('feature_names', [])) != SHAPE_FEATURES:
            return False
        try:
            self.mean = np.array(data['mean'], dtype=float)
            self.scale = np.array(data['scale'], dtype=float)
            self.w = np.array(data['w'], dtype=float)
        except (KeyError, TypeError, ValueError):
            return False
        if self.w.shape != (N_SHAPE + 1,):
            return False
        self.n_samples = int(data.get('n_samples', 0))
        self.trained_at = str(data.get('trained_at', ''))
        return True

    def save(self, path: str = SHAPE_MODEL_FILE) -> bool:
        try:
            tmp = path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False)
            os.replace(tmp, path)
            return True
        except OSError:
            return False

    def load(self, path: str = SHAPE_MODEL_FILE) -> bool:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return self.load_dict(json.load(f))
        except (OSError, ValueError):
            return False


_shape = ShapeEstimator()
_shape_loaded = _shape.load()


def get_shape_model() -> ShapeEstimator:
    return _shape


def estimate_true_diameter(core: Any) -> Optional[float]:
    """
    코어 상태에서 실측 정합 지름(cm)을 구한다. 표시/기록 전용이며
    제어에는 쓰지 않는다. 추정 불가 시 None.
    """
    if _shape.n_samples <= 0:
        return None
    try:
        x = build_shape_features(
            initial_stable_freq_hz=getattr(core, 'initial_stable_freq', 0.0),
            theory_slope=getattr(core, 'theory_slope', 0.0),
            theory_intercept=getattr(core, 'theory_intercept', 0.0),
            reg_r2=getattr(core, 'last_r2', 0.0),
            dynamic_q=getattr(core, 'dynamic_Q', 0.0),
            diameter_cm=getattr(core, 'current_diameter', 0.0),
            d_candidate_median_cm=getattr(core, 'last_d_candidate_median', 0.0),
            d_candidate_mad_cm=getattr(core, 'last_d_candidate_mad', 0.0),
            init_air_cm=getattr(core, 'init_air_len_cm', 0.0),
            noise_floor_level=getattr(core, 'noise_floor_level', 0.0),
            initial_stable_time_sec=getattr(core, 'initial_stable_time', 0.0),
            sound_speed=getattr(core, 'SOUND_SPEED', 34300.0),
        )
        return _shape.predict(x)
    except Exception:                                              # noqa: BLE001
        return None
