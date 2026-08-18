#!/usr/bin/env python3
# test_fill_corrector.py ── 보정 모듈 자체 점검
#
# 라즈베리파이 하드웨어(sounddevice/gpiozero) 없이 실행할 수 있다.
# smart_water_core 에 넣은 통합 코드가 기대대로 동작하는지, 그리고
# 안전장치가 실제로 걸리는지 확인한다.
#
#   python test_fill_corrector.py

from __future__ import annotations

import sys

import numpy as np

import fill_corrector as fc


_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = '') -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f'  [OK]   {name}')
    else:
        _failed += 1
        print(f'  [FAIL] {name} {detail}')


def sample_features(**over) -> np.ndarray:
    """
    실제 시행(Trial_0001, plastic_mug, 목표 50%)의 정지 시점 값을 그대로 쓴다.

    주파수·공기층·경과시간이 서로 물리적으로 맞아떨어지는 점을 써야 한다.
    임의로 지어낸 조합은 학습 분포 밖이라 분포 이탈 보호가 걸리면서
    보정량이 클리핑 한계에 붙어버려 시험 의미가 사라진다.
    """
    kw = dict(
        decision_fill_pct=55.128, current_air_cm=4.257, stop_freq_hz=1167.451,
        elapsed_sec=9.456, init_air_cm=9.488, diameter_cm=8.226, reg_r2=0.951,
        target_fill_pct=50.0, dynamic_q=15.0, bridge_used=False,
        initial_stable_freq_hz=1031.925, max_freq_seen_hz=1167.451,
        noise_floor_level=0.163, d_candidate_mad_cm=0.272, flow_rate_ml_s=15.0,
    )
    kw.update(over)
    return fc.build_features(**kw)


print('=' * 70)
print('  fill_corrector 자체 점검')
print('=' * 70)

# ── 1. 특징 생성 ────────────────────────────────────────────────
print('\n[1] 특징 생성')
x = sample_features()
check('정상 입력이면 특징 벡터를 만든다', x is not None)
check(f'특징 개수가 {fc.N_FEATURES}개', x is not None and x.shape == (fc.N_FEATURES,),
      f'실제 {None if x is None else x.shape}')
check('H 미확정(init_air=0)이면 None', sample_features(init_air_cm=0.0) is None)
check('D 미확정(diameter=0)이면 None', sample_features(diameter_cm=0.0) is None)
check('NaN 입력이면 None 또는 유한값',
      sample_features(stop_freq_hz=float('nan')) is None
      or np.all(np.isfinite(sample_features(stop_freq_hz=float('nan')))))

# ── 2. 콜드 스타트 안전장치 ──────────────────────────────────────
print('\n[2] 콜드 스타트 안전장치')
m = fc.FillCorrector()
val, info = m.predict(x)
check('학습 전에는 보정 0', val == 0.0, f'값={val}')
check('사유가 insufficient_data', info['reason'] == 'insufficient_data', f"사유={info['reason']}")
check('특징이 None이면 보정 0', m.predict(None)[0] == 0.0)

# ── 3. 학습과 예측 ──────────────────────────────────────────────
print('\n[3] 학습과 예측')
rng = np.random.default_rng(0)
X = np.vstack([sample_features(
    decision_fill_pct=float(t + rng.normal(0, 1)),
    target_fill_pct=float(t),
    elapsed_sec=float(8 + t * 0.1),
    current_air_cm=float(9.4 * (1 - t / 100.0)),
) for t in rng.uniform(50, 90, 200)])
# 인위적 편향: 실제로는 항상 2%p 덜 채워지는 장비라고 가정
y = np.full(len(X), -2.0) + rng.normal(0, 0.5, len(X))
m.fit_batch(X, y)
check('학습 후 표본 수가 기록된다', m.n_samples == len(X), f'{m.n_samples}')
val, info = m.predict(X[0])
check('학습 후 보정이 활성화된다', info['active'] is True, f"사유={info['reason']}")
check('일관된 -2%p 편향을 학습한다', -3.0 < val < -1.0, f'예측={val:.2f}')

# ── 4. 안전 클리핑 ──────────────────────────────────────────────
print('\n[4] 안전 클리핑 / 분포 이탈 보호')
extreme = sample_features(decision_fill_pct=5000.0, stop_freq_hz=90000.0,
                          elapsed_sec=5000.0, current_air_cm=900.0)
val, info = m.predict(extreme)
check(f'보정량이 ±{fc.MAX_CORRECTION_PCT:g}%p 를 넘지 않는다',
      abs(val) <= fc.MAX_CORRECTION_PCT + 1e-9, f'값={val:.2f}')
check('분포 이탈을 감지해 축소한다', info.get('shrink', 1.0) < 1.0,
      f"축소율={info.get('shrink')}")

# 가중치를 일부러 망가뜨려도 제어가 죽지 않아야 한다
broken = fc.FillCorrector()
broken.fit_batch(X, y)
broken.w = np.full(broken.dim, np.nan)
check('가중치가 NaN이면 보정 0', broken.predict(X[0])[0] == 0.0)

# ── 5. 온라인 학습 ──────────────────────────────────────────────
print('\n[5] 온라인 적응 학습')
m2 = fc.FillCorrector()
m2.fit_batch(X, y)                                                 # -2%p 편향으로 학습된 상태
before = m2.predict(X[0])[0]
# 장비 특성이 바뀌어 이제는 +3%p 넘치기 시작했다고 가정
for i in range(80):
    m2.update(X[i % len(X)], 3.0)
after = m2.predict(X[0])[0]
check('바뀐 편향 방향으로 따라간다', after > before, f'{before:.2f} → {after:.2f}')
check('온라인 표본 수가 증가한다', m2.n_online == 80, f'{m2.n_online}')
n_before = m2.n_samples
m2.update(X[0], 999.0)                                             # 실측 입력 오타 상황
check('말도 안 되는 실측값은 학습에서 배제', m2.n_samples == n_before)
m2.update(np.array([1.0, 2.0]), 1.0)                               # 차원이 틀린 입력
check('차원이 틀린 특징은 무시', m2.n_samples == n_before)

# ── 6. 저장과 복원 ──────────────────────────────────────────────
print('\n[6] 저장 / 복원')
import tempfile, os
tmp = os.path.join(tempfile.gettempdir(), 'fc_test_model.json')
check('저장 성공', m2.save(tmp))
m3 = fc.FillCorrector()
check('복원 성공', m3.load(tmp))
check('복원 후 예측이 동일', abs(m3.predict(X[0])[0] - m2.predict(X[0])[0]) < 1e-9)
check('복원 후 표본 수 유지', m3.n_samples == m2.n_samples)
# 특징 구성이 바뀐 옛 모델은 거부해야 한다
bad = m2.to_dict(); bad['feature_names'] = ['old_feature']
check('특징 구성이 다르면 로드 거부', fc.FillCorrector().load_dict(bad) is False)
os.remove(tmp)

# ── 7. 정지 임계값 반영 방향 ─────────────────────────────────────
print('\n[7] 정지 임계값 이동 방향 (코어 통합 로직과 동일한 식)')
TARGET = 70.0


def would_stop(fill_pct: float, ai_adv: float) -> bool:
    """smart_water_core 의 trigger 계산과 같은 식."""
    total_advance = 0.0 + 0.0 + 0.0 + ai_adv
    return fill_pct >= TARGET - total_advance


check('보정 0이면 기존 동작과 동일', would_stop(70.0, 0.0) and not would_stop(69.9, 0.0))
check('덜 채울 것으로 예측(-2)하면 더 늦게 멈춘다',
      not would_stop(70.0, -2.0) and would_stop(72.0, -2.0))
check('넘칠 것으로 예측(+2)하면 더 일찍 멈춘다', would_stop(68.0, +2.0))

# ── 8. 실기 성능 (모델 파일이 있으면) ─────────────────────────────
print('\n[8] 배포 모델 파일')
live = fc.FillCorrector()
if live.load():
    check('fill_corrector_model.json 로드 성공', True)
    check(f'학습 표본 {live.n_samples}회 (>= {fc.MIN_SAMPLES})', live.n_samples >= fc.MIN_SAMPLES)
    v, i = live.predict(x)
    check('전형적 입력에 대해 보정이 활성', i['active'] is True, f"사유={i['reason']}")
    check(f'보정량이 상식적 범위 (실제 {v:+.2f}%p)', abs(v) <= fc.MAX_CORRECTION_PCT)
else:
    print('  [건너뜀] fill_corrector_model.json 이 없습니다. '
          'train_fill_corrector.py 를 먼저 실행하세요.')

print('\n' + '=' * 70)
print(f'  통과 {_passed} / 실패 {_failed}')
print('=' * 70)
sys.exit(1 if _failed else 0)
