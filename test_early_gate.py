#!/usr/bin/env python3
# test_early_gate.py ── 초반 안정성 게이트 자체 점검 [v5.34-EARLY-GATE]
#
# 라즈베리파이 하드웨어(sounddevice/gpiozero) 없이 PC에서 실행할 수 있다.
# 부족한 모듈은 아래에서 더미로 채운 뒤 smart_water_core 를 불러온다.
#
#   python test_early_gate.py

from __future__ import annotations

import sys
import types


# ── 하드웨어 의존 모듈 더미 주입 ────────────────────────────────────
def _install_stubs() -> None:
    if 'sounddevice' not in sys.modules:
        sd = types.ModuleType('sounddevice')

        class _Stream:                                             # 오디오 스트림 흉내
            def __init__(self, *a, **k): pass
            def start(self): pass
            def stop(self): pass
            def close(self): pass

        sd.InputStream = _Stream
        sd.query_devices = lambda *a, **k: []
        sys.modules['sounddevice'] = sd

    if 'gpiozero' not in sys.modules:
        gz = types.ModuleType('gpiozero')

        class _Out:                                                # 솔레노이드 흉내
            def __init__(self, *a, **k): self.value = 0
            def on(self): pass
            def off(self): pass

        class _Btn:                                                # 스위치 흉내
            def __init__(self, *a, **k): self.when_pressed = None

        gz.OutputDevice = _Out
        gz.Button = _Btn
        sys.modules['gpiozero'] = gz


_install_stubs()

try:
    import smart_water_core as core
except Exception as exc:                                           # noqa: BLE001
    print(f'[오류] smart_water_core 를 불러오지 못했습니다: {exc}')
    sys.exit(1)


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


print('=' * 72)
print('  초반 안정성 게이트 자체 점검  [v5.34-EARLY-GATE]')
print('=' * 72)

# ── 1. 물리 밴드 ────────────────────────────────────────────────
print('\n[1] 초기 주파수 물리 밴드')
lo, hi = core.INIT_F_PHYS_MIN_HZ, core.INIT_F_PHYS_MAX_HZ
print(f'       밴드 = {lo:.0f} ~ {hi:.0f} Hz')
check('밴드가 H·D 허용 범위에서 유도된다',
      abs(lo - core.SOUND_SPEED / (4 * (core.INIT_AIR_MAX_CM
                                        + core.END_CORRECTION_K * core.DIAM_MAX_CM))) < 1e-6)
check('하한이 상한보다 작다', lo < hi, f'{lo:.0f} / {hi:.0f}')
check('실측 정상 범위(750~1100Hz)를 모두 통과시킨다',
      lo <= 750 and 1100 <= hi, f'{lo:.0f}~{hi:.0f}')
# 499회에서 모델 확정에 실패한 6건의 초기 주파수
for f_bad in (398.0, 1409.0, 1651.0, 1707.0):
    check(f'오검출 {f_bad:.0f}Hz 를 배제한다', not (lo <= f_bad <= hi))
# 물리 밴드만으로는 못 거르는 2건 — 지터 게이트가 담당한다
for f_edge in (538.0, 543.0):
    if lo <= f_edge <= hi:
        print(f'       (참고) {f_edge:.0f}Hz 는 밴드를 통과 → 지터 게이트가 담당')

# ── 2. 상승 스파이크 게이트 ─────────────────────────────────────
print('\n[2] 상승 방향 스파이크 게이트')
check('상승 게이트가 정의돼 있다', hasattr(core, 'REG_SPIKE_GATE_UP'))
check('상승 게이트가 하강 게이트보다 느슨하다',
      core.REG_SPIKE_GATE_UP > core.REG_SPIKE_GATE,
      f'up={core.REG_SPIKE_GATE_UP} down={core.REG_SPIKE_GATE}')


def _would_block(recent_median_inv: float, new_freq: float) -> bool:
    """코어의 스파이크 판정과 같은 식."""
    inv = 1.0 / new_freq
    dev = (inv - recent_median_inv) / recent_median_inv
    return dev > core.REG_SPIKE_GATE_EARLY or -dev > core.REG_SPIKE_GATE_UP


med = 1.0 / 1000.0                                                 # 최근 중앙값 1000Hz
check('정상적인 완만한 상승(1000→1050Hz)은 통과', not _would_block(med, 1050.0))
check('배음 튐(1000→2000Hz)은 차단', _would_block(med, 2000.0))
check('급락(1000→800Hz)은 차단', _would_block(med, 800.0))

# ── 3. 미확정 시간 상한 ─────────────────────────────────────────
print('\n[3] D/H 미확정 상태 시간 상한')
caps = {t: core._unlocked_time_cap(t, False) for t in (50, 60, 70, 80, 90)}
for t, c in caps.items():
    print(f'       목표 {t}% → {c:.1f}초 (지터경고 시 {core._unlocked_time_cap(t, True):.1f}초)')
check('목표가 낮을수록 상한이 짧다', caps[50] < caps[70] <= caps[90],
      str({k: round(v, 1) for k, v in caps.items()}))
check('절대 상한을 넘지 않는다', max(caps.values()) <= core.SAFETY_UNLOCKED_MAX_SEC)
check('절대 하한 밑으로 내려가지 않는다',
      core._unlocked_time_cap(0, True) >= core.SAFETY_UNLOCKED_MIN_SEC)
check('지터 경고 시 상한이 더 짧아진다',
      core._unlocked_time_cap(50, True) < core._unlocked_time_cap(50, False))
# 정상 시행의 최대 락 시각(목표50 기준 15.5초)보다는 여유가 있어야 한다
check('목표 50%% 상한이 실측 최대 락 시각(15.5초)보다 크다', caps[50] > 15.5,
      f'{caps[50]:.1f}초')
check('이상한 목표값(None)에도 죽지 않는다',
      core.SAFETY_UNLOCKED_MIN_SEC <= core._unlocked_time_cap(None, False)
      <= core.SAFETY_UNLOCKED_MAX_SEC)

# ── 3-2. 초기 주파수 확정 검증 [v5.35-INITF-RETRY] ──────────────
print('\n[3-2] 초기 주파수 확정 검증 / 재측정')
import random
random.seed(0)

def _buf(center, jitter_ratio, n=10):
    """평균 center, 상대 산포 jitter_ratio 인 표본 n개."""
    return [center * (1 + jitter_ratio * (1 if i % 2 else -1)) for i in range(n)]

# 실기에서 실제로 발생한 실패: 초반 튐이 396Hz 로 확정됨
ok, why, sp = core._validate_init_freq(396.0, _buf(396.0, 0.02))
check('초반 튐 396Hz 를 기각한다', not ok, f'사유={why}')
check('기각 사유가 물리범위임', '물리범위' in why, why)

# 정상 용기 (250mL 비커 계열 ~560Hz, 머그 ~1030Hz)
for f_ok in (560.0, 750.0, 1030.0, 1150.0):
    ok, why, sp = core._validate_init_freq(f_ok, _buf(f_ok, 0.02))
    check(f'정상 {f_ok:.0f}Hz 안정 신호는 채택', ok, f'사유={why}')

# 밴드 안이어도 요동치면 기각 (물이 바닥 때리는 구간)
ok, why, sp = core._validate_init_freq(900.0, _buf(900.0, 0.30))
check('밴드 안이지만 요동치면 기각', not ok, f'사유={why} 산포={sp:.2f}')
check('기각 사유가 요동임', '요동' in why, why)

# 배음 오검출
ok, why, sp = core._validate_init_freq(1800.0, _buf(1800.0, 0.02))
check('배음 1800Hz 를 기각한다', not ok, f'사유={why}')

# 방어
check('빈 표본이면 기각', not core._validate_init_freq(900.0, [])[0])
check('0Hz 면 기각', not core._validate_init_freq(0.0, _buf(900.0, 0.02))[0])
check('폴백 시각이 락 예상 시각보다 늦다', core.INIT_F_FALLBACK_SEC >= 10.0,
      f'{core.INIT_F_FALLBACK_SEC}초')
check('재측정 상태 초기값', core.init_f_retry_count == 0 and core.init_f_best_candidate == 0.0)

# ── 3-3. 순간 튐 / 배음 배제 [v5.37-TRANSIENT] ──────────────────
print('\n[3-3] 순간 튐 / 배음 배제')
_around = [840.0, 845.0, 850.0, 855.0, 850.0]        # 주변 ≈ 850Hz

# 실기 그래프에서 관측된 튐
drop, why, ratio = core._classify_transient(1800.0, _around)
check('실기 튐 1800Hz 를 버린다', drop, f'사유={why} 배율={ratio:.2f}')
check('배음으로 판정한다', '배음' in why, why)

for f_h, lab in ((1700.0, '2배음'), (2550.0, '3배음'), (425.0, '1/2 부저음')):
    drop, why, _ = core._classify_transient(f_h, _around)
    check(f'{lab} {f_h:.0f}Hz 를 버린다', drop, f'사유={why}')

# 정상적인 상승은 통과해야 한다 (정상 시행 95분위가 1.129배)
for f_ok, lab in ((880.0, '+4%'), (900.0, '+6%'), (950.0, '+12%')):
    drop, why, _ = core._classify_transient(f_ok, _around)
    check(f'정상 상승 {lab} 는 통과', not drop, f'사유={why}')

drop, why, _ = core._classify_transient(1200.0, _around)
check('+41% 급등은 버린다', drop, f'사유={why}')
drop, why, _ = core._classify_transient(500.0, _around)
check('급락도 버린다', drop, f'사유={why}')

# 방어
check('표본이 모자라면 판정 보류', not core._classify_transient(1800.0, [840, 845])[0])
check('0Hz 는 판정 보류', not core._classify_transient(0.0, _around)[0])
check('빈 이력이면 판정 보류', not core._classify_transient(1800.0, [])[0])
check('임계가 정상 99분위(1.238)보다 크다', core.TRANSIENT_RATIO_UP > 1.238,
      f'{core.TRANSIENT_RATIO_UP}')
check('홀드오프가 있어 실제 상승을 영구 차단하지 않는다',
      core.TRANSIENT_HOLDOFF_N >= 2, f'{core.TRANSIENT_HOLDOFF_N}')
check('TRANSIENT_ENABLE 스위치 존재', isinstance(core.TRANSIENT_ENABLE, bool))

# ── 3-4. 이상치 절삭 재시도 [v5.38-TRIM-RETRY] ─────────────────
print('\n[3-4] 회귀 실패 시 이상치 절삭 재시도')
import numpy as _np
check('절삭 재시도 함수 존재', hasattr(core, 'robust_slope_with_trim'))
check('TRIM_RETRY_ENABLE 스위치 존재', isinstance(core.TRIM_RETRY_ENABLE, bool))
check('제거 상한이 있다(억지 적합 방지)', 0 < core.TRIM_RETRY_MAX_FRAC <= 0.3,
      f'{core.TRIM_RETRY_MAX_FRAC}')

# r² 임계 판정이 워커와 일치하는지
check('r² 임계: 완만한 기울기 0.88', abs(core._r2_threshold(-0.00002) - 0.88) < 1e-9)
check('r² 임계: 중간 기울기 0.92', abs(core._r2_threshold(-0.00004) - 0.92) < 1e-9)
check('r² 임계: 가파른 기울기 0.95', abs(core._r2_threshold(-0.00008) - 0.95) < 1e-9)

# 깨끗한 직선 — 절삭 없이 통과해야 한다
_t = _np.linspace(6.0, 20.0, 40)
_slope = -2.5e-5
_clean = 0.0012 + _slope * _t
s1, b1, r1, k1 = core.robust_slope_with_trim(_t, _clean)
check('깨끗한 직선은 절삭 없이 통과', s1 is not None and k1 == 0, f'제거 {k1}개')

# 실기에서 실제로 락이 16.7초까지 밀렸던 Trial_0178 의 6~13초 구간이다.
# 합성 데이터로는 기존 LOESS+75분위 절삭이 웬만한 오염을 이미 견디므로,
# 실제로 실패했던 데이터를 그대로 쓴다.
_T178_T = [6.031, 6.235, 6.437, 6.640, 6.841, 7.041, 7.242, 7.442, 7.644, 7.845,
           8.045, 8.245, 8.445, 8.646, 8.846, 9.046, 9.246, 9.446, 9.646, 9.846,
           10.046, 10.247, 10.447, 10.647, 10.850, 11.050, 11.250, 11.450,
           11.651, 11.851, 12.052, 12.253, 12.454, 12.654, 12.854]
_T178_F = [839.8, 829.0, 882.9, 1184.3, 872.1, 882.9, 2097.3, 904.4, 882.9, 1531.0,
           1569.8, 1152.0, 960.4, 971.1, 952.8, 956.1, 967.9, 960.4, 971.1, 979.8,
           966.8, 963.6, 981.9, 990.5, 1016.4, 990.5, 1022.8, 1025.0, 1043.3,
           1048.7, 1093.9, 1070.2, 1076.7, 1098.2, 1096.0]
_inv = [1.0 / v for v in _T178_F]

_s0, _b0, _r0 = core.robust_slope_loess_lsm(_T178_T, _inv)
check('실기 실패 데이터가 기존 방식으로는 r² 미달',
      _s0 is not None and _r0 <= core._r2_threshold(_s0), f'r²={_r0:.3f}')
s2, b2, r2, k2 = core.robust_slope_with_trim(_T178_T, _inv)
check('절삭 재시도로 통과한다', s2 is not None and k2 > 0 and r2 > core._r2_threshold(s2),
      f'제거 {k2}개 r²={r2:.3f}')
check('절삭이 상한 이내', 0 < k2 <= int(len(_T178_T) * core.TRIM_RETRY_MAX_FRAC),
      f'제거 {k2}개')
check('절삭 후 기울기가 물리적으로 타당', s2 is not None and s2 < 0
      and abs(s2) > core.MIN_SLOPE_CUTOFF, f'{s2:.3e}')

# 상한을 넘는 오염은 억지로 통과시키지 않아야 한다
_hopeless = _clean + _np.random.default_rng(0).normal(0, abs(_slope) * 25, len(_t))
s4, b4, r4, k4 = core.robust_slope_with_trim(_t, _hopeless)
check('가망 없는 데이터는 억지 통과시키지 않는다', k4 in (-1, 0), f'제거 {k4}개')

# 표본이 모자라면 그대로 반환
s3, b3, r3, k3 = core.robust_slope_with_trim(_t[:8], _clean[:8])
check('표본 부족 시 죽지 않는다', k3 in (0, -1))
check('절삭 통계 초기값', core.trim_retry_count == 0 and core.trim_removed_total == 0)

# ── 4. 지터 설정 ────────────────────────────────────────────────
print('\n[4] 지터 측정 설정')
check('측정 창이 락 시각(≈9.5초)보다 먼저 끝난다', core.JITTER_WIN_END <= 9.0,
      f'{core.JITTER_WIN_END}초')
check('측정 창이 유효하다', core.JITTER_WIN_START < core.JITTER_WIN_END)
check('경고 임계가 정상 99분위(194Hz)보다 낮다', core.JITTER_ALERT_HZ < 194,
      f'{core.JITTER_ALERT_HZ}Hz')
check('경고 임계가 정상 90분위(69Hz)보다 높다', core.JITTER_ALERT_HZ > 69,
      f'{core.JITTER_ALERT_HZ}Hz')

# ── 5. 초기 상태 ────────────────────────────────────────────────
print('\n[5] 초기 상태와 스위치')
for attr, want in [('early_jitter_hz', 0.0), ('jitter_alert', False),
                   ('jitter_locked_in', False), ('init_f_rejected', 0),
                   ('spike_up_blocked_count', 0)]:
    check(f'{attr} 초기값', getattr(core, attr) == want, f'실제 {getattr(core, attr)}')
check('EARLY_GATE_ENABLE 스위치 존재', isinstance(core.EARLY_GATE_ENABLE, bool))
check('JITTER_ENABLE 스위치 존재', isinstance(core.JITTER_ENABLE, bool))

print('\n' + '=' * 72)
print(f'  통과 {_passed} / 실패 {_failed}')
print('=' * 72)
sys.exit(1 if _failed else 0)
