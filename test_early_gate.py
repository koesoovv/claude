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
