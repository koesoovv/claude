#!/usr/bin/env python3
# train_fill_corrector.py ── 급수 보정 모델 오프라인 사전학습
#
# 누적된 trial_summary.csv 를 읽어 Ridge 회귀 가중치를 만들고
# fill_corrector_model.json 으로 저장한다.
#
#   사용법:
#       python train_fill_corrector.py                       # 기본 경로 자동 탐색
#       python train_fill_corrector.py data/trial_summary.csv
#       python train_fill_corrector.py data/trial_summary.csv -o my_model.json
#
# 학습 이후에는 급수할 때마다 fill_corrector.observe()가 온라인으로 가중치를
# 갱신하므로, 이 스크립트는 최초 1회 또는 대규모 재학습 때만 실행하면 된다.

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import fill_corrector as fc


# 실험 기록기가 쓰는 기본 경로 후보들
DEFAULT_PATHS = [
    'trial_summary.csv',
    os.path.join('data', 'trial_summary.csv'),
    os.path.join('experiment_data', 'trial_summary.csv'),
]


def _to_float(value: Any) -> Optional[float]:
    """CSV 문자열을 float으로. 빈칸/비정상은 None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if np.isfinite(out) else None


def _is_true(value: Any) -> bool:
    return str(value).strip().upper() in ('TRUE', '1', 'Y', 'YES')


def _open_summary(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    trial_summary.csv 를 읽는다.

    기록기가 한글 설명 헤더를 첫 줄에 덧붙이는 경우가 있어,
    'Trial_ID'가 들어 있는 줄을 실제 헤더로 판단한다.
    인코딩도 utf-8-sig / cp949 순으로 시도한다.
    """
    last_err: Optional[Exception] = None
    for encoding in ('utf-8-sig', 'cp949', 'latin-1'):
        try:
            with open(path, 'r', newline='', encoding=encoding) as f:
                rows = list(csv.reader(f))
            break
        except (UnicodeDecodeError, LookupError) as exc:
            last_err = exc
    else:
        raise RuntimeError(f'CSV 인코딩을 판별하지 못했습니다: {last_err}')

    header_idx = None
    for i, row in enumerate(rows[:5]):                             # 앞부분만 훑어본다
        if 'Trial_ID' in row or 'Target_Fill_Pct' in row:
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError('헤더 행(Trial_ID)을 찾지 못했습니다.')

    header = rows[header_idx]
    records = [dict(zip(header, r)) for r in rows[header_idx + 1:] if any(c.strip() for c in r)]
    return records, header


def load_dataset(path: str, *, verbose: bool = True):
    """
    요약 CSV → (특징행렬 X, 목표값 y, 메타정보)

    특징 생성은 fill_corrector.build_features()를 그대로 호출한다.
    학습과 실제 추론이 반드시 동일한 코드를 지나가게 하기 위함이다.
    """
    records, _ = _open_summary(path)
    X: List[np.ndarray] = []
    y: List[float] = []
    meta: List[Dict[str, Any]] = []
    skipped = {'reject': 0, 'no_actual': 0, 'no_features': 0}

    for rec in records:
        if str(rec.get('Reject_Reason', '')).strip() not in ('ok', ''):
            skipped['reject'] += 1                                 # 모델이 성립 안 한 시행 제외
            continue

        fill_error = _to_float(rec.get('Fill_Error_Pct'))
        if fill_error is None:                                     # 실측 미입력 시행 제외
            actual = _to_float(rec.get('Actual_Fill_Pct'))
            target = _to_float(rec.get('Target_Fill_Pct'))
            if actual is None or target is None:
                skipped['no_actual'] += 1
                continue
            fill_error = actual - target

        vec = fc.build_features(
            decision_fill_pct=_to_float(rec.get('Decision_Fill_Pct')) or 0.0,
            current_air_cm=_to_float(rec.get('Estimated_Stop_Air_cm')) or 0.0,
            stop_freq_hz=_to_float(rec.get('Stop_Freq_Hz')) or 0.0,
            elapsed_sec=_to_float(rec.get('Stop_Elapsed_Sec')) or 0.0,
            init_air_cm=_to_float(rec.get('Init_Air_Locked_cm')) or 0.0,
            diameter_cm=_to_float(rec.get('Diameter_cm')) or 0.0,
            reg_r2=_to_float(rec.get('Reg_R2')) or 0.0,
            target_fill_pct=_to_float(rec.get('Target_Fill_Pct')) or 0.0,
            dynamic_q=_to_float(rec.get('Dynamic_Q')) or 0.0,
            bridge_used=_is_true(rec.get('Bridge_Used')),
            initial_stable_freq_hz=_to_float(rec.get('Initial_Stable_Freq_Hz')) or 0.0,
            max_freq_seen_hz=_to_float(rec.get('Max_Freq_Seen_Hz')) or 0.0,
            noise_floor_level=_to_float(rec.get('Noise_Floor_Level')) or 0.0,
            d_candidate_mad_cm=_to_float(rec.get('D_Candidate_MAD_cm')) or 0.0,
            flow_rate_ml_s=_to_float(rec.get('Flow_Rate_ml_s')) or 0.0,
        )
        if vec is None:
            skipped['no_features'] += 1
            continue

        X.append(vec)
        y.append(fill_error)
        meta.append({
            'Trial_ID': rec.get('Trial_ID', ''),
            'Cup_Name': rec.get('Cup_Name', ''),
            'Started_At': rec.get('Started_At', ''),
            'Target_Fill_Pct': _to_float(rec.get('Target_Fill_Pct')) or 0.0,
        })

    if verbose:
        print(f'  전체 {len(records)}행 → 학습 가능 {len(X)}행 '
              f'(제외: 모델무효 {skipped["reject"]}, 실측없음 {skipped["no_actual"]}, '
              f'특징불가 {skipped["no_features"]})')
    if not X:
        raise RuntimeError('학습에 쓸 수 있는 행이 없습니다.')
    return np.vstack(X), np.array(y, dtype=float), meta


def load_shape_dataset(path: str, *, verbose: bool = True):
    """
    요약 CSV → 치수 추정용 (X, 실제지름, 메타)

    cup_info.csv 에 등록된 실제 상단/하단 지름이 있는 행만 쓴다.
    실제 치수를 모르는 행은 학습에 쓸 수 없다.
    """
    records, _ = _open_summary(path)
    X, y, meta = [], [], []
    skipped = 0
    for rec in records:
        if str(rec.get('Reject_Reason', '')).strip() not in ('ok', ''):
            skipped += 1
            continue
        top = _to_float(rec.get('Cup_Top_Diameter_cm'))
        bot = _to_float(rec.get('Cup_Bottom_Diameter_cm'))
        if top is None or bot is None or top <= 0 or bot <= 0:
            skipped += 1                                           # 실제 치수 미등록
            continue

        vec = fc.build_shape_features(
            initial_stable_freq_hz=_to_float(rec.get('Initial_Stable_Freq_Hz')) or 0.0,
            theory_slope=_to_float(rec.get('Theory_Slope')) or 0.0,
            theory_intercept=_to_float(rec.get('Theory_Intercept')) or 0.0,
            reg_r2=_to_float(rec.get('Reg_R2')) or 0.0,
            dynamic_q=_to_float(rec.get('Dynamic_Q')) or 0.0,
            diameter_cm=_to_float(rec.get('Diameter_cm')) or 0.0,
            d_candidate_median_cm=_to_float(rec.get('D_Candidate_Median_cm')) or 0.0,
            d_candidate_mad_cm=_to_float(rec.get('D_Candidate_MAD_cm')) or 0.0,
            init_air_cm=_to_float(rec.get('Init_Air_Locked_cm')) or 0.0,
            noise_floor_level=_to_float(rec.get('Noise_Floor_Level')) or 0.0,
            initial_stable_time_sec=_to_float(rec.get('Initial_Stable_Time_Sec')) or 0.0,
        )
        if vec is None:
            skipped += 1
            continue
        X.append(vec)
        y.append((top + bot) / 2.0)                                # 실제 평균 지름
        meta.append({'Trial_ID': rec.get('Trial_ID', ''),
                     'Cup_Name': rec.get('Cup_Name', ''),
                     'Diameter_cm': _to_float(rec.get('Diameter_cm')) or 0.0})

    if verbose:
        print(f'  치수 학습 가능 {len(X)}행 (제외 {skipped})')
    if not X:
        raise RuntimeError('치수 학습에 쓸 수 있는 행이 없습니다. cup_info.csv 에 실제 치수를 등록하세요.')
    return np.vstack(X), np.array(y, dtype=float), meta


def _find_default_path() -> Optional[str]:
    for p in DEFAULT_PATHS:
        if os.path.exists(p):
            return p
    return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='급수 보정 모델 사전학습')
    ap.add_argument('summary', nargs='?', default=None, help='trial_summary.csv 경로')
    ap.add_argument('-o', '--out', default=fc.MODEL_FILE, help='모델 저장 경로')
    args = ap.parse_args(argv)

    path = args.summary or _find_default_path()
    if path is None:
        print('[오류] trial_summary.csv 를 찾지 못했습니다. 경로를 직접 지정하세요.')
        return 1
    if not os.path.exists(path):
        print(f'[오류] 파일이 없습니다: {path}')
        return 1

    print(f'[학습] 데이터: {path}')
    X, y, meta = load_dataset(path)

    model = fc.FillCorrector()
    model.fit_batch(X, y)
    model.trained_at = _dt.datetime.now().isoformat(timespec='seconds')
    if not model.save(args.out):
        print(f'[오류] 모델 저장 실패: {args.out}')
        return 1

    # 학습 데이터 자체에 대한 적합도 (일반화 성능은 eval_fill_corrector.py로 확인)
    pred = np.array([model.predict(x)[0] for x in X])
    resid = y - pred
    print(f'[학습] 완료 → {args.out}')
    print(f'  표본 수      : {len(X)}')
    print(f'  현재 오차    : MAE {np.abs(y).mean():.2f}%p  RMSE {np.sqrt((y ** 2).mean()):.2f}%p  '
          f'편향 {y.mean():+.2f}%p')
    print(f'  보정 후(내적합): MAE {np.abs(resid).mean():.2f}%p  RMSE {np.sqrt((resid ** 2).mean()):.2f}%p  '
          f'편향 {resid.mean():+.2f}%p')
    print('  ※ 위는 학습 데이터 기준입니다. 일반화 성능은 eval_fill_corrector.py 를 실행하세요.')

    cups = sorted({m['Cup_Name'] for m in meta if m['Cup_Name']})
    if cups:
        print(f'  학습에 포함된 용기: {", ".join(cups)}')

    # ── [v5.36-SHAPE] 실측 정합 지름 추정기도 함께 학습한다 ──────────
    # 제어에는 쓰지 않고 표시/기록 전용이므로, 실패해도 경고만 남긴다.
    try:
        Xs, ys, ms = load_shape_dataset(path)
        shape = fc.ShapeEstimator()
        shape.fit(Xs, ys)
        shape.trained_at = _dt.datetime.now().isoformat(timespec='seconds')
        if shape.save():
            cur = np.array([m['Diameter_cm'] for m in ms])
            pred = np.array([shape.predict(x) or 0.0 for x in Xs])
            print(f'[치수] 완료 → {fc.SHAPE_MODEL_FILE}')
            print(f'  코드 추정 지름 : MAE {np.abs(cur - ys).mean():.2f} cm '
                  f'(편향 {(cur - ys).mean():+.2f})')
            print(f'  학습 추정 지름 : MAE {np.abs(pred - ys).mean():.2f} cm '
                  f'(편향 {(pred - ys).mean():+.2f})  ※ 내적합')
        else:
            print('[치수] 저장 실패 — 표시용 지름 추정은 비활성됩니다.')
    except Exception as exc:                                       # noqa: BLE001
        print(f'[치수] 건너뜀: {exc}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
