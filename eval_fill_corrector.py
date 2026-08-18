#!/usr/bin/env python3
# eval_fill_corrector.py ── 급수 보정 모델 성능 검증
#
# 실제 배포되는 fill_corrector.FillCorrector 코드 그대로 교차검증한다.
# (검증용 별도 구현을 쓰지 않으므로, 여기 나온 숫자가 곧 탑재된 모델의 성능이다.)
#
#   사용법:
#       python eval_fill_corrector.py                        # 기본 경로 자동 탐색
#       python eval_fill_corrector.py data/trial_summary.csv
#
# 검증 시나리오
#   1) 현재 시스템          : 보정 없음 (기준선)
#   2) 학습된 용기          : 무작위 5-겹 교차검증
#   3) 처음 보는 용기        : 용기별 홀드아웃 (한 용기를 통째로 빼고 학습)
#   4) 온라인 적응 학습      : 시간순으로 과거 시행만 보고 다음 시행을 보정
#   5) 새 용기 적응 곡선     : 새 용기를 몇 회 겪으면 얼마나 좋아지는가

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import fill_corrector as fc
from train_fill_corrector import _find_default_path, load_dataset


def _metrics(err: np.ndarray) -> Dict[str, float]:
    err = np.asarray(err, dtype=float)
    return {
        'MAE': float(np.abs(err).mean()),
        'RMSE': float(np.sqrt((err ** 2).mean())),
        'over5': float(100.0 * (np.abs(err) > 5.0).mean()),
        'bias': float(err.mean()),
        'max': float(np.abs(err).max()),
    }


def _row(label: str, m: Dict[str, float], base: Optional[Dict[str, float]] = None) -> str:
    text = (f'{label:<24} MAE {m["MAE"]:5.2f}   RMSE {m["RMSE"]:5.2f}   '
            f'|오차|>5%p {m["over5"]:5.1f}%   편향 {m["bias"]:+5.2f}   최악 {m["max"]:5.1f}')
    if base is not None:
        text += f'   (MAE {(m["MAE"] - base["MAE"]) / base["MAE"] * 100:+.0f}%)'
    return text


def _fit_predict(X: np.ndarray, y: np.ndarray,
                 train_idx: Sequence[int], test_idx: Sequence[int]) -> np.ndarray:
    """학습 인덱스로 모델을 만들고 테스트 인덱스를 예측한다 (실제 배포 코드 경로)."""
    model = fc.FillCorrector()
    model.fit_batch(X[list(train_idx)], y[list(train_idx)])
    return np.array([model.predict(X[i])[0] for i in test_idx])


def eval_random_kfold(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """무작위 5-겹. 학습에 포함된 용기를 다시 쓰는 상황."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    folds = np.array_split(order, k)
    pred = np.zeros(len(y))
    for f in folds:
        train = np.setdiff1d(order, f)
        pred[f] = _fit_predict(X, y, train, f)
    return y - pred


def eval_group_holdout(X: np.ndarray, y: np.ndarray, groups: Sequence[str]) -> np.ndarray:
    """용기별 홀드아웃. 학습 때 한 번도 본 적 없는 용기로 시험한다."""
    groups = np.asarray(groups)
    pred = np.zeros(len(y))
    for g in np.unique(groups):
        test = np.flatnonzero(groups == g)
        train = np.flatnonzero(groups != g)
        pred[test] = _fit_predict(X, y, train, test)
    return y - pred


def eval_online(X: np.ndarray, y: np.ndarray, warmup: int = 60) -> np.ndarray:
    """
    온라인 적응 학습. 시간순으로 진행하며 '과거 시행만' 보고 다음 시행을 예측한 뒤,
    실측이 나오면 그 시행으로 모델을 갱신한다. 실제 운용과 동일한 조건이다.
    """
    model = fc.FillCorrector()
    model.fit_batch(X[:warmup], y[:warmup])                        # 초기 표본으로 부트스트랩
    pred = np.full(len(y), np.nan)
    for i in range(warmup, len(y)):
        pred[i] = model.predict(X[i])[0]                           # 먼저 예측 (미래 정보 없음)
        model.update(X[i], y[i])                                   # 실측 확정 후 학습
    mask = ~np.isnan(pred)
    return np.where(mask, y - np.nan_to_num(pred), np.nan)


def eval_new_cup_adaptation(X: np.ndarray, y: np.ndarray,
                            groups: Sequence[str]) -> List[Dict[str, Any]]:
    """새 용기를 처음 만난 뒤 몇 회 만에 적응하는지."""
    groups = np.asarray(groups)
    buckets = [('1~10회', 0, 10), ('11~30회', 10, 30), ('31회~', 30, 10 ** 9)]
    acc: Dict[str, List[List[float]]] = {b[0]: [[], []] for b in buckets}

    for g in np.unique(groups):
        test = np.flatnonzero(groups == g)
        train = np.flatnonzero(groups != g)
        model = fc.FillCorrector()
        model.fit_batch(X[train], y[train])                        # 다른 용기들로만 사전학습
        errs = []
        for i in test:
            errs.append(y[i] - model.predict(X[i])[0])             # 예측 먼저
            model.update(X[i], y[i])                               # 그다음 학습
        errs = np.array(errs)
        base = y[test]
        for name, lo, hi in buckets:
            seg = slice(lo, min(hi, len(errs)))
            if lo >= len(errs):
                continue
            acc[name][0].extend(np.abs(errs[seg]).tolist())
            acc[name][1].extend(np.abs(base[seg]).tolist())

    out = []
    for name, _, _ in buckets:
        ai, cur = acc[name]
        if ai:
            out.append({'구간': name, 'n': len(ai),
                        'AI': float(np.mean(ai)), '현재': float(np.mean(cur))})
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='급수 보정 모델 검증')
    ap.add_argument('summary', nargs='?', default=None, help='trial_summary.csv 경로')
    ap.add_argument('--warmup', type=int, default=60, help='온라인 학습 부트스트랩 표본 수')
    args = ap.parse_args(argv)

    path = args.summary or _find_default_path()
    if path is None or not os.path.exists(path):
        print('[오류] trial_summary.csv 를 찾지 못했습니다. 경로를 지정하세요.')
        return 1

    print(f'[검증] 데이터: {path}')
    X, y, meta = load_dataset(path)
    cups = [m['Cup_Name'] for m in meta]

    print(f'\n설정: 정규화 alpha={fc.RIDGE_ALPHA:g}, 보정 클리핑 ±{fc.MAX_CORRECTION_PCT:g}%p, '
          f'망각계수={fc.FORGET_FACTOR:g}, 특징 {fc.N_FEATURES}개')
    print('=' * 108)

    base = _metrics(y)
    print(_row('0) 현재 시스템', base))
    print(_row('1) 학습된 용기(5-겹)', _metrics(eval_random_kfold(X, y)), base))
    print(_row('2) 처음 보는 용기', _metrics(eval_group_holdout(X, y, cups)), base))

    on = eval_online(X, y, warmup=args.warmup)
    mask = ~np.isnan(on)
    if mask.any():
        print(_row('3) 온라인 적응 학습', _metrics(on[mask]), _metrics(y[mask])))
        print(f'{"":<24} └ 같은 구간 현재 시스템 기준선: MAE {np.abs(y[mask]).mean():.2f}, '
              f'|오차|>5%p {100 * (np.abs(y[mask]) > 5).mean():.1f}%  (n={int(mask.sum())})')
    print('=' * 108)

    rows = eval_new_cup_adaptation(X, y, cups)
    if rows:
        print('\n[새 용기 적응] 다른 용기들로만 사전학습 → 새 용기를 겪으며 온라인 갱신')
        print(f'  {"구간":<10}{"n":>5}{"AI 적용":>10}{"현재":>10}')
        for r in rows:
            print(f'  {r["구간"]:<10}{r["n"]:>5}{r["AI"]:>10.2f}{r["현재"]:>10.2f}')

    print('\n※ 2)는 5종 용기밖에 없어 한 종을 통째로 빼면 외삽이 됩니다.')
    print('  실제 운용에서는 그 용기를 몇 회 겪는 즉시 3)/[새 용기 적응] 쪽으로 수렴합니다.')
    print('※ 위 수치는 오프라인 재현입니다. 실제로는 보정이 정지 시점을 바꾸므로')
    print('  특징값도 함께 이동합니다. 실기 개선폭은 이보다 다소 작을 수 있습니다.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
