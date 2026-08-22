# CALM — Congestion-Aware Lookahead Mobility

다중 AMR 경로계획(MAPF)을 **lifelong PIBT** 솔버로 풀고, 그 위에 학습된 **혼잡(congestion) 히트맵 예측**을 결합해, 예측된 혼잡을 피하도록 계획을 조향하는 프로젝트입니다.

기존 MACPF(우선순위 계획, prioritized planning)에서 **솔버를 lifelong PIBT로 교체**해 새로 시작했습니다. 우선순위 계획은 밀도가 오르면 계획시간이 폭발했지만(200대·60초 ≈ 641s), PIBT는 거의 선형이라 같은 조건을 **~0.36s**에 풉니다.

> **브랜치:** `CLC` = 현재 개발 브랜치 · `main` = 이전 스냅샷 · `legacy/macpf` = 초기 MACPF (보존용)

## 레이어 구조

이 레포는 **라이브러리(`calm/`) / 실행(`scripts/`)** 두 층으로 나뉩니다.
`scripts/` 안에는 로직이 한 줄도 없고, `calm/` 안에는 **`parse_args()`도 `sys.argv`도 없습니다.**
런처가 파서를 만들어 파싱하고, 라이브러리는 파싱된 결과만 받습니다 (Isaac Lab의
`AppLauncher.add_app_launcher_args(parser)` 방식과 같습니다).

```
scripts/<단계>/    ← 실행은 전부 여기 (파서 생성 · 파싱 · 호출)
   │
   ▼
calm/__init__.py  ← 단일 import 창구 (경로 상수 + 지연 로딩 진입점)
   │
   ▼
calm/pibt · dataset · forecast · eval
   │
   ▼
calm/utils        ← console · parallel · paths · sweep  (공용 인프라)
```

### `calm/` — 라이브러리

그리는 코드는 네 패키지 모두 `viz/` 하위 패키지에 모아 두었으므로, **matplotlib을
import하는 파일은 `viz/` 안과 `utils/backend.py`뿐**입니다. 솔버나 평가부를 import해도
그리기 의존성이 딸려오지 않습니다.

어떤 플래그가 존재하는지는 그 패키지의 도메인 지식이므로 **인자 정의**(`add_*_args(parser)`)는
`calm/` 안에 남아 있습니다. 반면 **파싱**은 런처가 합니다. 그래서 `calm.eval.run_sweep(args)`를
노트북에서 직접 부를 수 있고, 명령줄을 흉내 낼 필요가 없습니다.

```
calm/
  __init__.py               한 줄 import 창구 (아래 "API로 쓰기") + CALM_*_DIR 경로 상수

  utils/                    공용 인프라 (Isaac Lab의 isaaclab.utils에 대응)
    paths.py                    CALM_ROOT_DIR · CALM_DATA/MODELS/REPORTS/CONFIGS_DIR
                                · <kind>/<yymmdd_hhmm> 실행 폴더
    console.py                  ANSI 제자리 진행보드 (AnsiBoard / live / bar / boxed)
    parallel.py                 프로세스 풀 + 즉시 반응하는 Ctrl+C
    sweep.py                    대수 × 분산비율 그리드 축 (생성·평가 공용)
    backend.py                  Agg 백엔드 + ffmpeg 경로 (그리는 모듈만 import)

  pibt/                     ★ lifelong PIBT MAPF 엔진 — numpy/pyyaml만 필요
    solver.py                   솔버 본체 (우선순위 상속·백트래킹·혼잡 페널티)
    scenario.py                 출발/목표 선택 (staging / distributed)
    grid.py, distance.py        walkability·이웃 / BFS 거리장
    metrics.py                  occupancy · additive congestion · collision
    factory_map_generator.py    50×80 공장 맵 (matplotlib 의존 없음)
    config.py, types.py
    viz/                        그리기 전용
      animate.py                  MP4 렌더
      overlays.py                 목표 마커·경로선·할당 커서
      background.py               맵 배경
      factory_map.py              맵 미리보기 PNG · 콘솔 요약

  dataset/                  혼잡 히트맵 데이터셋
    generate.py                 에피소드 생성 (PIBT 호출 → npz)
    viz/
      heatmap_video.py            데이터셋 → MP4 (재시뮬 없이 npz만 읽음)

  forecast/                 혼잡 예측 모델 (OpenSTL / SimVP)
    train.py                    학습 파이프라인 (SimVP)
    predict.py                  추론 래퍼 (best.ckpt → nn.Module)
    per_frame_accuracy.py       t+1..t+10 구간별 정확도 (계산부, CSV)
    viz/
      figures.py                  학습 결과 그림 일괄 생성
      loss_curve.py               train/val 손실곡선
      accuracy_curve.py           구간별 오차 그래프
      window_strip.py             GT vs 예측 10프레임 스트립
    OpenSTL/                    서드파티 클론 (레포 미포함)

  eval/                     혼잡회피 PIBT vs vanilla 비교
    pipeline.py                 sweep 전체 구동 (main)
    runner.py                   한 조건 실행·채점 (solve / evaluate / run_job)
    sweep_jobs.py, cli.py       작업 생성 / 인자 정의
    report.py, board.py         콘솔 표·요약 출력 / 진행보드
    viz/
      video.py                    비교영상 렌더
      sched.py                    NVENC 세션 스케줄링
      table.py                    λ별 지표 표 이미지
    make_analysis_summary.py    metrics.csv → analysis_summary.md
```

### `scripts/` — 실행

파이프라인 단계별로 폴더가 나뉘고, **폴더 이름의 숫자가 실행 순서**입니다.
각 런처는 파서를 만들고, `calm` 쪽 `add_*_args`로 플래그를 붙이고, 파싱한 결과를 `run_*`에 넘깁니다.
숫자 접두사는 `scripts/`에만 붙습니다 — 여기는 import 대상이 아니라 경로로만 쓰이기 때문입니다
(`calm/` 쪽은 `import calm.01_pibt`가 문법 오류라 붙일 수 없습니다).

```
scripts/
  _bootstrap.py              repo 루트 sys.path + OpenMP 가드 (런처 공용)
  01_pibt/
    preview_map.py             공장 맵 생성 + 미리보기 PNG
  02_dataset/
    generate_dataset.py        PIBT sweep → 혼잡 히트맵 데이터셋
    render_heatmap.py          데이터셋 → 에피소드별 MP4
  03_forecast/
    train.py                   SimVP 학습 (과거 10프레임 → 미래 10프레임)
    predict.py                 체크포인트 추론 검증
    per_frame_accuracy.py      예측 구간별 정확도 CSV + 그래프
    visualize.py               학습 결과 그림 · GT vs 예측 영상
    plot_loss.py               학습/검증 손실곡선
  04_eval/
    evaluate.py              ★ 혼잡회피 vs vanilla 비교 sweep (+ 영상)
    analyze.py                 metrics.csv → analysis_summary.md 재생성
```

모든 스크립트는 **레포 루트에서** 실행합니다 (`sys.path`를 스스로 잡습니다).

## 실행

### 1) PIBT 엔진 · 데이터 생성 (가벼움 — numpy, pyyaml)

```bash
python scripts/01_pibt/preview_map.py
python scripts/02_dataset/generate_dataset.py                                   # 기본 sweep
python scripts/02_dataset/generate_dataset.py --rounds 4 --seconds 3600 --num_of_process 8
python scripts/02_dataset/render_heatmap.py --dataset data/heatmap_dataset/<타임스탬프>
```

### 2) 혼잡 예측 (OpenSTL, GPU/CUDA)

```bash
conda env create -f environment.yml
conda activate OpenSTL
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# OpenSTL 클론 (서드파티 — 레포에 미포함)
cd calm/forecast
git clone -b OpenSTL-Lightning https://github.com/chengtan9907/OpenSTL.git
# 필수 패치: OpenSTL/openstl/utils/main_utils.py 의 gcc/head 환경탐지부를
#            try/except 로 감싸기 (Windows엔 gcc가 없어 그대로면 크래시)
```

```bash
python scripts/03_forecast/train.py                    # 학습 -> work_dirs/
TEST_ONLY=1 python scripts/03_forecast/train.py        # 학습 건너뛰고 best 체크포인트 테스트
python scripts/03_forecast/predict.py                  # 추론 검증
python scripts/03_forecast/visualize.py                # 학습 결과 그림
python scripts/03_forecast/plot_loss.py 20             # 20 에폭까지 손실곡선
python scripts/03_forecast/per_frame_accuracy.py       # t+1..t+10 정확도
```

### 3) 혼잡회피 평가·비교

```bash
python scripts/04_eval/evaluate.py                                                     # 비교 그리드 (+ 영상)
python scripts/04_eval/evaluate.py --video-only --from-run reports/CALM_comparison/<타임스탬프>   # 영상만 재인코딩
python scripts/04_eval/analyze.py reports/CALM_comparison/<타임스탬프>                 # csv -> analysis_summary.md
```

## API로 쓰기

노트북이나 외부 스크립트에서는 `calm` 하나만 import하면 됩니다.
모든 진입점은 **첫 접근 시점에** 해석되므로(PEP 562), `import calm` 자체는 비용이 없고
torch·matplotlib은 `CongestionPredictor` / `animate()`에 손댈 때만 로드됩니다.
Isaac Lab처럼 서브패키지를 직접 import해도 됩니다 (`from calm import pibt as mapf`).

```python
import calm
import numpy as np

env      = calm.build_map()                                 # 50x80 공장 맵
walkable = np.asarray(env["walkable_map"]).astype(bool)
config   = calm.load_config()                               # configs/default.yaml
starts, _ = calm.select_starts(env, walkable, config)

paths, summary = calm.plan(                                 # lifelong PIBT
    starts,
    calm.walkable_points(env, "pickup_points", walkable),
    calm.walkable_points(env, "delivery_points", walkable),
    walkable, config,
)

positions = calm.positions_from_paths(paths, config.max_time)   # (T, N, 2)
labels    = calm.congestion_labels(positions, *walkable.shape)  # (T, H, W)
print(summary["total_completed_deliveries"], calm.count_collisions(positions))
```

혼잡회피를 켜려면 예측기를 넘깁니다.

```python
predictor = calm.CongestionPredictor(device="cuda")         # 여기서만 torch 로드
paths, summary = calm.plan(..., congestion_predictor=predictor,
                           congestion_weight=0.05, predict_every=10)
```

## 출력(`reports/`) 분류

`reports/`는 여러 코드가 공유하므로, **어떤 코드가 / 언제 만들었는지** 구분되도록 코드별·실행시각별로 저장합니다.

```
reports/
├── congestion_prediction/<yymmdd_hhmm>/   # visualize · predict · 손실곡선 산출물
└── CALM_comparison/<yymmdd_hhmm>/          # evaluate 비교 실행 (metrics.csv·표·영상)
```

## 레포에 포함하지 않는 것 (재생성 / 외부)

- `data/` (히트맵 데이터셋), `models/`·`work_dirs/` (체크포인트·로그), `reports/` (영상·그림) — 용량이 커서 제외, 고정 시드로 재생성 가능
- `OpenSTL/` — 서드파티(자체 `.git`). 위 클론 + 패치로 받기

## 주요 인자 (`generate_dataset.py`)

- `--num_of_process` 병렬 프로세스 수
- `--base-seed` 시드 시작값(에피소드마다 +1, 스폰 위치 결정)
- `--seconds` 에피소드 길이(초) · `--rounds` 대수 sweep 왕복 횟수
- `--min-agents` / `--max-agents` 대수 sweep 범위
- `--distributed-start-frac` 맵 전체 분산 출발 비율(0~1)
- `--center-value` / `--step-value` 혼잡 라벨 정의

데이터는 `data/heatmap_dataset/<타임스탬프>/episode_*.npz` + `metadata.json`로 저장됩니다.
