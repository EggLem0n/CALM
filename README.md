# CALM — Congestion-Aware Lookahead Mobility

**가상 공장에서 수백 대의 로봇이 서로 막히지 않게 움직이도록, 혼잡을 미리 예측해 길을 비켜 가게 만드는 프로젝트입니다.**

## 무엇을 푸는 문제인가

50×80 격자의 자동차 부품 공급 공장에 AMR(자율주행 운반로봇)을 투입해 30분 동안 픽업·배송을 반복시킵니다. 로봇을 늘리면 처리량도 늘어야 할 것 같지만, 실제로는 이렇게 됩니다.

| 투입 로봇 | 30분간 배달 건수 |
|---:|---:|
| 300대 | 4,893건 |
| 450대 | 6,097건 |
| 750대 | 6,388건 |

**2.5배를 더 넣었는데 배달은 1.3배**에 그칩니다. 450대 부근에서 곡선이 꺾이는데, 로봇이 늘수록 서로의 길을 막기 때문입니다. 로봇을 더 사는 것으로는 이 벽을 넘을 수 없습니다.

CALM은 다른 길을 갑니다. **어디가 곧 막힐지 미리 예측해서, 로봇들이 그 자리를 피해 돌아가게** 합니다.

## 어떻게 동작하는가 — 네 단계가 고리를 이룹니다

```
  ①  경로계획         ②  데이터 생성        ③  혼잡 예측
   calm/pibt/    →    calm/dataset/    →    calm/forecast/
  로봇을 실제로       그 결과를 히트맵      과거 10프레임을 보고
  움직인다            으로 쌓는다           다음 10프레임을 그린다
      ↑                                          │
      └──────────────────────────────────────────┘
         예측된 혼잡을 이동 비용에 얹어 되먹임 (λ)

                    ④  calm/eval/  ← λ를 켜고 끄며 효과를 측정
```

핵심은 **닫힌 고리**라는 점입니다. 3단계에서 학습한 예측 모델이 1단계 솔버 안으로 다시 들어갑니다. 되먹임의 세기가 λ(`congestion_weight`)이고, **λ = 0이면 되먹임이 사라져 순수 PIBT**가 됩니다. 그래서 같은 코드가 실험군과 대조군을 모두 담당하고, 4단계는 그 스위치를 켜고 끄며 차이를 재는 자리입니다.

### 쓰인 알고리즘

- **경로계획: lifelong PIBT** — 로봇들이 우선순위를 주고받으며 한 걸음씩 동시에 정하는 방식입니다. 목적지에 도착하면 새 작업을 받아 계속 움직입니다(lifelong). 충돌은 알고리즘이 원천적으로 막아 줍니다.
- **혼잡 예측: SimVP** — 영상의 다음 장면을 예측하듯, 혼잡 히트맵 10장을 보고 다음 10장을 그립니다. [OpenSTL](https://github.com/chengtan9907/OpenSTL) 구현을 씁니다.

> **왜 PIBT로 바꿨나:** 이전 버전(MACPF)은 로봇을 하나씩 순서대로 계획하는 방식이라 밀도가 오르면 계획 시간이 폭발했습니다. 200대·60초 시나리오에 **641초**가 걸렸는데, PIBT는 거의 선형이라 같은 조건을 **0.36초**에 풉니다.

> **브랜치:** `CLC` = 현재 개발 · `main` = 이전 스냅샷 · `legacy/macpf` = 초기 MACPF(보존용)

## 빠르게 돌려보기

경로계획과 데이터 생성은 **numpy와 pyyaml만 있으면** 됩니다. GPU도 torch도 필요 없습니다.

```bash
python scripts/01_pibt/preview_map.py         # 공장 맵을 만들고 PNG로 확인
```

`data/maps/<타임스탬프>/factory_map_preview.png`가 생깁니다. 여기까지 되면 1단계가 정상입니다.

**설치는 필요 없습니다.** 각 런처가 저장소 루트를 스스로 `sys.path`에 넣기 때문에, 클론한 자리에서 바로 실행됩니다. 다만 노트북이나 다른 프로젝트에서 `import calm`을 하려면 한 번 설치해 두세요.

```bash
pip install -e .
```

## 파이프라인 실행

`scripts/` 폴더의 **숫자가 곧 실행 순서**입니다. 모든 명령은 저장소 루트에서 실행합니다.

### ① 맵 확인 · ② 데이터셋 생성 — 가볍습니다

```bash
python scripts/01_pibt/preview_map.py

python scripts/02_dataset/generate_dataset.py                 # 기본 스윕
python scripts/02_dataset/generate_dataset.py \
    --rounds 4 --seconds 3600 --num_of_process 8              # 크게 돌리기

python scripts/02_dataset/render_heatmap.py \
    --dataset data/heatmap_dataset/<타임스탬프>                # 결과를 영상으로 확인
```

데이터셋은 `data/heatmap_dataset/<타임스탬프>/`에 `episode_*.npz` + `metadata.json`으로 저장됩니다. 한 에피소드가 **(로봇 대수) × (분산 비율)** 격자의 한 칸에 해당하고, 라운드를 늘리면 같은 칸을 다른 시드로 다시 돕니다.

주요 인자는 `--help`가 항상 정답입니다. 자주 쓰는 것만 옮기면 이렇습니다.

| 인자 | 뜻 |
|---|---|
| `--min-agents` / `--max-agents` / `--agent-step` | 로봇 대수 축 (기본 300~750, 50 간격) |
| `--min-frac` / `--max-frac` / `--frac-step` | 출발 분산 비율 축 (0 = 통로 집중, 1 = 전면 분산) |
| `--rounds` | 격자 전체를 몇 번 반복할지 (매번 새 시드) |
| `--seconds` | 에피소드 길이(초) |
| `--num_of_process` | 병렬 프로세스 수 |
| `--base-seed` | 시드 시작값 (에피소드마다 +1) |
| `--center-value` / `--step-value` | 혼잡 라벨의 정의 |

### ③ 혼잡 예측 학습 — GPU가 필요합니다

먼저 환경과 서드파티 코드를 준비합니다.

```bash
conda env create -f environment.yml
conda activate OpenSTL
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# OpenSTL은 서드파티라 저장소에 포함돼 있지 않습니다
cd calm/forecast
git clone -b OpenSTL-Lightning https://github.com/chengtan9907/OpenSTL.git
```

> **윈도우 필수 패치:** `OpenSTL/openstl/utils/main_utils.py`의 gcc/head 환경 탐지 부분을 `try/except`로 감싸세요. 윈도우에는 gcc가 없어 그대로 두면 크래시합니다.

```bash
python scripts/03_forecast/train.py                # 학습 → work_dirs/
TEST_ONLY=1 python scripts/03_forecast/train.py    # 학습 건너뛰고 best 체크포인트만 테스트

python scripts/03_forecast/predict.py              # 체크포인트가 제 값을 내는지 검증
python scripts/03_forecast/visualize.py            # 그림·영상 일괄 생성
python scripts/03_forecast/plot_loss.py 20         # 20에폭까지 손실곡선
python scripts/03_forecast/per_frame_accuracy.py   # t+1 ~ t+10 구간별 정확도
```

### ④ 혼잡회피 효과 측정

```bash
python scripts/04_eval/evaluate.py                 # 비교 그리드 전체 (+ 영상)
python scripts/04_eval/evaluate.py --no-video      # 숫자만 (빠름)

# 이미 끝난 실행의 영상만 다시 인코딩 (재시뮬 없음)
python scripts/04_eval/evaluate.py --video-only \
    --from-run reports/CALM_comparison/<타임스탬프>

# metrics.csv 에서 요약 문서만 다시 생성
python scripts/04_eval/analyze.py reports/CALM_comparison/<타임스탬프>
```

결과는 `reports/CALM_comparison/<타임스탬프>/metrics.csv`에 쌓입니다. 각 컬럼이 무엇을 재고 어느 방향이 좋은지는 [`metrics_glossary.md`](metrics_glossary.md)에 정리해 두었습니다.

한 조건을 빠르게만 보고 싶으면 격자를 한 칸으로 좁히면 됩니다.

```bash
python scripts/04_eval/evaluate.py --no-video \
    --min-agents 500 --max-agents 500 --min-frac 0 --max-frac 0 \
    --weights 0 0.5 1 2 4
```

## 저장소 구조

**두 층으로 갈라져 있습니다.** `calm/`은 계산만 하는 라이브러리이고, `scripts/`는 실행 진입점입니다.

```
scripts/<번호>_<단계>/   ← 인자를 파싱해서 넘겨준다 (로직 없음)
        │
        ▼
calm/<단계>/             ← 계산은 전부 여기 (argparse 없음)
        │
        ▼
calm/utils/              ← 네 단계가 함께 쓰는 공용 인프라
```

이렇게 갈라 둔 덕에 얻는 것이 셋 있습니다.

1. **노트북에서 바로 씁니다.** `calm.eval.run_sweep(args)`처럼 함수를 그냥 부르면 됩니다. `calm/` 안에는 `sys.argv`를 읽는 코드가 없어서, 명령줄을 흉내 낼 필요가 없습니다.
2. **솔버를 import해도 matplotlib이 안 딸려옵니다.** 그리는 코드는 네 패키지 모두 `viz/` 하위에 모여 있고, matplotlib을 import하는 파일은 `viz/` 안과 `utils/backend.py`뿐입니다.
3. **torch도 마찬가지입니다.** `import calm`은 이름을 지연 해석하므로(PEP 562), torch는 `CongestionPredictor`에 실제로 손댈 때만 올라옵니다.

### `calm/` — 라이브러리

```
calm/
  __init__.py               한 줄 import 창구 (아래 "API로 쓰기") + 경로 상수

  utils/                    공용 인프라
    paths.py                    저장소 루트 · <종류>/<타임스탬프> 실행 폴더
    console.py                  제자리에서 갱신되는 진행 표시판
    parallel.py                 프로세스 풀 + 즉시 반응하는 Ctrl+C
    sweep.py                    대수 × 분산 격자 축 (생성·평가 공용)
    backend.py                  matplotlib Agg 백엔드 + ffmpeg 경로

  pibt/                   ① lifelong PIBT 경로계획 엔진 — numpy/pyyaml만 필요
    solver.py                   솔버 본체 (우선순위 상속 · 백트래킹 · 혼잡 페널티)
    scenario.py                 출발점·목표 선택 (통로 집중 ↔ 전면 분산)
    grid.py, distance.py        통행 가능 판정·이웃 / 목표별 BFS 거리장
    metrics.py                  점유맵 · 혼잡 라벨 정의 · 충돌 검사
    factory_map_generator.py    50×80 공장 맵 생성
    config.py, types.py
    viz/                        경로 애니메이션 (matplotlib은 여기서만)

  dataset/                ② 혼잡 히트맵 데이터셋
    generate.py                 에피소드 생성 (PIBT 호출 → npz)
    viz/heatmap_video.py        npz → MP4 (재시뮬 없이 기록만 재생)

  forecast/               ③ 혼잡 예측 모델 (SimVP)
    train.py                    학습 파이프라인
    predict.py                  추론 래퍼 (best.ckpt → nn.Module) · ①로 되먹임
    per_frame_accuracy.py       구간별 정확도 계산 (CSV)
    viz/                        손실곡선 · GT vs 예측 그림 · 비교 영상
    OpenSTL/                    서드파티 클론 (저장소 미포함)

  eval/                   ④ 혼잡회피 vs 순수 PIBT 비교
    pipeline.py                 스윕 전체 구동
    runner.py                   한 조건 실행 · 채점 (지표 정의가 여기 한 곳에)
    sweep_jobs.py, cli.py       작업 목록 생성 / 인자 정의
    report.py, board.py         콘솔 표 · 요약 / 진행 보드
    make_analysis_summary.py    metrics.csv → analysis_summary.md
    viz/                        비교 영상 · NVENC 스케줄링 · 지표 표 이미지
```

### `scripts/` — 실행 진입점

폴더 이름의 숫자가 실행 순서입니다. 각 런처는 파서를 만들고, `calm` 쪽 `add_*_args`로 플래그를 붙이고, 파싱한 결과를 `run_*`에 넘기는 것이 전부입니다. 열한 개를 합쳐 309줄입니다.

```
scripts/
  _bootstrap.py              저장소 루트를 sys.path에 + OpenMP 가드 (런처 공용)
  01_pibt/preview_map.py     공장 맵 생성 + 미리보기 PNG
  02_dataset/                generate_dataset.py · render_heatmap.py
  03_forecast/               train.py · predict.py · per_frame_accuracy.py
                             visualize.py · plot_loss.py
  04_eval/                   evaluate.py · analyze.py
```

숫자는 `scripts/`에만 붙습니다. `calm/` 쪽은 `import calm.01_pibt`가 문법 오류라 붙일 수 없습니다.

## API로 쓰기

노트북이나 외부 스크립트에서는 `calm` 하나만 import하면 됩니다.

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

혼잡회피를 켜려면 예측기를 넘깁니다. **여기서 처음으로 torch가 로드됩니다.**

```python
predictor = calm.CongestionPredictor(device="cuda")
paths, summary = calm.plan(..., congestion_predictor=predictor,
                           congestion_weight=0.05, predict_every=10)
```

서브패키지를 직접 import해도 됩니다: `from calm import pibt as mapf`.

## 결과물이 저장되는 곳

`reports/`는 여러 코드가 함께 쓰므로, **어떤 코드가 언제 만들었는지** 구분되도록 종류별·시각별로 나눕니다.

```
reports/
├── congestion_prediction/<타임스탬프>/   # ③ 학습 결과 그림 · 영상 · 손실곡선
└── CALM_comparison/<타임스탬프>/         # ④ 비교 실행 (metrics.csv · 표 · 영상)
```

## 저장소에 포함하지 않는 것

용량이 크거나 외부 코드라서 제외했습니다. 전부 다시 만들거나 받을 수 있습니다.

- `data/` (데이터셋) · `models/`, `work_dirs/` (체크포인트·로그) · `reports/` (영상·그림) — 고정 시드로 재생성 가능
- `calm/forecast/OpenSTL/` — 서드파티(자체 `.git`). 위 클론 + 패치로 받으세요
