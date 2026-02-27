# TODO: DexMachina 기반 Bimanual Dexterous Hand mini-VLA

## 개요

DexMachina 환경을 활용하여 양손 Allegro Hand 조작을 위한 mini-VLA를 구현.
Human demonstration (ARCTIC 데이터셋)을 활용한 imitation learning 파이프라인 구축.

### 환경 정보: DexMachina

| 항목 | 내용 |
|------|------|
| **시뮬레이터** | Genesis (GPU 물리 시뮬레이션) |
| **로봇 손** | Allegro Hand × 2 (양손), 총 6종 지원 |
| **Action Dim** | 44 (hybrid mode: 22 DoF × 2 양손) |
| **State Dim** | 410 (default) / 510 (with contact force obs) |
| **데이터셋** | ARCTIC (인간 양손 조작 비디오) |
| **물체** | box, ketchup, laptop, mixer, notebook, waffleiron (6종) |
| **GitHub** | https://github.com/MandiZhao/dexmachina |

### 현재 상태 vs 목표

| 항목 | 현재 (Meta-World) | 목표 (DexMachina) |
|------|-------------------|-------------------|
| Hand | 2-finger gripper | Allegro Hand × 2 |
| Action Dim | 4 | 44 (hybrid: 22 × 2) |
| State Dim | 39 | 410 (양손 joints + object + fingertip) |
| Task | Push, Pick | Bimanual manipulation |
| Simulator | MuJoCo | Genesis |

---

## Phase 1: DexMachina 환경 설치

### 1.1 DexMachina 설치
```bash
# 1. Conda 환경 생성
conda create -n dexmachina python=3.10
conda activate dexmachina

# 2. 의존성 설치
pip install torch==2.5.1

# 3. Genesis (커스텀 포크) 설치
git clone https://github.com/MandiZhao/Genesis.git
cd Genesis && pip install -e .
pip install libigl==2.5.1

# 4. rl_games (커스텀 포크) 설치
git clone https://github.com/MandiZhao/rl_games.git
cd rl_games && pip install -e .

# 5. 추가 패키지
pip install gymnasium ray seaborn wandb trimesh moviepy==1.0.3

# 6. DexMachina 설치
git clone https://github.com/MandiZhao/dexmachina.git
cd dexmachina && pip install -e .
```

### 1.2 환경 검증 (2026-02-27 완료)
- [x] Genesis 시뮬레이터 동작 확인 (v0.3.3, RTX 5090, ~630 FPS)
- [x] Allegro Hand 모델 로딩 확인 (28 DOF, URDF 정상 로드)
- [x] DexMachina 환경 초기화 (obs=410, action=44, ep_len=200)
- [x] Box RL policy 평가 (5000ep, reward 152.7, rollout 정상)
- [x] RL 학습 파이프라인 검증 (mixer dry-run 성공)
- [x] mini-dex-vla wrapper 연동 검증 (reward/step 정상 동작)
- [x] ARCTIC retarget 데이터 확인 (6 objects × allegro_hand)

### 1.3 DexMachina 코드 구조 파악
```
dexmachina/
├── dexmachina/
│   ├── envs/                # 환경 (base_env.py, robot.py, rewards.py 등)
│   ├── rl/                  # RL 학습 (train_rl_games.py, eval_rl_games.py)
│   ├── retargeting/         # ARCTIC → robot motion 변환
│   ├── hand_proc/           # 손 자산 처리
│   └── assets/              # URDF, mesh, retarget 데이터
│       ├── allegro_hand/    # Allegro Hand URDF (left/right 6DOF)
│       ├── arctic/          # ARCTIC 물체 (box, ketchup, laptop 등)
│       ├── retargeted/      # retarget 결과 (allegro_hand/s01/*.pt)
│       └── contact_retarget/# contact retarget (box, mixer만 존재)
├── examples/
│   └── train_allegro_all_objects.sh  # 6개 물체 순차 학습
└── logs/rl_games/           # 학습 체크포인트
```

### 1.4 RL 학습 현황

| Object | Retarget | Contact | RL 학습 | Reward | 상태 |
|--------|----------|---------|---------|--------|------|
| box | ✅ para | ✅ retarget | ✅ 5000ep | 152.7 | **완료** |
| waffleiron | ✅ para | ❌ | 중단 | - | 재학습 필요 |
| mixer | ✅ para | ✅ retarget | ❌ | - | **학습 가능** |
| ketchup | ✅ para | ❌ | ❌ | - | **학습 가능** |
| laptop | ✅ para | ❌ | ❌ | - | **학습 가능** |
| notebook | ✅ para | ❌ | ❌ | - | **학습 가능** |

나머지 학습 명령:
```bash
cd ~/Documents/dexmachina
bash examples/train_allegro_all_objects.sh mixer ketchup laptop notebook waffleiron
```

---

## Phase 2: Genesis 환경 래퍼 구현

### 2.1 환경 래퍼 생성 (완료)
- [x] `envs/dexmachina_env.py` 구현 완료
  - `use_rl_games=True` 사용 (step() 반환 형식 통일)
  - `batch_dofs_info=True` 설정 필수 (Genesis set_dofs_kp 호환)
  - state는 policy obs에서 추출 (410 dim)

### 2.2 State/Action Space 정의 (검증 완료)
- [x] **State dim = 410** (use_rl_games=True 기본 설정)
  - 양손 joint pos/vel, fingertip pos, object pose, demo targets 등 포함
  - -obf -obt 플래그 추가 시 510 (contact force + tip distance 관측 추가)
- [x] **Action dim = 44** (hybrid mode: 22 DoF × 2 hands)
  - hybrid_scales = [0.1, 1.0] (wrist residual + finger absolute)

### 2.3 Task 선택
DexMachina ARCTIC 물체 (Allegro Hand retarget 완료):
- [x] `box`: 박스 열기 **(RL 학습 완료, reward 152.7)**
- [ ] `mixer`: 믹서 조작 (contact retarget 있음)
- [ ] `ketchup`: 케첩 조작
- [ ] `laptop`: 노트북 열기/닫기
- [ ] `notebook`: 노트북 넘기기
- [ ] `waffleiron`: 와플 아이언 열기 (B=2048 제한)

**추천 시작 태스크**: `box` (RL 학습 완료, 바로 데이터 수집 가능)

---

## Phase 3: 데이터 수집 파이프라인

### 3.1 Expert Policy 확보
- [x] **Box**: RL expert 학습 완료 (5000ep, reward 152.7)
  - 체크포인트: `~/Documents/dexmachina/logs/rl_games/allegro_hand/allegro-allegro_box_box30-230-s01-u01_B3072_.../nn/last_allegro_hand_ep_5000_rew_152.72285.pth`
- [ ] **나머지 5개**: RL 학습 실행 필요
  ```bash
  cd ~/Documents/dexmachina
  bash examples/train_allegro_all_objects.sh mixer ketchup laptop notebook waffleiron
  ```

### 3.2 데이터 수집 스크립트 (완료)
- [x] `scripts/collect_dexmachina_data.py` 구현 완료
  - RL expert 또는 demonstration 기반 데이터 수집
  - 수집된 데이터: `data/dexmachina_box.npz`, `data/dexmachina_box_large.npz`

### 3.3 데이터 포맷
```python
# dexmachina_box.npz
{
    "images": (N, 160, 160, 3),  # RGB uint8
    "states": (N, 410),          # float32 (policy obs)
    "actions": (N, 44),          # float32 (hybrid mode)
    "text_ids": (N, T_text),     # int64
    "vocab": dict,
    "metadata": dict,
}
```

---

## Phase 4: 모델 확장

### 4.1 State Encoder 수정 (완료)
- [x] `models/encoders.py` - `BimanualStateEncoderMLP` 구현 (410→512→256→d_model)

### 4.2 Diffusion Head 수정 (완료)
- [x] `models/diffusion_head.py` - `LargerDiffusionPolicyHead` 구현
  - `action_dim`: 4 → 44
  - `hidden_dim`: 128 → 512
  - `diffusion_T`: 16 → 32

### 4.3 Hyperparameter 조정 (완료)
```python
# 기존 (Meta-World)
VLADiffusionPolicy(vocab_size, state_dim=39, action_dim=4, d_model=128)

# DexMachina용
VLADexMachinaPolicy(vocab_size, state_dim=410, action_dim=44, d_model=256)
```

### 4.4 Action Normalization (완료)
- [x] Joint limit 기반 normalization 추가 (`utils/action_normalizer.py`)
  - `AllegroActionNormalizer`: raw joint position <-> [-1, 1] 매핑
  - Wrist (6 DOF): prismatic [-5, 5], revolute [-6.2, 6.2]
  - Finger (16 DOF): URDF joint limits (각 joint별 상이)
  - Bimanual 44 DOF = left 22 + right 22
- [x] Allegro Hand joint limits: URDF에서 추출하여 하드코딩
- [x] `--action-norm-mode` 옵션 추가 (`joint_limits` / `statistical` / `none`)
  - `train_dexmachina.py`: 학습 시 normalization 모드 선택 (기본: joint_limits)
  - `eval_dexmachina.py`: checkpoint에 저장된 모드에 따라 자동 denormalize
  - `collect_dexmachina_data.py`: `--normalize-actions`로 수집 시 정규화 옵션

---

## Phase 5: 학습 및 평가

### 5.1 학습 스크립트 (완료)
- [x] `scripts/train_dexmachina.py` 구현 완료
  - 모델 크기: small/base/large
  - Warmup + cosine decay 스케줄링
  - Gradient clipping (max_norm=1.0)

### 5.2 평가 스크립트 (완료)
- [x] `scripts/eval_dexmachina.py` 구현 완료
  - Expert action 비교 기능
  - Action MSE 지표

### 5.3 평가 지표
- [ ] Task success rate (물체 조작 완료 여부)
- [ ] Object pose tracking error
- [ ] Joint position MSE
- [ ] Contact consistency

---

## Phase 6: 고급 기능 (Optional)

### 6.1 Action Chunking
- [ ] 여러 timestep action 예측 (ACT 스타일)
- [ ] `action_horizon`: 1 → 8/16
- [ ] Temporal consistency 향상

### 6.2 Multi-camera Input
- [ ] 양손 각각의 hand-centric 카메라
- [ ] Third-person 카메라
- [ ] Multi-view fusion

### 6.3 Contact-aware Learning
- [ ] DexMachina의 contact reward 활용
- [ ] Fingertip contact prediction auxiliary task

---

## 파일 구조 (현재)

```
mini-dex-vla/
├── envs/
│   ├── metaworld_env.py           # Meta-World 래퍼
│   ├── metaworld_mt1.py           # Expert policy 시각화
│   └── dexmachina_env.py          # DexMachina/Genesis 래퍼
├── models/
│   ├── encoders.py                # Image/Text/State 인코더
│   ├── fusion.py                  # FusionMLP
│   ├── diffusion_head.py          # Diffusion Policy Head
│   ├── vla_diffusion_policy.py    # Meta-World VLA
│   └── vla_dexmachina.py          # DexMachina VLA (메인)
├── scripts/
│   ├── collect_data.py            # Meta-World 데이터 수집
│   ├── collect_dexmachina_data.py # DexMachina 데이터 수집
│   ├── train.py                   # Meta-World 학습
│   ├── train_dexmachina.py        # DexMachina VLA 학습
│   ├── eval_dexmachina.py         # DexMachina VLA 평가
│   ├── test.py                    # Meta-World 테스트
│   └── inference.py               # 추론 스크립트
├── data/
│   ├── dexmachina_box.npz         # Box 학습 데이터
│   ├── dexmachina_box_large.npz   # Box 대규모 데이터
│   └── metaworld_push_bc.npz      # Meta-World 데이터
└── checkpoints/
    ├── dexmachina_large*.pt       # Large 모델 체크포인트
    ├── dexmachina_base*.pt        # Base 모델 체크포인트
    └── dexmachina_demo*.pt        # Demo 학습 체크포인트
```

---

## 진행 현황

| 순서 | Phase | 작업 | 상태 |
|------|-------|------|------|
| 1 | 1.1 | DexMachina 설치 | ✅ 완료 |
| 2 | 1.2 | 환경 검증 | ✅ 완료 (2026-02-27) |
| 3 | 2.1 | Genesis 래퍼 구현 | ✅ 완료 |
| 4 | 3.1 | Expert policy 확보 | 🔶 box 완료, 5개 남음 |
| 5 | 3.2 | 데이터 수집 스크립트 | ✅ 완료 |
| 6 | 4 | 모델 확장 | ✅ 완료 |
| 7 | 5 | 학습/평가 | ✅ 기본 완료 |
| 8 | 6 | 고급 기능 | ❌ 미착수 |

---

## 참고 자료

### DexMachina
- GitHub: https://github.com/MandiZhao/dexmachina
- 문서: https://mandizhao.github.io/dexmachina-docs
- 논문: https://arxiv.org/abs/2505.24853
- 프로젝트: https://project-dexmachina.github.io

### Genesis 시뮬레이터
- GitHub: https://github.com/MandiZhao/Genesis (커스텀 포크)

### ARCTIC 데이터셋
- 인간 양손 조작 비디오 데이터
- 6가지 articulated objects (box, ketchup, laptop, mixer, notebook, waffleiron)

### 관련 기술
- ACT (Action Chunking): https://github.com/tonyzhaozh/act
- Diffusion Policy: https://github.com/real-stanford/diffusion_policy
