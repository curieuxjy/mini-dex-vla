# TODO: DexMachina 기반 Bimanual Dexterous Hand mini-VLA

## 개요

DexMachina 환경을 활용하여 양손 Allegro Hand 조작을 위한 mini-VLA를 구현.
Human demonstration (ARCTIC 데이터셋)을 활용한 imitation learning 파이프라인 구축.

### 환경 정보: DexMachina

| 항목 | 내용 |
|------|------|
| **시뮬레이터** | Genesis (GPU 물리 시뮬레이션) |
| **로봇 손** | Allegro Hand × 2 (양손), 총 6종 지원 |
| **Action Dim** | 16 DoF × 2 = 32 (양손 finger joints) |
| **데이터셋** | ARCTIC (인간 양손 조작 비디오) |
| **물체** | 노트북, 믹서, 가위 등 5종 |
| **GitHub** | https://github.com/MandiZhao/dexmachina |

### 현재 상태 vs 목표

| 항목 | 현재 (Meta-World) | 목표 (DexMachina) |
|------|-------------------|-------------------|
| Hand | 2-finger gripper | Allegro Hand × 2 |
| Action Dim | 4 | 32 (16 × 2) |
| State Dim | 39 | 200+ (양손 joints + object) |
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

### 1.2 환경 검증
- [ ] Genesis 시뮬레이터 동작 확인
- [ ] Allegro Hand 모델 로딩 확인
- [ ] `bash examples/train_rl.sh` 실행 테스트
- [ ] ARCTIC 데이터 다운로드 및 전처리

### 1.3 DexMachina 코드 구조 파악
```
dexmachina/
├── dexmachina/
│   ├── hand_proc/          # 손 자산 처리
│   ├── rl/                  # RL 학습/평가
│   └── ...
├── examples/
│   └── train_rl.sh         # RL 학습 스크립트
└── assets/                  # 로봇 손, 물체 URDF
```

---

## Phase 2: Genesis 환경 래퍼 구현

### 2.1 환경 래퍼 생성
- [ ] `envs/dexmachina_env.py` 생성
```python
class DexMachinaWrapper:
    """
    DexMachina/Genesis 환경을 mini-VLA 인터페이스로 래핑
    - reset() -> (image, state, info)
    - step(action) -> (image, state, reward, done, info)
    """
    def __init__(self, task_name='laptop', hand_type='allegro'):
        # Genesis 시뮬레이터 초기화
        # 양손 Allegro Hand 로딩
        pass

    def _extract_state(self, obs):
        """
        양손 state 추출:
        - left_hand_joints: (16,)
        - right_hand_joints: (16,)
        - left_fingertip_pos: (4, 3)
        - right_fingertip_pos: (4, 3)
        - object_pose: (7,) position + quaternion
        """
        pass

    def _get_image(self):
        """카메라 렌더링 (RGB)"""
        pass
```

### 2.2 State/Action Space 정의
- [ ] Allegro Hand state 구조 파악:
  - Joint positions: 16 × 2 = 32
  - Joint velocities: 16 × 2 = 32
  - Fingertip positions: 4 × 3 × 2 = 24
  - Object pose: 7 (pos + quat)
  - **Total state_dim: ~95+**
- [ ] Action space: joint position targets (32 dim)

### 2.3 Task 선택
DexMachina ARCTIC 물체:
- [ ] `laptop`: 노트북 열기/닫기
- [ ] `mixer`: 믹서 조작
- [ ] `scissors`: 가위 사용
- [ ] `capsulemachine`: 캡슐머신 조작
- [ ] `box`: 박스 열기

**추천 시작 태스크**: `laptop` (비교적 단순한 articulated object)

---

## Phase 3: 데이터 수집 파이프라인

### 3.1 Expert Policy 확보 (택1)
- [ ] **Option A**: DexMachina RL로 expert 학습
  ```bash
  bash examples/train_rl.sh  # RL policy 학습
  ```
- [ ] **Option B**: ARCTIC human demonstration 활용
  - Kinematic retargeting으로 human → robot motion 변환
  - DexMachina의 `dex-retargeting` 도구 사용

### 3.2 데이터 수집 스크립트
- [ ] `scripts/collect_dexmachina_data.py` 생성
```python
def main():
    env = DexMachinaWrapper(task_name='laptop')
    policy = load_trained_policy(checkpoint_path)

    for episode in range(num_episodes):
        image, state, info = env.reset()
        while not done:
            action = policy.act(state)
            # 데이터 저장
            images.append(image)
            states.append(state)
            actions.append(action)
            texts.append(instruction)

            image, state, reward, done, info = env.step(action)
```

### 3.3 데이터 포맷
```python
# dexmachina_laptop.npz
{
    "images": (N, H, W, 3),           # RGB
    "states": (N, 95+),                # 양손 joints + object
    "actions": (N, 32),                # 양손 joint targets
    "text_ids": (N, T_text),
    "vocab": dict,
    # 추가 (선택)
    "left_fingertips": (N, 4, 3),
    "right_fingertips": (N, 4, 3),
    "object_pose": (N, 7),
}
```

---

## Phase 4: 모델 확장

### 4.1 State Encoder 수정
- [ ] `models/encoders.py` 수정
```python
class BimanualStateEncoderMLP(nn.Module):
    """양손 state를 위한 확장된 encoder"""
    def __init__(self, state_dim, d_model=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, d_model),
        )
        self.ln = nn.LayerNorm(d_model)
```

### 4.2 Diffusion Head 수정
- [ ] `models/diffusion_head.py` 수정
  - `action_dim`: 4 → 32
  - `hidden_dim`: 128 → 512
  - `diffusion_T`: 16 → 32

### 4.3 Hyperparameter 조정
```python
# 기존
VLADiffusionPolicy(vocab_size, state_dim=39, action_dim=4, d_model=128)

# DexMachina용
VLADiffusionPolicy(vocab_size, state_dim=95, action_dim=32, d_model=256)
```

### 4.4 Action Normalization
- [ ] Joint limit 기반 normalization 추가
- [ ] Allegro Hand joint limits: 각 joint별 min/max 값

---

## Phase 5: 학습 및 평가

### 5.1 학습 스크립트 수정
- [ ] `scripts/train_dexmachina.py` 생성
  - 더 큰 batch size (64 → 128)
  - Learning rate scheduling (warmup + decay)
  - Gradient clipping (max_norm=1.0)

### 5.2 평가 스크립트 수정
- [ ] `scripts/test_dexmachina.py` 생성
  - Genesis 환경에서 rollout
  - Task-specific success metric

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

## 파일 구조 (예상)

```
mini-vla/
├── envs/
│   ├── metaworld_env.py          # 기존
│   └── dexmachina_env.py         # 새로 추가
├── models/
│   ├── encoders.py               # 수정 (BimanualStateEncoder)
│   ├── diffusion_head.py         # 수정 (더 큰 hidden_dim)
│   └── vla_diffusion_policy.py
├── scripts/
│   ├── collect_data.py           # 기존
│   ├── collect_dexmachina_data.py # 새로 추가
│   ├── train.py                  # 기존
│   ├── train_dexmachina.py       # 새로 추가
│   ├── test.py                   # 기존
│   └── test_dexmachina.py        # 새로 추가
└── data/
    └── dexmachina_laptop.npz     # 새 데이터셋
```

---

## 우선순위

| 순서 | Phase | 작업 | 예상 난이도 |
|------|-------|------|------------|
| 1 | 1.1 | DexMachina 설치 | ★★☆ |
| 2 | 1.2 | 환경 검증 | ★☆☆ |
| 3 | 2.1 | Genesis 래퍼 구현 | ★★★ |
| 4 | 3.1 | Expert policy 확보 (RL 학습) | ★★★ |
| 5 | 3.2 | 데이터 수집 스크립트 | ★★☆ |
| 6 | 4 | 모델 확장 | ★★☆ |
| 7 | 5 | 학습/평가 | ★★☆ |

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
- 5가지 articulated objects

### 관련 기술
- ACT (Action Chunking): https://github.com/tonyzhaozh/act
- Diffusion Policy: https://github.com/real-stanford/diffusion_policy
