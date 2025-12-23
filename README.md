# mini-dex-vla

DexMachina 환경을 활용한 양손 Allegro Hand 조작을 위한 mini-VLA 구현 프로젝트.

## 개요

| 항목 | 내용 |
|------|------|
| **시뮬레이터** | Genesis (GPU 물리 시뮬레이션) |
| **로봇 손** | Allegro Hand × 2 (양손) |
| **Action Dim** | 16 DoF × 2 = 32 (양손 finger joints) |
| **데이터셋** | ARCTIC (인간 양손 조작 비디오) |
| **물체** | box, laptop, mixer, waffleiron 등 |

### 기존 mini-VLA와 비교

| 항목 | mini-VLA (Meta-World) | mini-dex-vla (DexMachina) |
|------|----------------------|---------------------------|
| Hand | 2-finger gripper | Allegro Hand × 2 |
| Action Dim | 4 | 32 (16 × 2) |
| State Dim | 39 | 95+ (양손 joints + object) |
| Task | Push, Pick | Bimanual manipulation |
| Simulator | MuJoCo | Genesis |

> 기존 mini-VLA 문서는 [README_MINI_VLA.md](README_MINI_VLA.md) 참조

---

## 설치

### 1. Conda 환경 생성

```bash
conda create -n mini-vla python=3.10
conda activate mini-vla
```

### 2. PyTorch 설치

```bash
pip install torch==2.5.1
```

### 3. Genesis (커스텀 포크) 설치

```bash
cd ~/Documents
git clone https://github.com/MandiZhao/Genesis.git
cd Genesis && pip install -e .
pip install libigl==2.5.1
```

### 4. rl_games (커스텀 포크) 설치

```bash
cd ~/Documents
git clone https://github.com/MandiZhao/rl_games.git
cd rl_games && pip install -e .
```

### 5. 추가 패키지 설치

```bash
pip install gymnasium ray seaborn wandb trimesh moviepy==1.0.3
```

### 6. DexMachina 설치

```bash
cd ~/Documents
git clone https://github.com/MandiZhao/dexmachina.git
cd dexmachina && pip install -e .
```

### 7. 호환성 수정

#### NumPy 다운그레이드 (Genesis 호환)
```bash
pip install "numpy<2.0"
```

#### wandb 업그레이드 (NumPy 1.x 호환)
```bash
pip install --upgrade wandb
```

#### torch.load 수정 (PyTorch 2.6+ 호환)

`dexmachina/envs/demo_data.py` 87번째 줄 수정:
```python
# 변경 전
data = torch.load(data_fname)

# 변경 후
data = torch.load(data_fname, weights_only=False)
```

---

## 설치 확인

### Genesis 확인
```bash
python -c "import genesis as gs; print('Genesis version:', gs.__version__)"
```

### DexMachina 확인
```bash
python -c "import dexmachina; print('DexMachina imported successfully!')"
```

### 디렉토리 구조

```
~/Documents/
├── Genesis/              # Genesis 시뮬레이터
├── rl_games/             # RL 학습 프레임워크
├── dexmachina/           # DexMachina 환경
│   └── dexmachina/
│       ├── assets/
│       │   ├── allegro_hand/      # Allegro Hand URDF
│       │   ├── arctic/            # ARCTIC 물체 assets
│       │   └── retargeted/        # Retargeted 데모 데이터
│       ├── envs/                  # 환경 코드
│       └── rl/                    # RL 학습 코드
└── mini-vla/             # 이 프로젝트
```

### 사용 가능한 데이터

| Hand | Subject | Object | 파일 |
|------|---------|--------|------|
| allegro_hand | s01 | box | box_use_01_vector_para.pt |
| allegro_hand | s01 | mixer | mixer_use_01_vector_para.pt |
| allegro_hand | s01 | waffleiron | waffleiron_use_01_vector_para.pt |

---

## 알려진 이슈

### Genesis scene.build() 에러

```
ValueError: setting an array element with a sequence.
The detected shape was (60,) + inhomogeneous part.
```

- **원인**: `link.inertial_quat` 배열 형태 불일치
- **상태**: 디버깅 중

---

## 참고 자료

- [DexMachina GitHub](https://github.com/MandiZhao/dexmachina)
- [DexMachina 문서](https://mandizhao.github.io/dexmachina-docs)
- [DexMachina 논문](https://arxiv.org/abs/2505.24853)
- [Genesis GitHub](https://github.com/MandiZhao/Genesis)
- [ARCTIC 데이터셋](https://arctic.is.tue.mpg.de/)

---

## TODO

- [ ] Genesis scene.build() 이슈 해결
- [ ] DexMachina 환경 래퍼 구현 (`envs/dexmachina_env.py`)
- [ ] 데이터 수집 파이프라인 구축
- [ ] 모델 확장 (state_dim=95, action_dim=32)
- [ ] 학습 및 평가
