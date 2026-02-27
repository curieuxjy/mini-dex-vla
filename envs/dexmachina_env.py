"""
DexMachina/Genesis 환경을 mini-VLA 인터페이스로 래핑

mini-VLA 인터페이스:
- reset() -> (image, state, info)
- step(action) -> (image, state, reward, done, info)
"""

import os
import sys
import torch
import numpy as np

# DexMachina 경로 추가
DEXMACHINA_PATH = os.path.expanduser("~/Documents/dexmachina")
if DEXMACHINA_PATH not in sys.path:
    sys.path.insert(0, DEXMACHINA_PATH)

import genesis as gs
from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.constructors import get_all_env_cfg, parse_clip_string, get_common_argparser


class DexMachinaWrapper:
    """
    DexMachina/Genesis 환경을 mini-VLA 인터페이스로 래핑

    Args:
        task_name: ARCTIC 물체 이름 (box, mixer, waffleiron 등)
        hand_type: 로봇 손 타입 (allegro_hand, inspire_hand 등)
        num_envs: 병렬 환경 수
        clip_range: 프레임 범위 (예: "30-230")
        subject: 피험자 ID (예: "s01")
        device: 디바이스 (cuda:0 등)
    """

    def __init__(
        self,
        task_name: str = 'box',
        hand_type: str = 'allegro_hand',
        num_envs: int = 1,
        clip_range: str = "30-230",
        subject: str = "s01",
        retarget_name: str = "para",
        device: str = 'cuda:0',
        render: bool = True,
        image_size: tuple = (160, 160),
    ):
        self.task_name = task_name
        self.hand_type = hand_type
        self.num_envs = num_envs
        self.device = device
        self.render_enabled = render
        self.image_size = image_size

        # Genesis 초기화
        gs.init(backend=gs.gpu, logging_level='warning')

        # DexMachina 환경 설정 - argparser 기본값 사용
        clip_str = f"{task_name}-{clip_range}-{subject}-u01"

        # argparser 기본값 가져오기
        parser = get_common_argparser()
        args = parser.parse_args([])  # 빈 리스트로 기본값만 사용

        # 필요한 값 덮어쓰기
        args.clip = clip_str
        args.hand = hand_type
        args.num_envs = num_envs
        args.retarget_name = retarget_name

        # 환경 설정 가져오기
        env_kwargs = get_all_env_cfg(args, device=device)
        env_kwargs['env_cfg']['use_rl_games'] = True
        # batch_dofs_info=True 필요: num_envs >= 1일 때 set_dofs_kp에 2D tensor 전달
        env_kwargs['env_cfg']['scene_kwargs']['batch_dofs_info'] = True

        # 카메라 설정
        if render:
            env_kwargs['env_cfg']['record_video'] = True
            env_kwargs['env_cfg']['camera_kwargs'] = {
                'front': {
                    'pos': (0.5, 1.5, 1.8),
                    'lookat': (0.0, -0.15, 1.0),
                    'res': image_size,
                    'fov': 30,
                }
            }

        # 환경 생성
        self.env = BaseEnv(**env_kwargs)

        # State/Action 차원 정보
        self._compute_dimensions()

    def _compute_dimensions(self):
        """State와 Action 차원 계산"""
        self.action_dim = self.env.action_dim

        # use_rl_games=True: get_observations() returns dict with 'policy' key
        sample_obs = self.env.get_observations()
        if isinstance(sample_obs, dict) and 'policy' in sample_obs:
            # policy obs shape: (num_envs, state_dim)
            self.state_dim = sample_obs['policy'].shape[-1]
        elif isinstance(sample_obs, dict):
            total_dim = 0
            for key, val in sample_obs.items():
                if isinstance(val, torch.Tensor):
                    total_dim += val.shape[-1] if len(val.shape) > 1 else 1
            self.state_dim = total_dim
        else:
            self.state_dim = sample_obs.shape[-1]

    def reset(self):
        """
        환경 리셋

        Returns:
            image: RGB 이미지 (H, W, 3) uint8
            state: 상태 벡터 (state_dim,) float32
            info: 추가 정보 dict
        """
        obs, extras = self.env.reset()

        # State 추출
        state = self._extract_state(obs)

        # Image 렌더링
        image = self._get_image()

        info = {
            'obs_dict': obs if isinstance(obs, dict) else None,
            'extras': extras,
        }

        return image, state, info

    def step(self, action: np.ndarray):
        """
        환경 스텝

        Args:
            action: 액션 벡터 (action_dim,) float32

        Returns:
            image: RGB 이미지 (H, W, 3) uint8
            state: 상태 벡터 (state_dim,) float32
            reward: 보상 스칼라
            done: 에피소드 종료 여부
            info: 추가 정보 dict
        """
        # numpy to torch
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.device)

        # (action_dim,) -> (num_envs, action_dim)
        if action.dim() == 1:
            action = action.unsqueeze(0).expand(self.num_envs, -1)

        # Step
        obs_dict, reward, terminated, truncated, extras = self.env.step(action)

        # State 추출
        state = self._extract_state(obs_dict)

        # Image 렌더링
        image = self._get_image()

        # Done 처리
        if terminated is None:
            done = False
        elif isinstance(terminated, torch.Tensor):
            done = terminated.any().item()
        else:
            done = bool(terminated)

        # Reward 처리
        if reward is None:
            reward = 0.0
        elif isinstance(reward, torch.Tensor):
            reward = reward.mean().item()

        info = {
            'obs_dict': obs_dict if isinstance(obs_dict, dict) else None,
            'extras': extras,
            'terminated': terminated,
            'truncated': truncated,
        }

        return image, state, reward, done, info

    def _extract_state(self, obs) -> np.ndarray:
        """
        Observation에서 state 벡터 추출

        use_rl_games=True일 때 obs는 dict with 'policy' key: (num_envs, state_dim)
        첫 번째 환경의 state만 반환
        """
        if isinstance(obs, dict) and 'policy' in obs:
            # rl_games 모드: policy obs (num_envs, state_dim) -> 첫 번째 env
            state = obs['policy'][0].cpu().numpy()
        elif isinstance(obs, dict):
            state_list = []
            for key, val in obs.items():
                if isinstance(val, torch.Tensor):
                    state_list.append(val[0].cpu().numpy().flatten() if val.dim() > 1 else val.cpu().numpy().flatten())
                elif isinstance(val, np.ndarray):
                    state_list.append(val.flatten())
            state = np.concatenate(state_list)
        elif isinstance(obs, torch.Tensor):
            state = obs[0].cpu().numpy() if obs.dim() > 1 else obs.cpu().numpy()
        else:
            state = np.array(obs).flatten()

        return state.astype(np.float32)

    def _get_image(self) -> np.ndarray:
        """
        카메라에서 RGB 이미지 렌더링

        Returns:
            image: (H, W, 3) uint8 RGB
        """
        if not self.render_enabled or not hasattr(self.env, 'cameras'):
            # 더미 이미지 반환
            return np.zeros((*self.image_size, 3), dtype=np.uint8)

        try:
            camera = self.env.cameras.get('front', None)
            if camera is None:
                return np.zeros((*self.image_size, 3), dtype=np.uint8)

            frame, depth, seg, normal = camera.render(segmentation=False)

            # RGB로 변환 (Genesis는 RGB 반환)
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            return frame

        except Exception as e:
            print(f"Render error: {e}")
            return np.zeros((*self.image_size, 3), dtype=np.uint8)

    def get_expert_action(self) -> np.ndarray:
        """
        Expert action 가져오기 (retargeted demonstration에서)

        Returns:
            action: (action_dim,) float32
        """
        # DexMachina의 demonstration 데이터 사용
        # 현재 프레임의 목표 joint position을 action으로 사용

        # TODO: 구현 필요 - retarget data에서 action 추출
        return np.zeros(self.action_dim, dtype=np.float32)

    @property
    def observation_space(self):
        """Gymnasium 호환 observation space"""
        import gymnasium as gym
        return gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, (*self.image_size, 3), dtype=np.uint8),
            'state': gym.spaces.Box(-np.inf, np.inf, (self.state_dim,), dtype=np.float32),
        })

    @property
    def action_space(self):
        """Gymnasium 호환 action space"""
        import gymnasium as gym
        return gym.spaces.Box(-1, 1, (self.action_dim,), dtype=np.float32)

    def close(self):
        """환경 종료"""
        if hasattr(self.env, 'scene'):
            # Genesis scene cleanup
            pass


def main():
    """래퍼 테스트"""
    print("DexMachina 래퍼 테스트 시작...")

    env = DexMachinaWrapper(
        task_name='box',
        hand_type='allegro_hand',
        num_envs=1,
        clip_range="30-100",
        render=False,  # headless 테스트
    )

    print(f"State dim: {env.state_dim}")
    print(f"Action dim: {env.action_dim}")

    # Reset 테스트
    image, state, info = env.reset()
    print(f"Image shape: {image.shape}")
    print(f"State shape: {state.shape}")

    # Step 테스트
    action = np.zeros(env.action_dim, dtype=np.float32)
    for i in range(10):
        image, state, reward, done, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}, done={done}")

    print("테스트 완료!")
    env.close()


if __name__ == "__main__":
    main()
