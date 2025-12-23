"""
DexMachina 환경에서 demonstration 데이터 수집

ARCTIC human demonstration의 retargeted 데이터를 사용하여
mini-VLA 학습용 데이터셋 생성

Usage:
    python -m scripts.collect_dexmachina_data \
        --task-name box \
        --clip-range 30-230 \
        --episodes 10 \
        --output-path data/dexmachina_box.npz
"""

import os
import sys
import argparse
import numpy as np
import torch

# DexMachina 경로 추가
DEXMACHINA_PATH = os.path.expanduser("~/Documents/dexmachina")
if DEXMACHINA_PATH not in sys.path:
    sys.path.insert(0, DEXMACHINA_PATH)

import genesis as gs
from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.constructors import get_all_env_cfg, parse_clip_string, get_common_argparser
from dexmachina.envs.demo_data import load_genesis_retarget_data

# mini-VLA 경로
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tokenizer import SimpleTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Collect DexMachina demonstration data")
    parser.add_argument("--task-name", type=str, default="box",
                        choices=["box", "mixer", "waffleiron"],
                        help="ARCTIC object name")
    parser.add_argument("--hand-type", type=str, default="allegro_hand",
                        help="Robot hand type")
    parser.add_argument("--clip-range", type=str, default="30-230",
                        help="Frame range (start-end)")
    parser.add_argument("--subject", type=str, default="s01",
                        help="ARCTIC subject ID")
    parser.add_argument("--retarget-name", type=str, default="para",
                        help="Retarget data name")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel environments")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to collect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=str, default="data/dexmachina_box.npz",
                        help="Output path for dataset")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Task instruction (auto-generated if not provided)")
    parser.add_argument("--render", action="store_true",
                        help="Enable rendering for image collection")
    parser.add_argument("--image-size", type=int, nargs=2, default=[160, 160],
                        help="Image size (H, W)")
    parser.add_argument("--action-mode", type=str, default="kinematic",
                        choices=["kinematic", "residual"],
                        help="Action mode: kinematic (replay demo) or residual")
    return parser.parse_args()


def get_task_instruction(task_name: str) -> str:
    """태스크별 기본 instruction 생성"""
    instructions = {
        "box": "open and close the box with both hands",
        "mixer": "operate the mixer with both hands",
        "waffleiron": "open and close the waffle iron",
        "laptop": "open and close the laptop",
        "scissors": "use the scissors with both hands",
    }
    return instructions.get(task_name, f"manipulate the {task_name} with both hands")


def create_env(args, device="cuda:0"):
    """DexMachina 환경 생성"""
    # Genesis 초기화
    gs.init(backend=gs.gpu, logging_level='warning')

    # clip 문자열 생성
    clip_str = f"{args.task_name}-{args.clip_range}-{args.subject}-u01"

    # argparser 기본값 가져오기
    dex_parser = get_common_argparser()
    dex_args = dex_parser.parse_args([])

    # 필요한 값 덮어쓰기
    dex_args.clip = clip_str
    dex_args.hand = args.hand_type
    dex_args.num_envs = args.num_envs
    dex_args.retarget_name = args.retarget_name
    dex_args.action_mode = args.action_mode
    dex_args.record_video = args.render

    # 환경 설정 가져오기
    env_kwargs = get_all_env_cfg(dex_args, device=device)
    env_kwargs['env_cfg']['use_rl_games'] = False

    # 카메라 설정
    if args.render:
        env_kwargs['env_cfg']['camera_kwargs'] = {
            'front': {
                'pos': (0.5, 1.5, 1.8),
                'lookat': (0.0, -0.15, 1.0),
                'res': tuple(args.image_size),
                'fov': 30,
            }
        }

    # 환경 생성
    env = BaseEnv(**env_kwargs)

    return env, env_kwargs


def to_numpy(val):
    """텐서를 numpy로 변환"""
    if isinstance(val, torch.Tensor):
        return val.cpu().numpy()
    return val


def get_expert_actions(env, retarget_data, frame_idx: int):
    """
    Retargeted demonstration에서 expert action 추출

    kinematic mode에서는 demonstration의 joint target을 action으로 사용
    """
    actions = []

    for side in ['left', 'right']:
        side_data = retarget_data[side]

        # qpos_targets 사용 (절대 위치) - 새로운 키 이름
        if 'qpos_targets' in side_data and side_data['qpos_targets'] is not None:
            qpos_targets = side_data['qpos_targets']
            # frame_idx에 해당하는 joint target 추출
            frame_actions = []
            for joint_name, values in qpos_targets.items():
                values_np = to_numpy(values)
                if frame_idx < len(values_np):
                    frame_actions.append(float(values_np[frame_idx]))
                else:
                    frame_actions.append(float(values_np[-1]))  # 마지막 값 사용
            actions.extend(frame_actions)
        elif 'residual_qpos' in side_data:
            # residual_qpos 사용 (fallback)
            residual_qpos = side_data['residual_qpos']
            frame_actions = []
            for joint_name, values in residual_qpos.items():
                values_np = to_numpy(values)
                if frame_idx < len(values_np):
                    frame_actions.append(float(values_np[frame_idx]))
                else:
                    frame_actions.append(float(values_np[-1]))
            actions.extend(frame_actions)
        else:
            print(f"Warning: No qpos data found for {side}")

    return np.array(actions, dtype=np.float32)


def extract_state(env) -> np.ndarray:
    """환경에서 state 벡터 추출"""
    obs = env.get_observations()

    if isinstance(obs, dict):
        state_list = []
        for key, val in obs.items():
            if isinstance(val, torch.Tensor):
                state_list.append(val.cpu().numpy().flatten())
            elif isinstance(val, np.ndarray):
                state_list.append(val.flatten())
        state = np.concatenate(state_list)
    elif isinstance(obs, torch.Tensor):
        state = obs[0].cpu().numpy() if obs.dim() > 1 else obs.cpu().numpy()
    else:
        state = np.array(obs).flatten()

    return state.astype(np.float32)


def render_image(env, image_size=(160, 160)) -> np.ndarray:
    """카메라에서 이미지 렌더링"""
    if not hasattr(env, 'cameras') or env.cameras is None:
        return np.zeros((*image_size, 3), dtype=np.uint8)

    try:
        camera = env.cameras.get('front', None)
        if camera is None:
            return np.zeros((*image_size, 3), dtype=np.uint8)

        frame, depth, seg, normal = camera.render(segmentation=False)

        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        return frame

    except Exception as e:
        print(f"Render error: {e}")
        return np.zeros((*image_size, 3), dtype=np.uint8)


def collect_episode_kinematic(env, retarget_data, demo_data, episode_length: int,
                              render: bool, image_size: tuple) -> dict:
    """
    Kinematic replay로 에피소드 데이터 수집

    demonstration의 joint target을 순차적으로 재생하면서 데이터 수집
    """
    images = []
    states = []
    actions = []

    # 환경 리셋
    env.reset()

    device = env.device

    for frame_idx in range(episode_length):
        # 현재 state 수집
        state = extract_state(env)
        states.append(state.copy())

        # 이미지 수집
        if render:
            image = render_image(env, image_size)
            images.append(image.copy())
        else:
            images.append(np.zeros((*image_size, 3), dtype=np.uint8))

        # Expert action (demonstration에서)
        action = get_expert_actions(env, retarget_data, frame_idx)
        actions.append(action.copy())

        # 환경 스텝 (action은 환경의 action_dim에 맞춰야 함)
        # kinematic mode에서는 action이 joint target으로 직접 사용됨
        action_tensor = torch.from_numpy(action).float().to(device)
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)

        # 액션 차원이 맞지 않으면 패딩 또는 잘라내기
        env_action_dim = env.action_dim
        if action_tensor.shape[1] < env_action_dim:
            padding = torch.zeros(action_tensor.shape[0], env_action_dim - action_tensor.shape[1], device=device)
            action_tensor = torch.cat([action_tensor, padding], dim=1)
        elif action_tensor.shape[1] > env_action_dim:
            action_tensor = action_tensor[:, :env_action_dim]

        try:
            env.step(action_tensor)
        except Exception as e:
            print(f"Step error at frame {frame_idx}: {e}")
            break

    return {
        'images': images,
        'states': states,
        'actions': actions,
    }


def main():
    args = parse_args()

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

    # Instruction 설정
    if args.instruction is None:
        instruction = get_task_instruction(args.task_name)
    else:
        instruction = args.instruction

    print(f"Task: {args.task_name}")
    print(f"Instruction: {instruction}")
    print(f"Clip range: {args.clip_range}")
    print(f"Hand type: {args.hand_type}")
    print(f"Episodes: {args.episodes}")
    print(f"Render: {args.render}")

    # 환경 생성
    print("\nCreating environment...")
    env, env_kwargs = create_env(args, device="cuda:0")

    # Retarget 데이터 로드
    print("Loading retarget data...")
    start, end = map(int, args.clip_range.split('-'))
    demo_data, retarget_data = load_genesis_retarget_data(
        obj_name=args.task_name,
        hand_name=args.hand_type,
        frame_start=start,
        frame_end=end,
        save_name=args.retarget_name,
        use_clip="01",
        subject_name=args.subject,
    )

    episode_length = end - start
    print(f"Episode length: {episode_length} frames")

    # 데이터 수집
    all_images = []
    all_states = []
    all_actions = []
    all_texts = []

    for ep in range(args.episodes):
        print(f"\nCollecting episode {ep+1}/{args.episodes}...")

        episode_data = collect_episode_kinematic(
            env=env,
            retarget_data=retarget_data,
            demo_data=demo_data,
            episode_length=episode_length,
            render=args.render,
            image_size=tuple(args.image_size),
        )

        all_images.extend(episode_data['images'])
        all_states.extend(episode_data['states'])
        all_actions.extend(episode_data['actions'])
        all_texts.extend([instruction] * len(episode_data['states']))

        print(f"  Collected {len(episode_data['states'])} frames")

    # 배열로 변환
    print("\nStacking arrays...")
    images = np.stack(all_images, axis=0)  # (N, H, W, 3)
    states = np.stack(all_states, axis=0)  # (N, state_dim)
    actions = np.stack(all_actions, axis=0)  # (N, action_dim)

    # Tokenize instructions
    print("Tokenizing instructions...")
    tokenizer = SimpleTokenizer(vocab=None)
    tokenizer.build_from_texts(all_texts)
    text_ids_list = [tokenizer.encode(t) for t in all_texts]
    max_len = max(len(seq) for seq in text_ids_list)
    text_ids = np.zeros((len(all_texts), max_len), dtype=np.int64)
    for i, seq in enumerate(text_ids_list):
        text_ids[i, :len(seq)] = np.array(seq, dtype=np.int64)

    # 추가 메타데이터
    metadata = {
        'task_name': args.task_name,
        'hand_type': args.hand_type,
        'clip_range': args.clip_range,
        'subject': args.subject,
        'instruction': instruction,
        'state_dim': states.shape[1],
        'action_dim': actions.shape[1],
        'num_episodes': args.episodes,
        'episode_length': episode_length,
    }

    # 저장
    print(f"\nSaving dataset to {args.output_path}...")
    np.savez_compressed(
        args.output_path,
        images=images,
        states=states,
        actions=actions,
        text_ids=text_ids,
        vocab=tokenizer.vocab,
        metadata=metadata,
    )

    print("\n" + "="*50)
    print("Dataset saved successfully!")
    print("="*50)
    print(f"  images:   {images.shape} ({images.dtype})")
    print(f"  states:   {states.shape} ({states.dtype})")
    print(f"  actions:  {actions.shape} ({actions.dtype})")
    print(f"  text_ids: {text_ids.shape} ({text_ids.dtype})")
    print(f"  vocab size: {len(tokenizer.vocab)}")
    print(f"  total frames: {len(states)}")


if __name__ == "__main__":
    main()
