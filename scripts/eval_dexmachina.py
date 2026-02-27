"""
Evaluate trained VLA model on DexMachina environment

학습된 VLA 모델을 DexMachina 환경에서 평가

Usage:
    python -m scripts.eval_dexmachina \
        --checkpoint checkpoints/dexmachina_model.pt \
        --task-name box \
        --episodes 5 \
        --save-video
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
from dexmachina.envs.constructors import get_all_env_cfg, get_common_argparser
from dexmachina.envs.demo_data import load_genesis_retarget_data

# mini-VLA 경로
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vla_dexmachina import VLADexMachinaPolicy, create_dexmachina_model
from utils.tokenizer import SimpleTokenizer
from utils.action_normalizer import AllegroActionNormalizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DexMachina VLA model")

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint")

    # Environment
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

    # Evaluation
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps per episode (default: clip length)")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Task instruction (auto if not specified)")

    # Output
    parser.add_argument("--save-video", action="store_true",
                        help="Save evaluation videos")
    parser.add_argument("--video-dir", type=str, default="videos",
                        help="Directory to save videos")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42)

    # Compare with expert
    parser.add_argument("--compare-expert", action="store_true",
                        help="Compare with expert demonstration actions")

    return parser.parse_args()


def get_task_instruction(task_name: str) -> str:
    """태스크별 기본 instruction"""
    instructions = {
        "box": "open and close the box with both hands",
        "mixer": "operate the mixer with both hands",
        "waffleiron": "open and close the waffle iron",
    }
    return instructions.get(task_name, f"manipulate the {task_name} with both hands")


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    """체크포인트에서 모델과 토크나이저 로드"""
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    vocab = ckpt["vocab"]
    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    model_size = ckpt.get("model_size", "base")
    action_stats = ckpt.get("action_stats", None)

    vocab_size = max(vocab.values()) + 1

    # 모델 생성
    model = create_dexmachina_model(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        model_size=model_size,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 토크나이저
    tokenizer = SimpleTokenizer(vocab=vocab)

    print(f"  Model size: {model_size}")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Parameters: {model.get_num_params():,}")

    return model, tokenizer, action_stats


def create_env(args, device="cuda:0"):
    """DexMachina 환경 생성"""
    gs.init(backend=gs.gpu, logging_level='warning')

    clip_str = f"{args.task_name}-{args.clip_range}-{args.subject}-u01"

    dex_parser = get_common_argparser()
    dex_args = dex_parser.parse_args([])

    dex_args.clip = clip_str
    dex_args.hand = args.hand_type
    dex_args.num_envs = 1
    dex_args.retarget_name = args.retarget_name
    dex_args.action_mode = "kinematic"
    dex_args.record_video = not args.no_render

    env_kwargs = get_all_env_cfg(dex_args, device=device)
    env_kwargs['env_cfg']['use_rl_games'] = False

    env = BaseEnv(**env_kwargs)
    return env, env_kwargs


def to_numpy(val):
    """텐서를 numpy로 변환"""
    if isinstance(val, torch.Tensor):
        return val.cpu().numpy()
    return val


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
    # DexMachina의 내장 렌더링 사용
    if hasattr(env, 'render_frame'):
        try:
            frame = env.render_frame()
            if frame is not None:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                return frame
        except Exception as e:
            pass

    # cameras dict에서 렌더링 시도
    if hasattr(env, 'cameras') and env.cameras is not None:
        try:
            camera = env.cameras.get('front', None)
            if camera is None and len(env.cameras) > 0:
                camera = list(env.cameras.values())[0]

            if camera is not None:
                frame, depth, seg, normal = camera.render(segmentation=False)
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                return frame
        except Exception as e:
            pass

    # scene에서 직접 렌더링 시도
    if hasattr(env, 'scene') and hasattr(env.scene, 'render'):
        try:
            frame = env.scene.render()
            if frame is not None:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                return frame
        except Exception as e:
            pass

    return np.zeros((*image_size, 3), dtype=np.uint8)


def get_expert_actions(retarget_data, frame_idx: int, action_dim: int) -> np.ndarray:
    """Expert demonstration에서 action 추출"""
    actions = []

    for side in ['left', 'right']:
        side_data = retarget_data[side]

        if 'qpos_targets' in side_data and side_data['qpos_targets'] is not None:
            qpos_targets = side_data['qpos_targets']
            frame_actions = []
            for joint_name, values in qpos_targets.items():
                values_np = to_numpy(values)
                if frame_idx < len(values_np):
                    frame_actions.append(float(values_np[frame_idx]))
                else:
                    frame_actions.append(float(values_np[-1]))
            actions.extend(frame_actions)
        elif 'residual_qpos' in side_data:
            residual_qpos = side_data['residual_qpos']
            frame_actions = []
            for joint_name, values in residual_qpos.items():
                values_np = to_numpy(values)
                if frame_idx < len(values_np):
                    frame_actions.append(float(values_np[frame_idx]))
                else:
                    frame_actions.append(float(values_np[-1]))
            actions.extend(frame_actions)

    action = np.array(actions, dtype=np.float32)

    # 패딩/트리밍
    if len(action) < action_dim:
        action = np.pad(action, (0, action_dim - len(action)))
    elif len(action) > action_dim:
        action = action[:action_dim]

    return action


def denormalize_action(action: np.ndarray, action_stats: dict) -> np.ndarray:
    """정규화된 action을 원래 스케일로 복원"""
    if action_stats is None:
        return action

    norm_mode = action_stats.get("norm_mode", "statistical")

    if norm_mode == "joint_limits" and "joint_limits" in action_stats:
        normalizer = AllegroActionNormalizer.from_limits(action_stats["joint_limits"])
        return normalizer.denormalize(action)
    else:
        # statistical fallback
        mean = action_stats["mean"]
        std = action_stats["std"]
        return action * std + mean


def main():
    args = parse_args()

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, tokenizer, action_stats = load_model_and_tokenizer(args.checkpoint, device)

    # Instruction
    if args.instruction is None:
        instruction = get_task_instruction(args.task_name)
    else:
        instruction = args.instruction

    print(f"\nInstruction: {instruction}")

    # Tokenize instruction
    instr_tokens = tokenizer.encode(instruction)
    text_ids = torch.tensor(instr_tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Create environment
    print("\nCreating environment...")
    env, env_kwargs = create_env(args, device=str(device))

    # Load expert data for comparison
    retarget_data = None
    if args.compare_expert:
        print("Loading expert demonstration...")
        start, end = map(int, args.clip_range.split('-'))
        _, retarget_data = load_genesis_retarget_data(
            obj_name=args.task_name,
            hand_name=args.hand_type,
            frame_start=start,
            frame_end=end,
            save_name=args.retarget_name,
            use_clip="01",
            subject_name=args.subject,
        )

    # Episode length
    start, end = map(int, args.clip_range.split('-'))
    episode_length = args.max_steps if args.max_steps else (end - start)

    # Video writer
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio

    # Evaluation metrics
    all_rewards = []
    all_action_errors = []

    print(f"\nStarting evaluation...")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps: {episode_length}")

    for ep in range(args.episodes):
        print(f"\nEpisode {ep+1}/{args.episodes}")

        # Reset
        env.reset()

        frames = []
        ep_reward = 0.0
        ep_action_errors = []

        for step in range(episode_length):
            # Get current state
            state = extract_state(env)

            # Render image
            if not args.no_render:
                image = render_image(env)
                frames.append(image.copy())
            else:
                image = np.zeros((160, 160, 3), dtype=np.uint8)

            # Prepare inputs (copy to handle negative strides)
            img_t = torch.from_numpy(image.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            state_t = torch.from_numpy(state).float().unsqueeze(0)

            img_t = img_t.to(device)
            state_t = state_t.to(device)

            # Model inference
            with torch.no_grad():
                action_t = model.act(img_t, text_ids, state_t)

            action_np = action_t.squeeze(0).cpu().numpy()

            # Denormalize action
            action_np = denormalize_action(action_np, action_stats)

            # Compare with expert
            if args.compare_expert and retarget_data is not None:
                expert_action = get_expert_actions(retarget_data, step, len(action_np))
                action_error = np.mean((action_np - expert_action) ** 2)
                ep_action_errors.append(action_error)

            # Step environment
            action_tensor = torch.from_numpy(action_np).float().to(device).unsqueeze(0)

            try:
                obs, reward, terminated, truncated, extras = env.step(action_tensor)

                if reward is not None:
                    if isinstance(reward, torch.Tensor):
                        ep_reward += reward.mean().item()
                    else:
                        ep_reward += float(reward)

            except Exception as e:
                print(f"  Step error at frame {step}: {e}")
                break

        # Episode stats
        all_rewards.append(ep_reward)
        if ep_action_errors:
            mean_error = np.mean(ep_action_errors)
            all_action_errors.append(mean_error)
            print(f"  Reward: {ep_reward:.4f}, Action MSE: {mean_error:.6f}")
        else:
            print(f"  Reward: {ep_reward:.4f}")

        # Save video
        if args.save_video and frames:
            video_path = os.path.join(
                args.video_dir,
                f"dexmachina_{args.task_name}_ep{ep+1:02d}.mp4"
            )
            with imageio.get_writer(video_path, fps=30) as writer:
                for f in frames:
                    writer.append_data(f)
            print(f"  Saved video: {video_path}")

    # Summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"  Task: {args.task_name}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Mean Reward: {np.mean(all_rewards):.4f} +/- {np.std(all_rewards):.4f}")

    if all_action_errors:
        print(f"  Mean Action MSE: {np.mean(all_action_errors):.6f} +/- {np.std(all_action_errors):.6f}")

    print("=" * 50)


if __name__ == "__main__":
    main()
