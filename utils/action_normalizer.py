"""
Allegro Hand Joint Limit 기반 Action Normalization

DexMachina hybrid action mode (44 dim = 22 DOF x 2 hands):
  - Per hand: wrist (6 DOF) + fingers (16 DOF)
  - Wrist: [tx, ty, tz, roll, pitch, yaw]
  - Fingers: 4 fingers x 4 joints each

Hybrid mode 동작 (robot.py translate_actions):
  - Wrist: residual control  -> target = current + scale * action
  - Finger: absolute control -> target = lower + (upper - lower) * (action + 1) / 2

따라서 RL policy 출력 action은 [-1, 1] 범위이지만,
데이터 수집 시 kinematic mode에서는 raw joint positions (qpos_targets)을 저장.

이 모듈은 raw joint position <-> [-1, 1] 정규화를 수행.
"""

import numpy as np


# Allegro Hand Joint Limits (from URDF: allegro_hand_left/right_6dof.urdf)
# 양손 동일한 limits 사용

# Wrist joints (6 DOF)
WRIST_LIMITS = np.array([
    [-5.0, 5.0],     # tx (prismatic)
    [-5.0, 5.0],     # ty (prismatic)
    [-5.0, 5.0],     # tz (prismatic)
    [-6.2, 6.2],     # roll (revolute)
    [-6.2, 6.2],     # pitch (revolute)
    [-6.2, 6.2],     # yaw (revolute)
], dtype=np.float32)

# Finger joints (16 DOF = 4 fingers x 4 joints)
FINGER_LIMITS = np.array([
    # Finger 1 (Thumb)
    [-0.47, 0.47],    # joint_0.0
    [-0.196, 1.61],   # joint_1.0
    [-0.174, 1.709],  # joint_2.0
    [-0.227, 1.618],  # joint_3.0
    # Finger 2 (Index)
    [-0.47, 0.47],    # joint_4.0
    [-0.196, 1.61],   # joint_5.0
    [-0.174, 1.709],  # joint_6.0
    [-0.227, 1.618],  # joint_7.0
    # Finger 3 (Middle)
    [-0.47, 0.47],    # joint_8.0
    [-0.196, 1.61],   # joint_9.0
    [-0.174, 1.709],  # joint_10.0
    [-0.227, 1.618],  # joint_11.0
    # Finger 4 (Pinky)
    [0.263, 1.396],   # joint_12.0
    [-0.105, 1.163],  # joint_13.0
    [-0.189, 1.644],  # joint_14.0
    [-0.162, 1.719],  # joint_15.0
], dtype=np.float32)

# Per-hand limits: wrist (6) + fingers (16) = 22 DOF
HAND_LIMITS = np.concatenate([WRIST_LIMITS, FINGER_LIMITS], axis=0)  # (22, 2)

# Bimanual limits: left hand (22) + right hand (22) = 44 DOF
BIMANUAL_LIMITS = np.concatenate([HAND_LIMITS, HAND_LIMITS], axis=0)  # (44, 2)


class AllegroActionNormalizer:
    """
    Allegro Hand joint limit 기반 action normalizer

    raw joint position -> [-1, 1]: normalize()
    [-1, 1] -> raw joint position: denormalize()

    공식:
        normalized = 2 * (raw - lower) / (upper - lower) - 1
        raw = lower + (upper - lower) * (normalized + 1) / 2
    """

    def __init__(self, action_dim: int = 44, clip: bool = True):
        """
        Args:
            action_dim: Action dimension (44 for bimanual Allegro)
            clip: Whether to clip denormalized actions to joint limits
        """
        if action_dim == 44:
            self.limits = BIMANUAL_LIMITS.copy()
        elif action_dim == 22:
            self.limits = HAND_LIMITS.copy()
        else:
            raise ValueError(f"Unsupported action_dim={action_dim}. Expected 22 or 44.")

        self.lower = self.limits[:, 0]
        self.upper = self.limits[:, 1]
        self.range = self.upper - self.lower
        self.clip = clip

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        """
        Raw joint positions -> [-1, 1]

        Args:
            actions: (..., action_dim) raw joint positions
        Returns:
            normalized: (..., action_dim) in [-1, 1]
        """
        normalized = 2.0 * (actions - self.lower) / self.range - 1.0
        if self.clip:
            normalized = np.clip(normalized, -1.0, 1.0)
        return normalized.astype(np.float32)

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        """
        [-1, 1] -> raw joint positions

        Args:
            actions: (..., action_dim) normalized actions
        Returns:
            raw: (..., action_dim) raw joint positions
        """
        raw = self.lower + self.range * (actions + 1.0) / 2.0
        if self.clip:
            raw = np.clip(raw, self.lower, self.upper)
        return raw.astype(np.float32)

    def get_limits(self) -> dict:
        """Return joint limits as dict for checkpoint storage"""
        return {
            "lower": self.lower.copy(),
            "upper": self.upper.copy(),
        }

    @staticmethod
    def from_limits(limits_dict: dict, clip: bool = True) -> "AllegroActionNormalizer":
        """Restore normalizer from saved limits dict"""
        normalizer = AllegroActionNormalizer.__new__(AllegroActionNormalizer)
        normalizer.lower = np.array(limits_dict["lower"], dtype=np.float32)
        normalizer.upper = np.array(limits_dict["upper"], dtype=np.float32)
        normalizer.range = normalizer.upper - normalizer.lower
        normalizer.limits = np.stack([normalizer.lower, normalizer.upper], axis=1)
        normalizer.clip = clip
        return normalizer
