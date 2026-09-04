# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================
"""Keyboard controller for SE(3) control with custom key bindings for mobile robot."""

from __future__ import annotations

import numpy as np
import torch
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

import carb
import omni

try:
    from isaaclab.devices.device_base import DeviceBase, DeviceCfg
except ImportError:
    from isaaclab.devices import DeviceBase, DeviceCfg


class Se3KeyboardMobile(DeviceBase):
    """A keyboard controller for sending SE(3) commands for mobile robot control.

    This class is designed to provide a keyboard controller for a mobile robot.
    It uses the Omniverse keyboard interface to listen to keyboard events and map them to robot's
    task-space commands.

    The command comprises of:
    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * For mobile robot: x=forward/backward, z=height, yaw=rotation

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Move forward/backward (x-axis) W                 S
        Rotate left/right (yaw)        A                 D
        Move up/down (height, z-axis) Z                 X
        ============================== ================= =================

    .. seealso::
        The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.
    """

    def __init__(self, cfg: Se3KeyboardMobileCfg):
        """Initialize the keyboard layer.

        Args:
            cfg: Configuration object for keyboard settings.
        """
        # store inputs
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.height_sensitivity = cfg.height_sensitivity
        self._sim_device = cfg.sim_device

        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        # bindings for keyboard to command
        self._create_key_bindings()

        # command buffers
        # 非增量式控制：ws(速度)和ad(转向) - 按下时设置值，松开时归零
        self._velocity_cmd = np.zeros(2)  # [v_x, omega_z] - 速度命令
        # 增量式控制：zx(高度) - 按下时累加，松开时保持
        self._height_delta = 0.0  # 高度增量
        self._current_height = 0.0  # 当前高度值（累积值）
        self._default_height = 0.0  # 默认高度值
        self._height_range = [0.15, 0.55]  # 高度范围限制

        # 按键状态跟踪（用于非增量式控制）
        self._pressed_keys = set()

        # dictionary for additional callbacks
        self._additional_callbacks = dict()
        
        # 环境引用（用于设置命令）
        self._env = None
        # 手动跳跃开关（默认启用）
        self._enable_manual_jump = getattr(cfg, "enable_manual_jump", True)

    def __del__(self):
        """Release the keyboard interface."""
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_keyboard_sub"):
            if self._keyboard_sub is not None:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
                self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of keyboard controller."""
        msg = f"Keyboard Controller for SE(3) Mobile Robot: {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tReset all commands: L\n"
        msg += "\tMove forward/backward (x-axis): W/S\n"
        msg += "\tRotate left/right (yaw): A/D\n"
        msg += "\tMove up/down (height, z-axis): Z/X\n"
        msg += "\tTrigger jump window (JUMP onehot): Q"
        return msg

    """
    Operations
    """

    def reset(self):
        """Reset all commands to zero."""
        self._velocity_cmd = np.zeros(2)  # [v_x, omega_z]
        self._height_delta = 0.0
        self._current_height = self._default_height
        self._pressed_keys.clear()
        # 重置环境命令
        self._apply_commands_to_env()

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """Provides the result from keyboard event state.

        Returns:
            torch.Tensor: A 6-element tensor containing delta pose:
                [x, y, z, rx, ry, rz] in meters and radians.
                For mobile robot: x=forward, y=0, z=height, yaw=rotation
        """
        # 为了兼容性，返回6元素向量
        # x=速度, y=0, z=高度, yaw=角速度
        delta_pos = np.array([self._velocity_cmd[0], 0.0, self._current_height])
        delta_rot = np.array([0.0, 0.0, self._velocity_cmd[1]])
        rot_vec = Rotation.from_euler("XYZ", delta_rot).as_rotvec()
        command = np.concatenate([delta_pos, rot_vec])
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    def get_velocity_command(self) -> torch.Tensor:
        """Get velocity command for mobile robot (v_x, v_y, omega_z).

        Returns:
            torch.Tensor: A 3-element tensor [v_x, v_y, omega_z]
        """
        # 非增量式控制：直接返回当前速度命令
        v_x = self._velocity_cmd[0]
        v_y = 0.0  # mobile robot doesn't move sideways
        omega_z = self._velocity_cmd[1]
        return torch.tensor([v_x, v_y, omega_z], dtype=torch.float32, device=self._sim_device)

    def get_height_command(self) -> float:
        """Get height command increment (z-axis).

        Returns:
            float: Height command increment value (z-axis)
        """
        # 返回高度增量（用于增量式控制）
        return float(self._height_delta)
    
    def get_current_height(self) -> float:
        """Get current accumulated height value.

        Returns:
            float: Current height value
        """
        return float(self._current_height)
    
    def set_current_height(self, height: float):
        """Set current height value (used when resetting).

        Args:
            height: Height value to set
        """
        self._current_height = float(height)
    
    def set_default_height(self, height: float):
        """Set default height value.

        Args:
            height: Default height value
        """
        self._default_height = float(height)
        if self._current_height == 0.0:
            self._current_height = self._default_height
    
    def set_height_range(self, height_range: list | tuple):
        """Set height range limits.

        Args:
            height_range: [min_height, max_height]
        """
        self._height_range = [float(height_range[0]), float(height_range[1])]
    
    def set_env(self, env):
        """Set environment reference for applying commands.

        Args:
            env: Environment object
        """
        self._env = env
        # 默认绑定 Q 触发手动跳跃（如果环境支持）
        if self._enable_manual_jump:
            self.add_callback("Q", self._trigger_manual_jump)
    
    def advance_step(self, dt: float) -> dict:
        """Advance one step and apply commands to environment.

        Args:
            dt: Time step

        Returns:
            dict: Command information with keys: v_x, v_y, omega_z, height
        """
        # 处理高度增量式控制
        if abs(self._height_delta) > 0.001:
            self._current_height += self._height_delta * dt
            self._current_height = max(self._height_range[0], min(self._height_range[1], self._current_height))
        
        # 应用命令到环境
        self._apply_commands_to_env()
        
        # 返回命令信息
        vel_cmd = self.get_velocity_command()
        return {
            "v_x": float(vel_cmd[0].item()),
            "v_y": float(vel_cmd[1].item()),
            "omega_z": float(vel_cmd[2].item()),
            "height": self._current_height,
            "has_input": abs(vel_cmd[0].item()) > 0.01 or abs(vel_cmd[2].item()) > 0.01 or abs(self._height_delta) > 0.001
        }
    
    def _apply_commands_to_env(self):
        """Apply current commands to environment."""
        if self._env is None:
            return
        
        env_unwrapped = self._env.unwrapped
        vel_cmd = self.get_velocity_command()
        v_x = float(vel_cmd[0].item())
        v_y = float(vel_cmd[1].item())
        omega_z = float(vel_cmd[2].item())
        
        # 检测是 manager 模式还是 direct 模式
        is_manager_mode = hasattr(env_unwrapped, "command_manager")
        
        # 设置速度命令到环境
        if is_manager_mode:
            # Manager 模式：通过 command_manager 设置
            if hasattr(env_unwrapped.command_manager, "_terms"):
                # 查找速度命令 term，通常名为 "base_velocity"
                for term_name, term in env_unwrapped.command_manager._terms.items():
                    if hasattr(term, "vel_command_b"):
                        # UniformVelocityCommand term
                        term.vel_command_b[:, 0].fill_(v_x)
                        term.vel_command_b[:, 1].fill_(v_y)
                        term.vel_command_b[:, 2].fill_(omega_z)
                        break
        else:
            # Direct 模式：直接设置 command 属性
            if hasattr(env_unwrapped, "command_generator") and hasattr(env_unwrapped.command_generator, "command"):
                env_unwrapped.command_generator.command[:, 0].fill_(v_x)
                env_unwrapped.command_generator.command[:, 1].fill_(v_y)
                env_unwrapped.command_generator.command[:, 2].fill_(omega_z)
            elif hasattr(env_unwrapped, "command"):
                env_unwrapped.command[:, 0].fill_(v_x)
                env_unwrapped.command[:, 1].fill_(v_y)
                env_unwrapped.command[:, 2].fill_(omega_z)
        
        # 设置高度命令到环境
        if is_manager_mode:
            # Manager 模式：通过 command_manager 设置
            if hasattr(env_unwrapped.command_manager, "_terms"):
                # 查找高度命令 term，通常名为 "height"
                for term_name, term in env_unwrapped.command_manager._terms.items():
                    if hasattr(term, "height_command"):
                        # UniformHeightCommand term
                        term.height_command[:] = self._current_height
                        break
        else:
            # Direct 模式：直接设置 height_cmd 属性
            if hasattr(env_unwrapped, "height_cmd"):
                env_unwrapped.height_cmd[:] = self._current_height

    def _trigger_manual_jump(self):
        """触发一次手动跳跃窗口，将 jump onehot 置 JUMP 指定步数。"""
        if self._env is None:
            print("[WARNING] 未绑定环境，无法触发手动跳跃")
            return
        env_unwrapped = self._env.unwrapped
        if hasattr(env_unwrapped, "trigger_manual_jump"):
            jump_h = getattr(env_unwrapped, "manual_jump_height", None)
            manual_window = getattr(env_unwrapped, "manual_jump_window", None)
            ok, steps = env_unwrapped.trigger_manual_jump(
                duration_steps=manual_window, jump_height=jump_h
            )
            if ok:
                print(f"[INFO] 手动跳跃触发：JUMP onehot 保持 {steps} 步（按 Q）")
            else:
                print("[WARNING] 当前环境未开启跳跃或无法触发手动跳跃")
        else:
            print("[WARNING] 当前环境不支持手动跳跃触发")

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # 处理 event.input 可能是字符串或对象的情况
        if isinstance(event.input, str):
            key_name = event.input
        else:
            key_name = event.input.name
        
        # 处理按键按下
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key_name == "L":
                self.reset()
            # 非增量式控制：W/S (速度) - 按下时设置值
            elif key_name in ["W", "S"]:
                if key_name in self._VELOCITY_KEY_MAPPING:
                    self._velocity_cmd[0] = self._VELOCITY_KEY_MAPPING[key_name]
                    self._pressed_keys.add(key_name)
            # 非增量式控制：A/D (转向) - 按下时设置值
            elif key_name in ["A", "D"]:
                if key_name in self._VELOCITY_KEY_MAPPING:
                    self._velocity_cmd[1] = self._VELOCITY_KEY_MAPPING[key_name]
                    self._pressed_keys.add(key_name)
            # 增量式控制：Z/X (高度) - 按下时设置增量
            elif key_name in ["Z", "X"]:
                if key_name in self._HEIGHT_KEY_MAPPING:
                    self._height_delta = self._HEIGHT_KEY_MAPPING[key_name]
                    self._pressed_keys.add(key_name)
            
            # additional callbacks
            if key_name in self._additional_callbacks:
                self._additional_callbacks[key_name]()

        # 处理按键释放
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # 非增量式控制：W/S/A/D - 松开时归零
            if key_name in ["W", "S"]:
                if key_name in self._pressed_keys:
                    self._velocity_cmd[0] = 0.0
                    self._pressed_keys.discard(key_name)
            elif key_name in ["A", "D"]:
                if key_name in self._pressed_keys:
                    self._velocity_cmd[1] = 0.0
                    self._pressed_keys.discard(key_name)
            # 增量式控制：Z/X - 松开时保持当前值（增量设为0）
            elif key_name in ["Z", "X"]:
                if key_name in self._pressed_keys:
                    self._height_delta = 0.0
                    self._pressed_keys.discard(key_name)

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding for mobile robot."""
        # 非增量式控制：速度命令映射 [v_x, omega_z]
        self._VELOCITY_KEY_MAPPING = {
            "W": self.pos_sensitivity,      # 前进速度
            "S": -self.pos_sensitivity,     # 后退速度
            "A": self.rot_sensitivity,      # 左转（正角速度）
            "D": -self.rot_sensitivity,     # 右转（负角速度）
        }
        # 增量式控制：高度增量映射
        self._HEIGHT_KEY_MAPPING = {
            "Z": self.height_sensitivity,   # 增加高度
            "X": -self.height_sensitivity,  # 降低高度
        }


@dataclass
class Se3KeyboardMobileCfg(DeviceCfg):
    """Configuration for SE3 keyboard devices for mobile robot."""

    pos_sensitivity: float = 0.8  # sensitivity for forward/backward movement
    rot_sensitivity: float = 1.0  # sensitivity for rotation (yaw)
    height_sensitivity: float = 0.1  # sensitivity for height change
    class_type: type[DeviceBase] = Se3KeyboardMobile
