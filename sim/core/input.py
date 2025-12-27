"""
입력 상태 관리 및 처리 모듈
"""
import taichi as ti
from typing import Dict, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MouseState:
    """마우스 상태를 관리하는 데이터클래스"""
    x: float = 0.0
    y: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    wheel_delta: float = 0.0
    
    # 버튼 상태 (현재 프레임)
    left_pressed: bool = False
    right_pressed: bool = False
    middle_pressed: bool = False
    
    # 버튼 상태 (이전 프레임)
    left_pressed_prev: bool = False
    right_pressed_prev: bool = False
    middle_pressed_prev: bool = False
    
    def update_previous_state(self):
        """이전 프레임 상태를 업데이트합니다"""
        self.left_pressed_prev = self.left_pressed
        self.right_pressed_prev = self.right_pressed
        self.middle_pressed_prev = self.middle_pressed
    
    def is_left_clicked(self) -> bool:
        """왼쪽 버튼이 클릭되었는지 확인 (눌림 -> 떼짐)"""
        return self.left_pressed_prev and not self.left_pressed
    
    def is_right_clicked(self) -> bool:
        """오른쪽 버튼이 클릭되었는지 확인"""
        return self.right_pressed_prev and not self.right_pressed
    
    def is_middle_clicked(self) -> bool:
        """중간 버튼이 클릭되었는지 확인"""
        return self.middle_pressed_prev and not self.middle_pressed
    
    def is_left_just_pressed(self) -> bool:
        """왼쪽 버튼이 방금 눌렸는지 확인 (떼짐 -> 눌림)"""
        return not self.left_pressed_prev and self.left_pressed
    
    def is_right_just_pressed(self) -> bool:
        """오른쪽 버튼이 방금 눌렸는지 확인"""
        return not self.right_pressed_prev and self.right_pressed
    
    def is_middle_just_pressed(self) -> bool:
        """중간 버튼이 방금 눌렸는지 확인"""
        return not self.middle_pressed_prev and self.middle_pressed


@dataclass
class KeyboardState:
    """키보드 상태를 관리하는 데이터클래스"""
    pressed_keys: Set[str] = field(default_factory=set)
    pressed_keys_prev: Set[str] = field(default_factory=set)
    
    def update_previous_state(self):
        """이전 프레임 상태를 업데이트합니다"""
        self.pressed_keys_prev = self.pressed_keys.copy()
    
    def is_key_pressed(self, key: str) -> bool:
        """키가 현재 눌려있는지 확인"""
        return key in self.pressed_keys
    
    def is_key_just_pressed(self, key: str) -> bool:
        """키가 방금 눌렸는지 확인"""
        return key in self.pressed_keys and key not in self.pressed_keys_prev
    
    def is_key_just_released(self, key: str) -> bool:
        """키가 방금 떼어졌는지 확인"""
        return key not in self.pressed_keys and key in self.pressed_keys_prev


class InputState:
    """전체 입력 상태를 관리하는 클래스"""
    
    def __init__(self):
        self.mouse = MouseState()
        self.keyboard = KeyboardState()
        
        # 키 매핑 (Taichi 키 코드 -> 문자열)
        self.key_map = {}
        
        # 안전하게 키 매핑 추가
        key_mappings = [
            ("SPACE", "space"),
            ("ESCAPE", "escape"),
            ("RETURN", "enter"),  # ENTER 대신 RETURN 사용
            ("TAB", "tab"),
            ("BACKSPACE", "backspace"),
            ("DELETE", "delete"),
            ("LEFT", "left"),
            ("RIGHT", "right"),
            ("UP", "up"),
            ("DOWN", "down"),
            ("LSHIFT", "shift"),  # SHIFT 대신 LSHIFT 사용
            ("LCTRL", "ctrl"),    # CTRL 대신 LCTRL 사용
            ("LALT", "alt"),      # ALT 대신 LALT 사용
        ]
        
        for ti_key_name, key_name in key_mappings:
            if hasattr(ti.GUI, ti_key_name):
                self.key_map[getattr(ti.GUI, ti_key_name)] = key_name
        
        # 문자 키들 추가
        for i in range(26):
            char = chr(ord('a') + i)
            ti_key_name = char.upper()
            if hasattr(ti.GUI, ti_key_name):
                self.key_map[getattr(ti.GUI, ti_key_name)] = char
        
        # 숫자 키들 추가
        for i in range(10):
            ti_key_name = f"_{i}"
            if hasattr(ti.GUI, ti_key_name):
                self.key_map[getattr(ti.GUI, ti_key_name)] = str(i)
    
    def update_previous_state(self):
        """이전 프레임 상태를 업데이트합니다"""
        self.mouse.update_previous_state()
        self.keyboard.update_previous_state()
    
    def clear_frame_data(self):
        """프레임별 데이터를 초기화합니다"""
        self.mouse.dx = 0.0
        self.mouse.dy = 0.0
        self.mouse.wheel_delta = 0.0


def poll_input(window, input_state: InputState) -> bool:
    """
    윈도우에서 입력을 폴링하고 InputState를 업데이트합니다.
    
    Args:
        window: Taichi UI 윈도우 (ti.ui.Window 또는 ti.GUI)
        input_state: 업데이트할 입력 상태
        
    Returns:
        bool: 윈도우가 계속 실행되어야 하는지 여부
    """
    # 이전 상태 저장
    input_state.update_previous_state()
    input_state.clear_frame_data()
    
    # 윈도우 실행 상태 확인
    if not window.running:
        return False
    
    # 마우스 위치 업데이트
    mouse_x, mouse_y = window.get_cursor_pos()
    prev_x, prev_y = input_state.mouse.x, input_state.mouse.y
    
    input_state.mouse.x = mouse_x
    input_state.mouse.y = mouse_y
    input_state.mouse.dx = mouse_x - prev_x
    input_state.mouse.dy = mouse_y - prev_y
    
    # 마우스 버튼 상태 업데이트 (ti.ui.Window와 ti.GUI 모두 지원)
    try:
        # ti.ui.Window 방식 시도
        input_state.mouse.left_pressed = window.is_pressed(ti.ui.LMB)
        input_state.mouse.right_pressed = window.is_pressed(ti.ui.RMB)
        input_state.mouse.middle_pressed = window.is_pressed(ti.ui.MMB)
    except (AttributeError, TypeError):
        # ti.GUI 방식으로 폴백
        input_state.mouse.left_pressed = window.is_pressed(ti.GUI.LMB)
        input_state.mouse.right_pressed = window.is_pressed(ti.GUI.RMB)
        input_state.mouse.middle_pressed = window.is_pressed(ti.GUI.MMB)
    
    # 키보드 상태 업데이트
    input_state.keyboard.pressed_keys.clear()
    
    for ti_key, key_name in input_state.key_map.items():
        try:
            if window.is_pressed(ti_key):
                input_state.keyboard.pressed_keys.add(key_name)
        except (AttributeError, TypeError):
            # 일부 키가 지원되지 않을 수 있음
            pass
    
    # 이벤트 처리
    try:
        for event in window.get_events():
            if hasattr(event, 'type'):
                # ti.GUI 스타일 이벤트
                if hasattr(ti.GUI, 'WHEEL') and event.type == ti.GUI.WHEEL:
                    input_state.mouse.wheel_delta = event.delta[1]
            elif hasattr(event, 'key'):
                # ti.ui.Window 스타일 이벤트 (필요시 추가 구현)
                pass
    except (AttributeError, TypeError):
        # 이벤트 처리가 지원되지 않는 경우
        pass
    
    return True


def get_camera_control_input(input_state: InputState) -> Dict[str, float]:
    """
    카메라 컨트롤을 위한 입력값을 추출합니다.
    
    Args:
        input_state: 입력 상태
        
    Returns:
        dict: 카메라 컨트롤 입력 정보
    """
    result = {
        "orbit_dx": 0.0,
        "orbit_dy": 0.0,
        "pan_dx": 0.0,
        "pan_dy": 0.0,
        "zoom_delta": 0.0,
    }
    
    # 궤도 회전 (왼쪽 마우스 버튼 드래그)
    if input_state.mouse.left_pressed:
        result["orbit_dx"] = input_state.mouse.dx
        result["orbit_dy"] = input_state.mouse.dy
    
    # 팬 (오른쪽 또는 중간 마우스 버튼 드래그)
    if input_state.mouse.right_pressed or input_state.mouse.middle_pressed:
        result["pan_dx"] = input_state.mouse.dx
        result["pan_dy"] = input_state.mouse.dy
    
    # 줌 (마우스 휠)
    result["zoom_delta"] = input_state.mouse.wheel_delta
    
    return result


def get_simulation_control_input(input_state: InputState) -> Dict[str, bool]:
    """
    시뮬레이션 컨트롤을 위한 입력값을 추출합니다.
    
    Args:
        input_state: 입력 상태
        
    Returns:
        dict: 시뮬레이션 컨트롤 입력 정보
    """
    return {
        "pause_toggle": input_state.keyboard.is_key_just_pressed("space"),
        "single_step": input_state.keyboard.is_key_just_pressed("n"),
        "reset": input_state.keyboard.is_key_just_pressed("r"),
        "quit": (input_state.keyboard.is_key_just_pressed("q") or 
                input_state.keyboard.is_key_just_pressed("escape")),
    }