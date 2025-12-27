"""
시간 관리 및 고정 타임스텝 모듈
"""
import time
from typing import Optional
from .config import SimConfig


class FixedStepper:
    """고정 타임스텝을 관리하는 클래스"""
    
    def __init__(self, config: SimConfig):
        self.config = config
        self.dt = config.dt
        self.substeps = config.substeps
        self.effective_dt = config.effective_dt
        
        # 시뮬레이션 상태
        self.frame_idx = 0
        self.sim_time = 0.0
        self.paused = False
        self.single_step_requested = False
        
        # 타이밍 관리
        self.last_frame_time = time.perf_counter()
        self.accumulator = 0.0
        self.max_accumulator = self.dt * 5  # 최대 5프레임까지 누적
        
        # 성능 측정
        self.frame_times = []
        self.max_frame_history = 60
        
    def reset(self):
        """시뮬레이션 시간을 리셋합니다"""
        self.frame_idx = 0
        self.sim_time = 0.0
        self.accumulator = 0.0
        self.last_frame_time = time.perf_counter()
        self.frame_times.clear()
        print(f"시뮬레이션 리셋: frame={self.frame_idx}, time={self.sim_time:.3f}s")
    
    def pause(self):
        """시뮬레이션을 일시정지합니다"""
        if not self.paused:
            self.paused = True
            print("시뮬레이션 일시정지")
    
    def resume(self):
        """시뮬레이션을 재개합니다"""
        if self.paused:
            self.paused = False
            self.last_frame_time = time.perf_counter()  # 시간 점프 방지
            print("시뮬레이션 재개")
    
    def toggle_pause(self):
        """일시정지 상태를 토글합니다"""
        if self.paused:
            self.resume()
        else:
            self.pause()
    
    def request_single_step(self):
        """단일 스텝 실행을 요청합니다"""
        if self.paused:
            self.single_step_requested = True
            print(f"단일 스텝 요청: frame {self.frame_idx} -> {self.frame_idx + 1}")
    
    def should_step(self) -> bool:
        """시뮬레이션 스텝을 실행해야 하는지 확인합니다"""
        current_time = time.perf_counter()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # 프레임 시간 기록
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)
        
        # 일시정지 상태 처리
        if self.paused:
            if self.single_step_requested:
                self.single_step_requested = False
                return True
            return False
        
        # 고정 타임스텝 로직
        self.accumulator += frame_time
        
        # 너무 많은 시간이 누적되면 제한
        if self.accumulator > self.max_accumulator:
            self.accumulator = self.max_accumulator
        
        # 한 프레임 분량의 시간이 누적되었는지 확인
        return self.accumulator >= self.dt
    
    def step(self) -> int:
        """
        시뮬레이션 스텝을 실행합니다.
        
        Returns:
            int: 실행된 서브스텝 수
        """
        if not self.should_step():
            return 0
        
        # 타임스텝 소모
        self.accumulator -= self.dt
        
        # 시뮬레이션 시간 업데이트
        self.sim_time += self.dt
        self.frame_idx += 1
        
        return self.substeps
    
    def get_interpolation_alpha(self) -> float:
        """
        렌더링을 위한 보간 알파값을 반환합니다.
        
        Returns:
            float: 0.0~1.0 사이의 보간값
        """
        if self.paused:
            return 1.0
        return min(self.accumulator / self.dt, 1.0)
    
    def get_average_fps(self) -> float:
        """평균 FPS를 계산합니다"""
        if len(self.frame_times) < 2:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        if avg_frame_time <= 0:
            return 0.0
        
        return 1.0 / avg_frame_time
    
    def get_current_fps(self) -> float:
        """현재 FPS를 계산합니다"""
        if not self.frame_times:
            return 0.0
        
        current_frame_time = self.frame_times[-1]
        if current_frame_time <= 0:
            return 0.0
        
        return 1.0 / current_frame_time
    
    def get_status(self) -> dict:
        """현재 상태를 딕셔너리로 반환합니다"""
        return {
            "frame_idx": self.frame_idx,
            "sim_time": self.sim_time,
            "paused": self.paused,
            "dt": self.dt,
            "substeps": self.substeps,
            "effective_dt": self.effective_dt,
            "target_fps": 1.0 / self.dt,
            "current_fps": self.get_current_fps(),
            "average_fps": self.get_average_fps(),
            "accumulator": self.accumulator,
        }
    
    def __str__(self) -> str:
        """문자열 표현"""
        status = "PAUSED" if self.paused else "RUNNING"
        return (f"FixedStepper({status}, frame={self.frame_idx}, "
                f"time={self.sim_time:.3f}s, fps={self.get_current_fps():.1f})")