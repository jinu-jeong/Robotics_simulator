"""
런타임 통계 및 성능 모니터링 모듈
"""
import time
from typing import List, Dict, Optional


class RuntimeStats:
    """런타임 통계를 관리하는 클래스"""
    
    def __init__(self, history_size: int = 60):
        """
        Args:
            history_size: 유지할 프레임 히스토리 크기
        """
        self.history_size = history_size
        
        # 타이밍 데이터 (리스트 사용으로 최적화)
        self.frame_times: List[float] = []
        self.render_times: List[float] = []
        self.update_times: List[float] = []
        
        # 현재 프레임 타이밍
        self.frame_start_time = 0.0
        self.update_start_time = 0.0
        self.render_start_time = 0.0
        
        # 누적 통계
        self.total_frames = 0
        self.total_time = 0.0
        self.start_time = time.perf_counter()
        
        # 성능 임계값
        self.target_fps = 60.0
        self.warning_fps_threshold = 30.0
        self.critical_fps_threshold = 15.0
    
    def begin_frame(self):
        """프레임 시작을 기록합니다"""
        self.frame_start_time = time.perf_counter()
    
    def begin_update(self):
        """업데이트 단계 시작을 기록합니다"""
        self.update_start_time = time.perf_counter()
    
    def end_update(self):
        """업데이트 단계 종료를 기록합니다"""
        if self.update_start_time > 0:
            update_time = time.perf_counter() - self.update_start_time
            self.update_times.append(update_time)
            if len(self.update_times) > self.history_size:
                self.update_times.pop(0)
    
    def begin_render(self):
        """렌더링 단계 시작을 기록합니다"""
        self.render_start_time = time.perf_counter()
    
    def end_render(self):
        """렌더링 단계 종료를 기록합니다"""
        if self.render_start_time > 0:
            render_time = time.perf_counter() - self.render_start_time
            self.render_times.append(render_time)
            if len(self.render_times) > self.history_size:
                self.render_times.pop(0)
    
    def end_frame(self):
        """프레임 종료를 기록합니다"""
        if self.frame_start_time > 0:
            frame_time = time.perf_counter() - self.frame_start_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.history_size:
                self.frame_times.pop(0)
            self.total_frames += 1
            self.total_time += frame_time
    
    def get_fps(self) -> float:
        """현재 FPS를 반환합니다"""
        if not self.frame_times:
            return 0.0
        
        recent_frame_time = self.frame_times[-1]
        if recent_frame_time <= 0:
            return 0.0
        
        return 1.0 / recent_frame_time
    
    def get_average_fps(self) -> float:
        """평균 FPS를 반환합니다"""
        if len(self.frame_times) < 2:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        if avg_frame_time <= 0:
            return 0.0
        
        return 1.0 / avg_frame_time
    
    def get_min_fps(self) -> float:
        """최소 FPS를 반환합니다 (최대 프레임 시간의 역수)"""
        if not self.frame_times:
            return 0.0
        
        max_frame_time = max(self.frame_times)
        if max_frame_time <= 0:
            return 0.0
        
        return 1.0 / max_frame_time
    
    def get_max_fps(self) -> float:
        """최대 FPS를 반환합니다 (최소 프레임 시간의 역수)"""
        if not self.frame_times:
            return 0.0
        
        min_frame_time = min(self.frame_times)
        if min_frame_time <= 0:
            return 0.0
        
        return 1.0 / min_frame_time
    
    def get_frame_time_ms(self) -> float:
        """현재 프레임 시간을 밀리초로 반환합니다"""
        if not self.frame_times:
            return 0.0
        return self.frame_times[-1] * 1000.0
    
    def get_average_frame_time_ms(self) -> float:
        """평균 프레임 시간을 밀리초로 반환합니다"""
        if not self.frame_times:
            return 0.0
        
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return avg_time * 1000.0
    
    def get_update_time_ms(self) -> float:
        """현재 업데이트 시간을 밀리초로 반환합니다"""
        if not self.update_times:
            return 0.0
        return self.update_times[-1] * 1000.0
    
    def get_render_time_ms(self) -> float:
        """현재 렌더링 시간을 밀리초로 반환합니다"""
        if not self.render_times:
            return 0.0
        return self.render_times[-1] * 1000.0
    
    def get_uptime(self) -> float:
        """애플리케이션 실행 시간을 초로 반환합니다"""
        return time.perf_counter() - self.start_time
    
    def get_performance_level(self) -> str:
        """현재 성능 수준을 반환합니다"""
        current_fps = self.get_fps()
        
        if current_fps >= self.target_fps * 0.9:  # 90% 이상
            return "excellent"
        elif current_fps >= self.warning_fps_threshold:
            return "good"
        elif current_fps >= self.critical_fps_threshold:
            return "warning"
        else:
            return "critical"
    
    def get_frame_time_percentiles(self) -> Dict[str, float]:
        """프레임 시간 백분위수를 반환합니다"""
        if not self.frame_times:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
        
        sorted_times = sorted(self.frame_times)
        n = len(sorted_times)
        
        def percentile(p: float) -> float:
            idx = int(n * p / 100.0)
            idx = min(max(idx, 0), n - 1)
            return sorted_times[idx] * 1000.0  # ms로 변환
        
        return {
            "p50": percentile(50),
            "p90": percentile(90),
            "p95": percentile(95),
            "p99": percentile(99),
        }
    
    def get_summary(self) -> Dict:
        """통계 요약을 반환합니다"""
        percentiles = self.get_frame_time_percentiles()
        
        return {
            "fps": {
                "current": self.get_fps(),
                "average": self.get_average_fps(),
                "min": self.get_min_fps(),
                "max": self.get_max_fps(),
                "target": self.target_fps,
            },
            "frame_time_ms": {
                "current": self.get_frame_time_ms(),
                "average": self.get_average_frame_time_ms(),
                "p50": percentiles["p50"],
                "p90": percentiles["p90"],
                "p95": percentiles["p95"],
                "p99": percentiles["p99"],
            },
            "timing_ms": {
                "update": self.get_update_time_ms(),
                "render": self.get_render_time_ms(),
            },
            "system": {
                "total_frames": self.total_frames,
                "uptime": self.get_uptime(),
                "performance_level": self.get_performance_level(),
            }
        }
    
    def reset(self):
        """통계를 리셋합니다"""
        self.frame_times.clear()
        self.render_times.clear()
        self.update_times.clear()
        self.total_frames = 0
        self.total_time = 0.0
        self.start_time = time.perf_counter()
    
    def __str__(self) -> str:
        """문자열 표현"""
        return (f"RuntimeStats(fps={self.get_fps():.1f}, "
                f"avg_fps={self.get_average_fps():.1f}, "
                f"frame_time={self.get_frame_time_ms():.2f}ms, "
                f"frames={self.total_frames})")