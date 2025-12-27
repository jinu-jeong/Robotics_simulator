"""
Taichi 기반 실시간 3D 시뮬레이션 애플리케이션 메인 엔트리포인트
"""
import argparse
import sys
import logging
from typing import Optional

from .core.config import SimConfig
from .core.app import TaichiApp
from .core.time import FixedStepper
from .core.input import InputState, poll_input, get_simulation_control_input
from .core.stats import RuntimeStats
from .render.ti_viewer import TaichiViewer3D
from .render.hud import HUD
from .render.debug_draw import DebugDraw


def parse_arguments() -> argparse.Namespace:
    """명령행 인수를 파싱합니다"""
    parser = argparse.ArgumentParser(
        description="Taichi 기반 실시간 3D 시뮬레이션",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 아키텍처 설정
    parser.add_argument(
        "--arch", 
        choices=["cpu", "metal", "vulkan", "cuda", "auto"],
        default="auto",
        help="Taichi 백엔드 아키텍처"
    )
    
    # 시뮬레이션 파라미터
    parser.add_argument(
        "--dt", 
        type=float, 
        default=1.0/60.0,
        help="시뮬레이션 타임스텝 (초)"
    )
    
    parser.add_argument(
        "--substeps", 
        type=int, 
        default=1,
        help="서브스텝 수"
    )
    
    # 윈도우 설정
    parser.add_argument(
        "--width", 
        type=int, 
        default=1024,
        help="윈도우 너비"
    )
    
    parser.add_argument(
        "--height", 
        type=int, 
        default=768,
        help="윈도우 높이"
    )
    
    parser.add_argument(
        "--vsync", 
        type=int, 
        choices=[0, 1],
        default=1,
        help="수직 동기화 (0=비활성화, 1=활성화)"
    )
    
    # 기타 옵션
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="랜덤 시드"
    )
    
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true",
        help="상세 로그 출력"
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """로깅을 설정합니다"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )


def create_config(args: argparse.Namespace) -> SimConfig:
    """명령행 인수로부터 설정을 생성합니다"""
    return SimConfig(
        dt=args.dt,
        substeps=args.substeps,
        window_width=args.width,
        window_height=args.height,
        vsync=bool(args.vsync),
        arch=args.arch,
        seed=args.seed,
    )


class SimulationApp:
    """시뮬레이션 애플리케이션 클래스"""
    
    def __init__(self, config: SimConfig):
        self.config = config
        self.running = False
        
        # 핵심 컴포넌트
        self.taichi_app: Optional[TaichiApp] = None
        self.stepper: Optional[FixedStepper] = None
        self.viewer: Optional[TaichiViewer3D] = None
        self.input_state: Optional[InputState] = None
        self.stats: Optional[RuntimeStats] = None
        self.hud: Optional[HUD] = None
        self.debug_draw: Optional[DebugDraw] = None
        
        # 로거
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> bool:
        """애플리케이션을 초기화합니다"""
        self.logger.info(f"시뮬레이션 초기화 시작: {self.config}")
        
        try:
            # Taichi 초기화
            self.taichi_app = TaichiApp(self.config)
            if not self.taichi_app.initialize():
                self.logger.error("Taichi 초기화 실패")
                return False
            
            # 시간 관리자 초기화
            self.stepper = FixedStepper(self.config)
            
            # 3D 뷰어 초기화
            self.viewer = TaichiViewer3D(self.config)
            if not self.viewer.initialize():
                self.logger.error("3D 뷰어 초기화 실패")
                return False
            
            # 입력 상태 초기화
            self.input_state = InputState()
            
            # 통계 관리자 초기화
            self.stats = RuntimeStats()
            
            # HUD 초기화
            self.hud = HUD(self.config)
            
            # 디버그 그리기 초기화
            self.debug_draw = DebugDraw()
            
            self.logger.info("시뮬레이션 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"초기화 중 오류 발생: {e}")
            return False
    
    def run(self) -> int:
        """메인 루프를 실행합니다"""
        if not self.initialize():
            return 1
        
        self.running = True
        self.logger.info("시뮬레이션 시작")
        
        try:
            while self.running and self.viewer.is_running():
                self.stats.begin_frame()
                
                # 입력 처리
                if not self._handle_input():
                    break
                
                # 시뮬레이션 업데이트
                self._update_simulation()
                
                # 렌더링
                self._render_frame()
                
                self.stats.end_frame()
            
            self.logger.info("시뮬레이션 종료")
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("사용자에 의한 중단")
            return 0
        except Exception as e:
            self.logger.error(f"실행 중 오류 발생: {e}")
            return 1
        finally:
            self._cleanup()
    
    def _handle_input(self) -> bool:
        """입력을 처리합니다"""
        window = self.viewer.get_window()
        if not poll_input(window, self.input_state):
            return False
        
        # 카메라 컨트롤 업데이트
        self.viewer.update_camera_from_input(self.input_state)
        
        # 시뮬레이션 컨트롤 처리
        sim_input = get_simulation_control_input(self.input_state)
        
        if sim_input["quit"]:
            self.logger.info("종료 요청")
            return False
        
        if sim_input["pause_toggle"]:
            self.stepper.toggle_pause()
        
        if sim_input["single_step"]:
            self.stepper.request_single_step()
        
        if sim_input["reset"]:
            self._reset_simulation()
        
        return True
    
    def _update_simulation(self):
        """시뮬레이션을 업데이트합니다"""
        self.stats.begin_update()
        
        # 시뮬레이션 스텝 실행
        substeps = self.stepper.step()
        
        # 여기에 물리 시뮬레이션 로직이 들어갈 예정
        # 현재는 스캐폴딩 단계이므로 빈 상태
        
        self.stats.end_update()
    
    def _render_frame(self):
        """프레임을 렌더링합니다"""
        self.stats.begin_render()
        
        # 3D 렌더링 시작
        self.viewer.begin_frame()
        
        # 참조 지오메트리 렌더링
        self._render_reference_geometry()
        
        # 3D 렌더링 완료
        self.viewer.end_frame()
        
        # HUD 렌더링
        self._render_hud()
        
        self.stats.end_render()
    
    def _render_reference_geometry(self):
        """참조 지오메트리를 렌더링합니다"""
        # 바닥 그리드
        self.viewer.draw_ground_grid(size=10.0, divisions=20)
        
        # 월드 축
        self.viewer.draw_world_axes(length=1.0)
        
        # 추가 디버그 요소들 (선택적)
        if hasattr(self, 'show_debug_shapes') and self.show_debug_shapes:
            scene = self.viewer.scene
            if scene:
                # 예시: 원점에 작은 박스
                self.debug_draw.draw_box_wireframe(
                    scene, 
                    center=(0, 0.5, 0), 
                    size=(0.5, 0.5, 0.5),
                    color="yellow"
                )
    
    def _render_hud(self):
        """HUD를 렌더링합니다"""
        window = self.viewer.get_window()
        camera_info = self.viewer.get_camera_info()
        
        self.hud.render(
            window=window,
            stepper=self.stepper,
            stats=self.stats,
            camera_info=camera_info
        )
    
    def _reset_simulation(self):
        """시뮬레이션을 리셋합니다"""
        self.logger.info("시뮬레이션 리셋")
        self.stepper.reset()
        self.stats.reset()
        # 추가 리셋 로직이 필요하면 여기에 추가
    
    def _cleanup(self):
        """리소스를 정리합니다"""
        self.logger.info("리소스 정리 중...")
        
        if self.viewer:
            self.viewer.cleanup()
        
        if self.taichi_app:
            self.taichi_app.cleanup()
        
        self.logger.info("정리 완료")


def main() -> int:
    """메인 함수"""
    # 명령행 인수 파싱
    args = parse_arguments()
    
    # 로깅 설정
    setup_logging(args.verbose)
    
    # 설정 생성
    try:
        config = create_config(args)
    except Exception as e:
        print(f"설정 오류: {e}")
        return 1
    
    # 애플리케이션 실행
    app = SimulationApp(config)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())