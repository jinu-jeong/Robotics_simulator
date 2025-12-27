"""
Taichi 초기화 및 애플리케이션 설정 모듈
"""
import taichi as ti
import logging
from typing import Optional
from .config import SimConfig

logger = logging.getLogger(__name__)


def init_taichi(config: SimConfig) -> bool:
    """
    Taichi를 초기화합니다.
    
    Args:
        config: 시뮬레이션 설정
        
    Returns:
        bool: 초기화 성공 여부
    """
    arch_map = {
        "cpu": ti.cpu,
        "metal": ti.metal,
        "vulkan": ti.vulkan,
        "cuda": ti.cuda,
    }
    
    if config.arch not in arch_map:
        logger.error(f"지원하지 않는 아키텍처: {config.arch}")
        return False
    
    target_arch = arch_map[config.arch]
    
    try:
        # 첫 번째 시도: 요청된 아키텍처로 초기화
        init_kwargs = {
            "arch": target_arch,
            "debug": False,
            "log_level": ti.INFO
        }
        if config.seed is not None:
            init_kwargs["random_seed"] = config.seed
        
        ti.init(**init_kwargs)
        logger.info(f"Taichi 초기화 성공: {config.arch}")
        return True
        
    except Exception as e:
        logger.warning(f"{config.arch} 아키텍처 초기화 실패: {e}")
        
        # macOS에서 Metal 실패 시 CPU로 폴백
        if config.arch == "metal":
            try:
                logger.info("Metal에서 CPU로 폴백 시도...")
                ti.reset()  # 이전 초기화 상태 리셋
                fallback_kwargs = {
                    "arch": ti.cpu,
                    "debug": False,
                    "log_level": ti.INFO
                }
                if config.seed is not None:
                    fallback_kwargs["random_seed"] = config.seed
                
                ti.init(**fallback_kwargs)
                config.arch = "cpu"  # 설정 업데이트
                logger.info("CPU 아키텍처로 폴백 성공")
                return True
            except Exception as fallback_e:
                logger.error(f"CPU 폴백도 실패: {fallback_e}")
                return False
        
        # 다른 아키텍처에서 실패 시 CPU로 폴백
        try:
            logger.info(f"{config.arch}에서 CPU로 폴백 시도...")
            ti.reset()
            fallback_kwargs = {
                "arch": ti.cpu,
                "debug": False,
                "log_level": ti.INFO
            }
            if config.seed is not None:
                fallback_kwargs["random_seed"] = config.seed
            
            ti.init(**fallback_kwargs)
            config.arch = "cpu"
            logger.info("CPU 아키텍처로 폴백 성공")
            return True
        except Exception as fallback_e:
            logger.error(f"CPU 폴백 실패: {fallback_e}")
            return False


def get_arch_info() -> dict:
    """
    현재 Taichi 아키텍처 정보를 반환합니다.
    
    Returns:
        dict: 아키텍처 정보
    """
    try:
        arch = ti.lang.impl.current_cfg().arch
        return {
            "arch": str(arch),
            "device_memory_GB": ti.lang.impl.current_cfg().device_memory_GB,
            "debug": ti.lang.impl.current_cfg().debug,
        }
    except Exception as e:
        logger.warning(f"아키텍처 정보 조회 실패: {e}")
        return {"arch": "unknown", "error": str(e)}


class TaichiApp:
    """Taichi 애플리케이션 래퍼 클래스"""
    
    def __init__(self, config: SimConfig):
        self.config = config
        self.initialized = False
        
    def initialize(self) -> bool:
        """애플리케이션 초기화"""
        if self.initialized:
            logger.warning("이미 초기화된 애플리케이션입니다")
            return True
            
        success = init_taichi(self.config)
        if success:
            self.initialized = True
            logger.info(f"TaichiApp 초기화 완료: {self.get_info()}")
        else:
            logger.error("TaichiApp 초기화 실패")
            
        return success
    
    def cleanup(self):
        """리소스 정리"""
        if self.initialized:
            try:
                ti.reset()
                self.initialized = False
                logger.info("TaichiApp 정리 완료")
            except Exception as e:
                logger.error(f"TaichiApp 정리 중 오류: {e}")
    
    def get_info(self) -> dict:
        """애플리케이션 정보 반환"""
        info = {
            "initialized": self.initialized,
            "config": self.config.to_dict(),
        }
        
        if self.initialized:
            info.update(get_arch_info())
            
        return info
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.cleanup()