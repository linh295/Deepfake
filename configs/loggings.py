import logging
import sys

from loguru import logger


class InterceptHandler(logging.Handler):
    """
    Handler để "bắt" tất cả log mặc định của Python (ví dụ: uvicorn, aio-pika)
    và chuyển hướng chúng qua Loguru.
    """

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # Đảm bảo frame còn tồn tại trước khi truy cập
        if frame:
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
        else:
            # Fallback nếu không tìm thấy frame
            logger.opt(exception=record.exc_info).log(level, record.getMessage())


def setup_logging():
    """
    Thiết lập Loguru làm logger chính.
    """
    # Tắt các handler mặc định và "bắt" log
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Xóa handler mặc định của loguru
    logger.remove()

    # Thêm handler mới với format và MÀU SẮC
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,  # BẬT MÀU
    )