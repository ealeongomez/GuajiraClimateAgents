# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""Sistema de logging centralizado para el proyecto."""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_logger(
    name: str = "ClimateGuajira",
    log_dir: Path = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """Configura y retorna un logger con handlers de consola y archivo.
    
    Args:
        name: Nombre del logger.
        log_dir: Directorio donde guardar los logs. Si None, usa logs/ en la raíz.
        console_level: Nivel de logging para consola (default: INFO).
        file_level: Nivel de logging para archivo (default: DEBUG).
    
    Returns:
        Logger configurado con handlers de consola y archivo rotativo.
    
    Example:
        >>> logger = setup_logger("TelegramBot")
        >>> logger.info("Bot iniciado")
        >>> logger.error("Error al procesar mensaje")
    """
    # Obtener o crear logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capturar todo, los handlers filtran
    
    # Evitar duplicar handlers si ya existe
    if logger.handlers:
        return logger
    
    # Directorio de logs
    if log_dir is None:
        project_root = Path(__file__).parent.parent.parent
        log_dir = project_root / "logs"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Formato detallado para logs
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Formato simple para consola
    console_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 1. Handler de consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 2. Handler de archivo general (rotativo)
    general_log = log_dir / "telegram_bot.log"
    file_handler = RotatingFileHandler(
        general_log,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,  # Mantener 5 archivos de respaldo
        encoding='utf-8'
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # 3. Handler de errores (solo ERROR y CRITICAL)
    error_log = log_dir / "errors.log"
    error_handler = RotatingFileHandler(
        error_log,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    # 4. Handler de interacciones de usuarios (solo para telegram_bot)
    if name == "TelegramBot":
        user_log = log_dir / f"user_interactions_{datetime.now().strftime('%Y%m')}.log"
        user_handler = RotatingFileHandler(
            user_log,
            maxBytes=20 * 1024 * 1024,  # 20 MB
            backupCount=3,
            encoding='utf-8'
        )
        user_handler.setLevel(logging.INFO)
        user_handler.setFormatter(detailed_formatter)
        logger.addHandler(user_handler)
    
    return logger


def log_user_interaction(logger: logging.Logger, user_id: int, message: str, response_length: int):
    """Registra una interacción de usuario de forma estructurada.
    
    Args:
        logger: Logger a usar.
        user_id: ID del usuario de Telegram.
        message: Mensaje del usuario.
        response_length: Longitud de la respuesta generada.
    """
    logger.info(
        f"USER_INTERACTION | user_id={user_id} | "
        f"message_length={len(message)} | "
        f"response_length={response_length} | "
        f"message='{message[:100]}...'"
    )


def log_error_with_context(logger: logging.Logger, error: Exception, context: dict):
    """Registra un error con contexto adicional.
    
    Args:
        logger: Logger a usar.
        error: Excepción capturada.
        context: Diccionario con contexto adicional (user_id, message, etc.).
    """
    context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
    logger.error(
        f"ERROR | {type(error).__name__}: {str(error)} | {context_str}",
        exc_info=True
    )

