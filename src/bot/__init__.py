# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""
Telegram Bot Package for ClimateGuajira Agent.

This package provides a production-ready Telegram bot interface
for the ClimateGuajira agent with:
- Per-user conversation history
- Image generation and delivery
- State persistence with checkpointing
- Usage statistics
"""

from src.bot.telegram_bot import ClimateBot
from src.bot.thread_manager import ThreadManager
from src.bot.image_handler import ImageHandler
from src.bot.checkpointer import get_checkpointer

__all__ = [
    "ClimateBot",
    "ThreadManager",
    "ImageHandler",
    "get_checkpointer",
]

