# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""Checkpointer para persistir estado del agente entre conversaciones."""

import sqlite3
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver


def get_checkpointer(db_path: str = None):
    """Crea y retorna un checkpointer SQLite para persistencia.
    
    El checkpointer guarda el estado completo de cada conversación,
    permitiendo:
    - Mantener historial entre reinicios del bot
    - Conversaciones independientes por usuario
    - Recuperación de estado en caso de errores
    
    Args:
        db_path: Ruta del archivo SQLite. Si None, usa ruta por defecto.
    
    Returns:
        SqliteSaver configurado y listo para usar.
    
    Example:
        >>> checkpointer = get_checkpointer()
        >>> graph = create_graph(checkpointer=checkpointer)
    """
    if db_path is None:
        # Usar ruta por defecto en data/checkpoints/
        project_root = Path(__file__).parent.parent.parent
        db_path = project_root / "data" / "checkpoints" / "telegram_checkpoints.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Crear conexión SQLite directa
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    
    # Crear checkpointer con la conexión
    checkpointer = SqliteSaver(conn)
    
    return checkpointer

