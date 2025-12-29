# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""Gestión de threads por usuario para mantener historial de conversaciones."""

import uuid
from datetime import datetime
from typing import Dict
from dataclasses import dataclass, field


@dataclass
class UserThread:
    """Thread de conversación de un usuario.
    
    Attributes:
        thread_id: Identificador único del thread (para checkpointer).
        user_id: ID del usuario de Telegram.
        created_at: Fecha de creación del thread.
        last_activity: Última vez que el usuario interactuó.
        message_count: Número total de mensajes enviados.
        image_count: Número de imágenes generadas para el usuario.
    """
    thread_id: str
    user_id: int
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    image_count: int = 0


class ThreadManager:
    """Gestiona threads de conversación independientes por usuario.
    
    Cada usuario de Telegram tiene su propio thread_id, lo que permite:
    - Mantener historial separado por usuario
    - Estadísticas independientes
    - Capacidad de reiniciar conversaciones
    
    Example:
        >>> manager = ThreadManager()
        >>> thread_id = manager.get_or_create_thread(123456)
        >>> config = {"configurable": {"thread_id": thread_id}}
        >>> response = graph.invoke({"messages": [...]}, config=config)
    """
    
    def __init__(self):
        """Inicializa el gestor de threads."""
        self.threads: Dict[int, UserThread] = {}
    
    def get_or_create_thread(self, user_id: int) -> str:
        """Obtiene o crea un thread para un usuario.
        
        Si el usuario no tiene thread, se crea uno nuevo con un ID único.
        Si ya existe, se actualiza la última actividad y se retorna el ID.
        
        Args:
            user_id: ID del usuario de Telegram.
        
        Returns:
            thread_id: ID único para usar en el checkpointer de LangGraph.
        """
        if user_id not in self.threads:
            # Crear nuevo thread con formato: telegram_<user_id>_<uuid>
            thread_id = f"telegram_{user_id}_{uuid.uuid4().hex[:8]}"
            self.threads[user_id] = UserThread(
                thread_id=thread_id,
                user_id=user_id
            )
        
        # Actualizar última actividad
        self.threads[user_id].last_activity = datetime.now()
        
        return self.threads[user_id].thread_id
    
    def reset_thread(self, user_id: int) -> str:
        """Reinicia el thread de un usuario (nueva conversación).
        
        Crea un nuevo thread_id, lo que efectivamente inicia una
        conversación nueva sin el historial anterior.
        
        Args:
            user_id: ID del usuario de Telegram.
        
        Returns:
            thread_id: Nuevo ID del thread.
        """
        # Crear nuevo thread con UUID diferente
        thread_id = f"telegram_{user_id}_{uuid.uuid4().hex[:8]}"
        self.threads[user_id] = UserThread(
            thread_id=thread_id,
            user_id=user_id
        )
        return thread_id
    
    def update_stats(self, user_id: int, has_image: bool = False):
        """Actualiza estadísticas del usuario.
        
        Args:
            user_id: ID del usuario.
            has_image: True si el mensaje incluye una imagen generada.
        """
        if user_id in self.threads:
            self.threads[user_id].message_count += 1
            if has_image:
                self.threads[user_id].image_count += 1
            self.threads[user_id].last_activity = datetime.now()
    
    def get_user_stats(self, user_id: int) -> Dict[str, any]:
        """Obtiene estadísticas del usuario.
        
        Args:
            user_id: ID del usuario.
        
        Returns:
            Diccionario con estadísticas: messages, images, last_activity.
        """
        if user_id not in self.threads:
            return {
                'messages': 0,
                'images': 0,
                'last_activity': 'Nunca'
            }
        
        thread = self.threads[user_id]
        return {
            'messages': thread.message_count,
            'images': thread.image_count,
            'last_activity': thread.last_activity.strftime('%Y-%m-%d %H:%M:%S'),
            'thread_id': thread.thread_id
        }
    
    def get_all_users(self) -> list[int]:
        """Obtiene lista de todos los user_ids con threads activos.
        
        Returns:
            Lista de user_ids.
        """
        return list(self.threads.keys())

