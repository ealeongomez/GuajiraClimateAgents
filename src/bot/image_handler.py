# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""Manejo de imágenes generadas para Telegram."""

import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List


class ImageHandler:
    """Gestiona imágenes generadas y su limpieza periódica.
    
    Funcionalidades:
    - Guarda copias de imágenes para cada usuario
    - Limpia imágenes antiguas automáticamente
    - Organiza imágenes por usuario
    
    Attributes:
        base_dir: Directorio base para imágenes de usuarios.
    """
    
    def __init__(self, base_dir: Path):
        """Inicializa el handler de imágenes.
        
        Args:
            base_dir: Directorio base donde se guardarán las imágenes.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_user_image(self, user_id: int, image_path: Path) -> Path:
        """Guarda una copia de la imagen para el usuario.
        
        Crea un directorio específico para cada usuario y guarda
        la imagen con timestamp para evitar sobreescrituras.
        
        Args:
            user_id: ID del usuario de Telegram.
            image_path: Ruta de la imagen original generada.
        
        Returns:
            Ruta donde se guardó la copia de la imagen.
        """
        # Crear directorio del usuario
        user_dir = self.base_dir / str(user_id)
        user_dir.mkdir(exist_ok=True)
        
        # Copiar imagen con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{timestamp}_{image_path.name}"
        dest_path = user_dir / new_name
        
        shutil.copy2(image_path, dest_path)
        
        return dest_path
    
    def get_user_images(self, user_id: int) -> List[Path]:
        """Obtiene todas las imágenes guardadas de un usuario.
        
        Args:
            user_id: ID del usuario.
        
        Returns:
            Lista de rutas de imágenes del usuario.
        """
        user_dir = self.base_dir / str(user_id)
        
        if not user_dir.exists():
            return []
        
        return sorted(user_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    def cleanup_old_images(self, days: int = 7) -> int:
        """Limpia imágenes antiguas para liberar espacio.
        
        Elimina imágenes que tienen más de N días de antigüedad.
        
        Args:
            days: Número de días. Imágenes más antiguas se eliminan.
        
        Returns:
            Número de imágenes eliminadas.
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        # Iterar por todos los directorios de usuarios
        for user_dir in self.base_dir.iterdir():
            if user_dir.is_dir():
                # Revisar todas las imágenes PNG
                for img_file in user_dir.glob("*.png"):
                    file_time = datetime.fromtimestamp(img_file.stat().st_mtime)
                    
                    if file_time < cutoff_date:
                        img_file.unlink()
                        deleted_count += 1
                
                # Eliminar directorio si está vacío
                if not any(user_dir.iterdir()):
                    user_dir.rmdir()
        
        return deleted_count
    
    def get_total_size(self) -> int:
        """Calcula el tamaño total de imágenes almacenadas.
        
        Returns:
            Tamaño en bytes.
        """
        total_size = 0
        
        for user_dir in self.base_dir.iterdir():
            if user_dir.is_dir():
                for img_file in user_dir.glob("*.png"):
                    total_size += img_file.stat().st_size
        
        return total_size
    
    def get_total_images(self) -> int:
        """Cuenta el número total de imágenes almacenadas.
        
        Returns:
            Número de imágenes.
        """
        count = 0
        
        for user_dir in self.base_dir.iterdir():
            if user_dir.is_dir():
                count += len(list(user_dir.glob("*.png")))
        
        return count

