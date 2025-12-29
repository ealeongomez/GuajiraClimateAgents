# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""Telegram Bot para ClimateGuajira Agent."""

import os
import re
from pathlib import Path
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from dotenv import load_dotenv

from src.agents.climate_guajira import create_graph, Configuration
from src.bot.thread_manager import ThreadManager
from src.bot.image_handler import ImageHandler
from src.bot.checkpointer import get_checkpointer
from src.utils.logger import setup_logger, log_user_interaction, log_error_with_context

load_dotenv()

# Configurar logger para el bot
logger = setup_logger("TelegramBot")

# Configuración
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Inicializar componentes globales con diagnóstico
logger.info("🔧 Inicializando componentes del bot...")
config = Configuration()

# Verificar configuración de DB
db_config = config.get_db_config()
logger.info(f"📊 DB Config: server={db_config['server']}, database={db_config['database']}")

checkpointer = get_checkpointer()
logger.info("✅ Checkpointer inicializado")

graph = create_graph(config, checkpointer=checkpointer)
logger.info("✅ Grafo inicializado")

thread_manager = ThreadManager()
image_handler = ImageHandler(PROJECT_ROOT / "data" / "user_images")
logger.info("✅ Managers inicializados")


class ClimateBot:
    """Bot de Telegram para ClimateGuajira Agent.
    
    Proporciona interfaz de Telegram para el agente climático con:
    - Historial de conversación persistente por usuario
    - Generación y envío de gráficas
    - Comandos de control (/start, /help, /reset, /stats)
    - Manejo de errores robusto
    """
    
    def __init__(self):
        """Inicializa el bot de Telegram."""
        if not TELEGRAM_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN no está configurado en .env")
        
        self.app = Application.builder().token(TELEGRAM_TOKEN).build()
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Configura los handlers de comandos y mensajes."""
        # Comandos
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("reset", self.reset_command))
        self.app.add_handler(CommandHandler("stats", self.stats_command))
        
        # Mensajes de texto
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler para comando /start.
        
        Presenta el bot al usuario y crea su thread de conversación.
        """
        user_id = update.effective_user.id
        username = update.effective_user.first_name or "Usuario"
        
        logger.info(f"🆕 User {user_id} ({username}) | Command: /start")
        
        welcome_msg = f"""
¡Hola {username}! 👋

🌬️ Soy el **Asistente de Clima de La Guajira**

Estoy especializado en:
📊 Datos climáticos históricos (2015-2025)
📚 Atlas Eólico de Colombia
📈 Visualizaciones y gráficas
🌍 13 municipios de La Guajira

**Puedes preguntarme sobre:**
• Estadísticas de viento y temperatura
• Comparaciones entre municipios
• Potencial eólico de la región
• Patrones horarios y temporales
• ¡Y mucho más!

Escribe tu pregunta o usa /help para ver ejemplos.
"""
        await update.message.reply_text(welcome_msg)
        
        # Crear thread para el usuario
        thread_manager.get_or_create_thread(user_id)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler para comando /help.
        
        Muestra ejemplos de uso y comandos disponibles.
        """
        help_msg = """
📋 **Ejemplos de preguntas:**

**📊 Datos históricos:**
• "Dame las estadísticas de Riohacha"
• "Compara el viento entre Uribia y Maicao"
• "¿Cuál es el municipio con más viento?"
• "Datos de Manaure en enero 2025"

**📈 Visualizaciones:**
• "Gráfica del viento en Riohacha en enero 2025"
• "Patrón horario de Uribia en diciembre 2024"
• "Compara visualmente Maicao, Riohacha y Uribia"
• "Grafica viento vs temperatura en Maicao"

**📚 Atlas Eólico:**
• "¿Cuál es el potencial eólico de La Guajira?"
• "¿Qué zonas son aptas para parques eólicos?"
• "Capacidad de generación eólica en Colombia"

**⚙️ Comandos:**
/start - Iniciar bot
/help - Ver esta ayuda
/reset - Reiniciar conversación
/stats - Ver tus estadísticas
"""
        await update.message.reply_text(help_msg, parse_mode='Markdown')
    
    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler para comando /reset.
        
        Reinicia la conversación del usuario (nuevo thread_id).
        """
        user_id = update.effective_user.id
        old_thread = thread_manager.threads.get(user_id)
        new_thread_id = thread_manager.reset_thread(user_id)
        
        logger.info(f"🔄 User {user_id} | Command: /reset | Old thread: {old_thread.thread_id if old_thread else 'N/A'} | New thread: {new_thread_id}")
        
        await update.message.reply_text(
            "✅ Conversación reiniciada.\n\n"
            "Empecemos de nuevo! ¿En qué puedo ayudarte?"
        )
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler para comando /stats.
        
        Muestra estadísticas de uso del usuario.
        """
        user_id = update.effective_user.id
        stats = thread_manager.get_user_stats(user_id)
        
        logger.info(f"📊 User {user_id} | Command: /stats | Messages: {stats['messages']} | Images: {stats['images']}")
        
        # Obtener info de imágenes
        total_images_stored = len(image_handler.get_user_images(user_id))
        
        stats_msg = f"""
📊 **Tus estadísticas:**

• Mensajes enviados: {stats['messages']}
• Gráficas generadas: {stats['images']}
• Imágenes guardadas: {total_images_stored}
• Última actividad: {stats['last_activity']}
• Thread ID: `{stats.get('thread_id', 'N/A')}`

💡 Usa /reset para empezar una conversación nueva.
"""
        await update.message.reply_text(stats_msg, parse_mode='Markdown')
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler principal para mensajes de texto del usuario.
        
        Procesa el mensaje, invoca el agente con historial persistente,
        y envía la respuesta con imágenes si fueron generadas.
        """
        user_id = update.effective_user.id
        user_message = update.message.text
        
        # Mostrar indicador "escribiendo..."
        await update.message.chat.send_action("typing")
        
        try:
            # Obtener thread del usuario (con historial)
            thread_id = thread_manager.get_or_create_thread(user_id)
            
            # Configurar checkpointer con thread_id
            config_dict = {"configurable": {"thread_id": thread_id}}
            
            # Invocar el agente con historial persistente
            logger.info(f"🤖 User {user_id} | Thread: {thread_id} | Message: '{user_message[:50]}...'")
            response = graph.invoke(
                {"messages": [("user", user_message)]},
                config=config_dict
            )
            
            # Extraer respuesta del asistente
            assistant_message = response["messages"][-1].content
            logger.info(f"✅ User {user_id} | Response generated: {len(assistant_message)} chars")
            
            # Log de interacción estructurado
            log_user_interaction(logger, user_id, user_message, len(assistant_message))
            
            # Buscar si hay imágenes generadas en la respuesta
            image_paths = self._extract_image_paths(assistant_message)
            
            if image_paths:
                # Limpiar mensaje de rutas de imágenes
                clean_message = self._remove_image_paths(assistant_message)
                await update.message.reply_text(clean_message)
                
                logger.info(f"📷 User {user_id} | Sending {len(image_paths)} image(s)")
                
                # Enviar cada imagen encontrada
                for img_path_str in image_paths:
                    img_path = Path(img_path_str)
                    
                    if img_path.exists():
                        # Mostrar indicador "subiendo foto..."
                        await update.message.chat.send_action("upload_photo")
                        
                        # Enviar imagen
                        with open(img_path, 'rb') as photo_file:
                            await update.message.reply_photo(
                                photo=photo_file,
                                caption=f"📊 {img_path.name}"
                            )
                        
                        logger.info(f"✅ User {user_id} | Image sent: {img_path.name}")
                        
                        # Guardar copia para el usuario
                        image_handler.save_user_image(user_id, img_path)
                    else:
                        logger.warning(f"⚠️ User {user_id} | Image not found: {img_path}")
            else:
                # Solo texto, sin imágenes
                await update.message.reply_text(assistant_message)
            
            # Actualizar estadísticas del usuario
            thread_manager.update_stats(user_id, has_image=bool(image_paths))
            
        except Exception as e:
            # Manejo de errores robusto con logging detallado
            error_msg = (
                f"❌ Ocurrió un error al procesar tu mensaje:\n\n"
                f"`{str(e)}`\n\n"
                f"Intenta de nuevo o usa /reset para reiniciar la conversación."
            )
            await update.message.reply_text(error_msg, parse_mode='Markdown')
            
            # Log estructurado del error
            log_error_with_context(
                logger,
                e,
                {
                    'user_id': user_id,
                    'message': user_message[:100],
                    'thread_id': thread_manager.threads.get(user_id, {}).thread_id if user_id in thread_manager.threads else 'N/A'
                }
            )
    
    def _extract_image_paths(self, message: str) -> list[str]:
        """Extrae rutas absolutas de imágenes del mensaje del agente.
        
        Busca múltiples patrones:
        - IMG_PATH: /path/to/image.png
        - (sandbox:/path/to/image.png)
        - /Users/.../images/*.png
        
        Args:
            message: Mensaje del asistente.
        
        Returns:
            Lista de rutas absolutas de imágenes encontradas.
        """
        image_paths = []
        
        # Patrón 1: IMG_PATH: /path/to/image.png
        pattern1 = r'IMG_PATH:\s*([/\w\-_.]+\.png)'
        matches1 = re.findall(pattern1, message)
        image_paths.extend(matches1)
        
        # Patrón 2: (sandbox:/path/to/image.png) o (/path/to/image.png)
        pattern2 = r'\((?:sandbox:)?([/\w\-_.]+/images/[\w\-_.]+\.png)\)'
        matches2 = re.findall(pattern2, message)
        image_paths.extend(matches2)
        
        # Patrón 3: Cualquier ruta absoluta que contenga /images/*.png
        pattern3 = r'(/[^\s\)]+/images/[\w\-_.]+\.png)'
        matches3 = re.findall(pattern3, message)
        for match in matches3:
            if match not in image_paths:  # Evitar duplicados
                image_paths.append(match)
        
        return image_paths
    
    def _remove_image_paths(self, message: str) -> str:
        """Remueve líneas con rutas de imágenes del mensaje.
        
        Args:
            message: Mensaje original.
        
        Returns:
            Mensaje limpio sin rutas de archivos.
        """
        lines = message.split('\n')
        clean_lines = [
            line for line in lines 
            if 'IMG_PATH:' not in line
            and '📁 Imagen:' not in line
        ]
        return '\n'.join(clean_lines).strip()
    
    def run(self):
        """Inicia el bot en modo polling.
        
        El bot se ejecutará continuamente hasta que se detenga
        con Ctrl+C o se reciba una señal de interrupción.
        """
        logger.info("🤖 ClimateGuajira Bot iniciado")
        logger.info(f"📊 Usuarios activos: {len(thread_manager.get_all_users())}")
        logger.info("🔄 Bot en modo polling - Esperando mensajes...")
        
        # Iniciar polling
        self.app.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True  # Ignorar mensajes pendientes al iniciar
        )


if __name__ == "__main__":
    bot = ClimateBot()
    bot.run()

