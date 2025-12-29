# GuajiraClimateAgents

An intelligent AI agent system for climate analysis and wind energy potential assessment in La Guajira, Colombia.

## 🌟 Overview

GuajiraClimateAgents is a GenAI-powered platform that combines Retrieval-Augmented Generation (RAG) with historical climate databases to provide insights about wind energy potential and climate patterns in La Guajira region. The system uses LangGraph agents to intelligently query both the Colombian Wind Atlas and real-time climate observations database.

## ✨ Features

### Core Agent Features
- **🤖 Intelligent Agent**: LangGraph-powered agent that automatically selects the right tools to answer questions
- **📚 RAG System**: Query the Colombian Wind Atlas (Atlas Eólico de Colombia) using semantic search
- **📊 Climate Database**: Access historical climate data (2015-2025) from 13 municipalities in La Guajira
- **🌬️ Wind Analysis**: Compare wind speeds, analyze seasonal patterns, and identify optimal hours for wind energy generation
- **⚡ Temporal Optimization**: Efficient queries using indexed temporal columns (year, month, day, hour)
- **🧠 LSTM Models**: Pre-trained models for wind speed forecasting (available for all municipalities)

### Telegram Bot Features
- **💬 Interactive Interface**: Production-ready Telegram bot for easy access
- **👤 Per-User Sessions**: Independent conversation history for each user
- **📈 Image Generation**: Automatic chart and graph creation sent directly to Telegram
- **💾 State Persistence**: Conversation history saved with LangGraph checkpointing
- **📊 Usage Statistics**: Track messages, images, and activity per user
- **🔄 Session Management**: Commands to reset conversations and view stats

## 🏗️ Project Structure

```
GuajiraClimateAgents/
├── config/                  # Configuration files (YAML)
├── src/                     # Source code
│   ├── agents/             # LangGraph agent implementation
│   │   └── climate_guajira/ # Main climate agent
│   ├── bot/                # Telegram bot implementation
│   │   ├── telegram_bot.py  # Main bot logic
│   │   ├── thread_manager.py # Per-user session management
│   │   ├── image_handler.py  # Image storage and cleanup
│   │   └── checkpointer.py   # State persistence
│   ├── llm/                # LLM clients (Claude, GPT)
│   ├── prompt_engineering/ # Prompt templates and chains
│   └── utils/              # Utilities (vector store, caching, etc.)
├── data/                    # Data repository
│   ├── embeddings/         # Vector embeddings storage
│   ├── models/             # Trained LSTM models
│   ├── wind/               # Wind speed CSV files
│   ├── pdfs/               # Source documents
│   ├── user_images/        # User-specific generated images
│   └── checkpoints/        # Conversation state persistence
├── examples/               # Example implementations
├── notebooks/              # Jupyter notebooks for analysis
├── handlers/               # Error and response handlers
├── main_telegram.py        # Telegram bot entry point
└── pyproject.toml          # Project dependencies (uv)
```

## 🛠️ Technologies

- **LangChain & LangGraph**: Agent orchestration and tool calling
- **OpenAI GPT-4**: Language model for agent reasoning
- **python-telegram-bot**: Telegram bot framework
- **ChromaDB**: Vector database for document embeddings
- **SQL Server (pymssql)**: Historical climate observations database
- **PyTorch**: LSTM models for time series forecasting
- **Matplotlib & Pandas**: Data visualization and analysis
- **Python 3.11+**: Core programming language

## 📋 Requirements

- Python 3.11 or higher
- OpenAI API key
- Telegram Bot Token (for bot interface)
- SQL Server database with climate data
- uv (Python package manager)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/ealeongomez/GuajiraClimateAgents.git
cd GuajiraClimateAgents
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Create a `.env` file with your credentials:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_key

# SQL Server Configuration
DB_SERVER=your_sql_server
DB_PORT=1433
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=ClimateDB

# Telegram Bot (Optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
```

4. (Optional) Set up the vector database:
Run the notebooks in `notebooks/` to create embeddings from the Wind Atlas PDF.

## 💻 Usage

### Option 1: Telegram Bot (Recommended)

The easiest way to interact with the agent is through Telegram:

1. Create a bot with [@BotFather](https://t.me/botfather) and get your token
2. Add the token to your `.env` file
3. Start the bot:

```bash
python main_telegram.py
```

4. Find your bot on Telegram and start chatting!

**Available Commands:**
- `/start` - Initialize the bot and create your session
- `/help` - View examples and available features
- `/reset` - Reset your conversation history
- `/stats` - View your usage statistics

### Option 2: Interactive CLI

Run the agent directly in your terminal:

```bash
python -m src.agents.climate_guajira
```

Or use the console example:

```bash
python examples/console_ClimateAgent.py
```

### Example Questions

The agent can answer questions like:

**Climate Data Queries:**
- "What is the wind energy potential in La Guajira?"
- "Compare wind speeds between Riohacha and Maicao"
- "What are the windiest hours in Uribia during January 2024?"
- "Show me monthly wind statistics for Manaure in 2023"
- "Which municipalities have the highest average wind speed?"

**With Visualizations (works on Telegram and CLI):**
- "Create a graph of wind speed in Riohacha for January 2025"
- "Show me hourly patterns in Uribia for December 2024"
- "Compare visually Maicao, Riohacha and Uribia"
- "Plot wind speed vs temperature for Maicao"

## 🔧 Available Tools

### RAG Tools (Wind Atlas)
- `consultar_atlas_eolico`: Query the Colombian Wind Atlas
- `buscar_documentos`: Search for specific documents in the atlas

### Database Tools (Historical Data)
- `obtener_estadisticas_municipio`: Get climate statistics for a municipality
- `comparar_municipios_viento`: Compare wind speeds between municipalities
- `listar_municipios_disponibles`: List all available municipalities
- `obtener_estadisticas_por_mes`: Get monthly statistics for a specific year
- `obtener_estadisticas_por_hora`: Analyze hourly patterns
- `comparar_anios`: Compare statistics between two years

## 🗺️ Available Municipalities

Albania, Barrancas, Distracción, El Molino, Fonseca, Hatonuevo, La Jagua del Pilar, Maicao, Manaure, Mingueo, Riohacha, San Juan del Cesar, Uribia

## 🤖 Telegram Bot Architecture

The Telegram bot implementation (`src/bot/`) provides a production-ready interface with advanced features:

### Components

- **`telegram_bot.py`**: Main bot logic with command and message handlers
- **`thread_manager.py`**: Manages independent conversation threads per user
- **`image_handler.py`**: Handles image storage, organization, and cleanup
- **`checkpointer.py`**: SQLite-based state persistence for conversation history

### Features

1. **Per-User Sessions**: Each user has an independent conversation thread with full history
2. **Persistent State**: Conversations are saved and restored using LangGraph checkpointing
3. **Image Management**: Generated charts are automatically sent to users and saved
4. **Automatic Cleanup**: Old images are periodically removed to save space
5. **Usage Statistics**: Track messages, images, and activity per user
6. **Error Handling**: Robust error management with detailed logging

### Data Storage

- **Conversation State**: `data/checkpoints/telegram_checkpoints.db` (SQLite)
- **User Images**: `data/user_images/{user_id}/` (organized by user)

## 📓 Notebooks

Explore the `notebooks/` directory for:
- RAG system testing and experimentation
- Model training and evaluation
- Database connectivity examples
- Data analysis and visualization
- LSTM model performance analysis

## 📦 LSTM Models

Pre-trained LSTM models are available in `data/models/LSTM/` for all 13 municipalities. These models were optimized using Optuna hyperparameter tuning and can forecast wind speeds based on historical patterns.

## 🚀 Deployment

### Local Development

```bash
# CLI Mode
python -m src.agents.climate_guajira

# Telegram Bot Mode
python main_telegram.py
```

### Production Deployment

The bot can be deployed on any server with Python 3.11+. Recommended options:

1. **VPS/Cloud Server** (DigitalOcean, AWS, Azure)
   - Install dependencies with `uv sync`
   - Configure `.env` with production credentials
   - Run with `python main_telegram.py`
   - Use `systemd` or `supervisor` for process management

2. **Docker** (Coming Soon)
   - Dockerfile included in project structure
   - Container-based deployment for easy scaling

### Environment Variables

Required for production:
- `OPENAI_API_KEY`: Your OpenAI API key
- `TELEGRAM_BOT_TOKEN`: Bot token from @BotFather
- `DB_SERVER`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`: SQL Server connection

### Monitoring

- Logs are written to console (stdout/stderr)
- Consider using `systemd` journal or log aggregation tools
- Monitor API usage to control costs
- Track conversation state in SQLite checkpointer

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Eder Arley León Gómez**
- GitHub: [@ealeongomez](https://github.com/ealeongomez)

## 🤝 Contributing

This is a doctoral research project. For questions or collaborations, please open an issue on GitHub.

## 🔧 Troubleshooting

### Telegram Bot Issues

**Bot doesn't respond:**
- Verify `TELEGRAM_BOT_TOKEN` is correctly set in `.env`
- Check that the bot is running (`python main_telegram.py`)
- Use `/reset` to restart your conversation

**Database connection errors:**
- Verify SQL Server credentials in `.env`
- Ensure SQL Server is accessible from your network
- Check that `climate_observations` table exists

**Images not being sent:**
- Verify write permissions in `data/user_images/` directory
- Check matplotlib installation: `uv sync`
- Review logs for specific error messages

### General Issues

**Import errors:**
- Run `uv sync` to install all dependencies
- Verify Python version: `python --version` (must be 3.11+)
- Check that you're in the correct virtual environment

**OpenAI API errors:**
- Verify `OPENAI_API_KEY` is valid
- Check your OpenAI account has credits
- Monitor rate limits and quotas

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{leon2024guajiraclimateagents,
  author = {León Gómez, Eder Arley},
  title = {GuajiraClimateAgents: AI Agents for Climate and Wind Energy Analysis},
  year = {2024},
  url = {https://github.com/ealeongomez/GuajiraClimateAgents}
}
```

## 🌟 Acknowledgments

- **Colombian Wind Atlas**: Source of wind energy potential data
- **Open-Meteo**: Historical climate data provider
- **LangChain/LangGraph**: Agent orchestration framework

---

Built with ❤️ for renewable energy research in La Guajira, Colombia
