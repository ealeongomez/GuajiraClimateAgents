# GuajiraClimateAgents

An intelligent AI agent system for climate analysis and wind energy potential assessment in La Guajira, Colombia.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-green.svg)](https://langchain.com/)

## 📑 Table of Contents

- [Overview](#-overview)
- [Quick Start](#-quick-start)
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Technologies](#-technologies)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Available Tools](#-available-tools)
- [Available Municipalities](#-available-municipalities)
- [Telegram Bot Architecture](#-telegram-bot-architecture)
- [Automatic Database Updates](#-automatic-database-updates)
- [Wind Speed Forecasting](#-wind-speed-forecasting)
- [Notebooks](#-notebooks)
- [LSTM Forecasting System](#-lstm-forecasting-system)
- [Security Features](#-security-features)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Additional Documentation](#-additional-documentation)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

## 🌟 Overview

GuajiraClimateAgents is a GenAI-powered platform that combines Retrieval-Augmented Generation (RAG) with historical climate databases to provide insights about wind energy potential and climate patterns in La Guajira region. The system uses LangGraph agents to intelligently query both the Colombian Wind Atlas and real-time climate observations database.

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ealeongomez/GuajiraClimateAgents.git
cd GuajiraClimateAgents

# 2. Install dependencies
uv sync

# 3. Configure environment variables
cp .env.example .env  # Edit with your credentials

# 4. Start the Telegram bot
python main_telegram.py

# Or run the CLI agent
python -m src.agents.climate_guajira
```

## 🏛️ System Architecture

The GuajiraClimateAgents system is built with a modular architecture that integrates multiple AI technologies:

<div align="center">
  <img src="docs/chatbot-doctorade.svg" alt="System Architecture Diagram" width="100%">
  <p><em>Complete system architecture showing data flow and component interactions</em></p>
</div>

### Architecture Overview

The system consists of several interconnected layers:

1. **User Interfaces Layer**
   - 💬 Telegram Bot (primary interface)
   - 🖥️ CLI Console (development & testing)
   - 📊 LangGraph Studio (visual debugging)

2. **Agent Layer (LangGraph)**
   - 🤖 Intelligent agent with ReAct pattern
   - 🔒 Security prompt with injection protection
   - 🧠 Tool selection and orchestration
   - 💾 State management with checkpointing

3. **Tools & Services Layer**
   - 📚 RAG Tools (Wind Atlas queries)
   - 📊 Database Tools (historical data queries)
   - 📈 Visualization Tools (chart generation)
   - 🔄 Update Services (data synchronization)

4. **Data Layer**
   - 🗄️ SQL Server (Climate observations 2015-2025 + forecasts)
   - 📦 ChromaDB (Vector embeddings for RAG)
   - 🧠 PyTorch Models (LSTM forecasting - 13 trained models)
   - 💾 SQLite (Conversation checkpoints)

5. **External Services**
   - 🌐 Open-Meteo API (real-time climate data)
   - 🤖 OpenAI API (GPT-4 & embeddings)
   - 📡 Telegram API (bot communication)

### Data Flow

```
User Query (Telegram/CLI)
    ↓
LangGraph Agent
    ↓
Tool Selection & Execution
    ↓
├─→ RAG System → ChromaDB → Wind Atlas Documents
├─→ SQL Queries → SQL Server → Climate Data (Historical)
├─→ SQL Queries → SQL Server → Forecast Data (Predictions)
├─→ Visualization → Matplotlib → Charts/Graphs
└─→ LSTM Models → PyTorch → Real-time Forecasts
    ↓
Response Generation (GPT-4)
    ↓
User (with text + images + forecasts)
```

## ✨ Features

### Core Agent Features
- **🤖 Intelligent Agent**: LangGraph-powered agent that automatically selects the right tools to answer questions
- **📚 RAG System**: Query the Colombian Wind Atlas (Atlas Eólico de Colombia) using semantic search
- **📊 Climate Database**: Access historical climate data (2015-2025) from 13 municipalities in La Guajira
- **🌬️ Wind Analysis**: Compare wind speeds, analyze seasonal patterns, and identify optimal hours for wind energy generation
- **⚡ Temporal Optimization**: Efficient queries using indexed temporal columns (year, month, day, hour)
- **🔮 LSTM Forecasting**: Real-time wind speed predictions (24-hour ahead forecasts)
- **📈 Data Visualization**: Automatic generation of charts, graphs, and polar plots
- **🔒 Security**: Built-in protection against prompt injection attacks
- **🔄 Auto-Update System**: Automated database updates from Open-Meteo API

### Telegram Bot Features
- **💬 Interactive Interface**: Production-ready Telegram bot for easy access
- **👤 Per-User Sessions**: Independent conversation history for each user
- **📈 Image Generation**: Automatic chart and graph creation sent directly to Telegram
- **🔮 Forecast Delivery**: Real-time wind predictions with visual graphs
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
│   │       ├── tools.py      # Agent tools (RAG, DB, viz, forecast)
│   │       ├── prompts.py    # System prompts with security
│   │       ├── graph.py      # LangGraph agent definition
│   │       ├── state.py      # Agent state management
│   │       └── configuration.py # Agent configuration
│   ├── bot/                # Telegram bot implementation
│   │   ├── telegram_bot.py  # Main bot logic
│   │   ├── thread_manager.py # Per-user session management
│   │   ├── image_handler.py  # Image storage and cleanup
│   │   └── checkpointer.py   # State persistence
│   ├── models/             # Machine Learning models
│   │   └── forecast_generator.py # LSTM forecast generator
│   ├── scheduler/          # Automatic updates
│   │   └── update_scheduler.py # Cron-based scheduler
│   ├── llm/                # LLM clients (Claude, GPT)
│   ├── prompt_engineering/ # Prompt templates and chains
│   └── utils/              # Utilities
│       ├── vector_store.py  # ChromaDB interface
│       ├── db_updater.py    # Climate data update system
│       ├── forecast_db_updater.py # Forecast data updater
│       ├── climate_data.py  # Open-Meteo API client
│       └── logger.py        # Logging configuration
├── data/                    # Data repository
│   ├── embeddings/         # Vector embeddings storage (ChromaDB)
│   ├── models/             # Trained LSTM models
│   │   ├── LSTM/          # Final trained models (.pt)
│   │   └── optuna-LSTM/   # Hyperparameter optimization results
│   ├── wind/               # Historical wind CSV files (2015-2025)
│   ├── pdfs/               # Source documents (Wind Atlas)
│   ├── user_images/        # User-specific generated images
│   ├── checkpoints/        # Conversation state persistence
│   ├── outputs/            # Training results and exports
│   └── cache/              # API response caching
├── docs/                   # Documentation
│   ├── DATABASE_UPDATE_GUIDE.md # Complete update system guide
│   └── ORGANIZATION_SUMMARY.md  # System architecture overview
├── scripts/                # Maintenance scripts
│   ├── update_db_simple.py # Manual database update script
│   └── update_db.sh        # Cron-compatible bash script
├── examples/               # Example implementations
│   ├── console_ClimateAgent.py # CLI agent example
│   ├── console_SimpleRAG.py    # Basic RAG example
│   └── query_forecast.py       # Forecast query examples
├── notebooks/              # Jupyter notebooks (13 notebooks)
├── logs/                   # Application logs
├── main_telegram.py        # Telegram bot entry point
├── main.py                 # Alternative entry point
├── langgraph.json          # LangGraph Studio configuration
└── pyproject.toml          # Project dependencies (uv)
```

## 🛠️ Technologies

### AI & Agent Framework
- **LangChain & LangGraph**: Agent orchestration and tool calling
- **OpenAI GPT-4**: Language model for agent reasoning
- **LangGraph Studio**: Visual debugging and development (supported via `langgraph.json`)

### Data & Storage
- **ChromaDB**: Vector database for document embeddings
- **SQL Server (pymssql)**: Historical climate observations database (2015-2025)
- **SQLite**: State persistence for conversation checkpointing

### Machine Learning
- **PyTorch**: LSTM models for time series forecasting
- **Optuna**: Hyperparameter optimization for LSTM training

### Interfaces & Communication
- **python-telegram-bot**: Production-ready Telegram bot framework
- **Matplotlib & Pandas**: Data visualization and analysis

### Automation & Scheduling
- **APScheduler**: Automatic database updates with cron expressions
- **Open-Meteo API**: Real-time climate data source

### Development Tools
- **Python 3.11+**: Core programming language
- **uv**: Fast Python package manager
- **python-dotenv**: Environment variable management

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

### Option 3: LangGraph Studio (Development)

For visual debugging and development, use LangGraph Studio:

```bash
# Install LangGraph CLI (if not already installed)
pip install langgraph-cli[inmem]

# Start LangGraph Studio
langgraph up

# Open browser at http://localhost:8000
```

The `langgraph.json` configuration file provides two graphs:
- `agent` - Generic agent from `src/agents/main.py`
- `climate_guajira` - Main climate agent from `src/agents/climate_guajira/graph.py`

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

**Wind Speed Forecasting:**
- "What is the wind forecast for Riohacha?"
- "Show me the prediction for Maicao for the next 24 hours"
- "Generate a forecast graph for Uribia"
- "Compare historical vs predicted wind for Manaure"

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

### Visualization Tools (Charts & Graphs)
- `graficar_serie_temporal_municipio`: Time series plot of wind speed for a municipality
- `graficar_comparacion_municipios`: Visual comparison between multiple municipalities
- `graficar_patron_horario`: 24-hour wind pattern (polar plot)
- `graficar_viento_temperatura`: Wind speed vs temperature correlation plot

### Forecasting Tools (LSTM Predictions)
- `obtener_prediccion_municipio`: Get the latest 24-hour wind speed forecast for a municipality
- `graficar_prediccion_municipio`: Generate forecast visualization with 48h historical + 24h prediction

All charts are automatically generated and sent to users via Telegram or saved to the `images/` directory.

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

## 🔄 Automatic Database Updates

The system includes an automated update mechanism to keep climate data current:

### Components

- **`src/utils/db_updater.py`**: Main updater module with `ClimateDBUpdater` class
- **`src/scheduler/update_scheduler.py`**: Scheduler for periodic automatic updates
- **`scripts/update_db_simple.py`**: Simple script for manual updates
- **`scripts/update_db.sh`**: Bash script for cron integration

### Features

1. **Incremental Updates**: Only downloads new data since last update
2. **Duplicate Prevention**: Uses SQL MERGE to avoid duplicate records
3. **Batch Processing**: Efficient bulk inserts in batches of 1000 records
4. **Multi-Municipality**: Updates all 13 municipalities automatically
5. **Error Recovery**: Robust error handling with detailed logging
6. **Flexible Scheduling**: Support for custom cron expressions

### Update Methods

**Option 1: Cron (Recommended for Production)**
```bash
# Edit crontab
crontab -e

# Add line to update every hour at :05
5 * * * * /path/to/GuajiraClimateAgents/scripts/update_db.sh
```

**Option 2: Python Scheduler**
```bash
# Run with default schedule (every hour at :05)
python src/scheduler/update_scheduler.py

# Custom schedule (every 30 minutes)
python src/scheduler/update_scheduler.py --cron "*/30 * * * *"

# Run once and exit
python src/scheduler/update_scheduler.py --once
```

**Option 3: Manual Update**
```bash
# Simple Python script
python scripts/update_db_simple.py
```

**Option 4: Programmatic**
```python
from src.utils.db_updater import update_database_from_env

results = update_database_from_env()
print(f"Inserted: {results['total_inserted']} records")
```

### Logging

All updates are logged to:
- `logs/scheduler.log` - Scheduler activity
- `logs/errors.log` - Error tracking
- `logs/cron_updates_YYYYMM.log` - Monthly cron logs

## 🔮 Wind Speed Forecasting

The system features an advanced LSTM-based forecasting pipeline that provides 24-hour ahead wind speed predictions.

### How It Works

1. **Historical Data Retrieval**
   - Fetches last 48 hours of wind speed data from SQL Server
   - Data is normalized and prepared for the model

2. **Model Prediction**
   - Loads pre-trained LSTM model for the municipality
   - Generates 24 hourly predictions
   - Models use Random Fourier Features (DenseRFF) for enhanced learning

3. **Database Storage**
   - Predictions stored in `Forecast` table
   - Includes both input (48h) and output (24h) arrays
   - Automatic cleanup of old forecasts

4. **User Access**
   - Query predictions via Telegram bot or CLI
   - Get text summaries with statistics
   - Visual graphs showing historical vs predicted values

### Forecast Features

- **🎯 Accuracy**: Models optimized with Optuna hyperparameter tuning
- **⚡ Speed**: Batch processing for all 13 municipalities
- **📊 Rich Output**: Min/max/average statistics for historical and predicted values
- **📈 Visualization**: Graphs with color-coded historical and forecast periods
- **🔄 Auto-Update**: Can be automated with cron jobs or schedulers

### Example Forecast Query

**Text Query:**
```
User: "What is the wind forecast for Riohacha?"

Agent Response:
🔮 PREDICCIÓN DE VIENTO - RIOHACHA

📅 Predicción generada: 2025-01-05 14:30:00
📅 Primera hora predicha: 2025-01-05 15:00:00

📊 HISTÓRICO (últimas 48 horas):
   • Mínimo: 3.2 m/s
   • Máximo: 8.5 m/s
   • Promedio: 5.8 m/s

🔮 PREDICCIÓN (próximas 24 horas):
   • Mínimo: 4.1 m/s
   • Máximo: 7.9 m/s
   • Promedio: 6.2 m/s
```

**Visual Query:**
```
User: "Show me a forecast graph for Uribia"

Agent: [Sends image with 48h historical + 24h predicted wind speeds]
```

### Generating Forecasts

**Interactive (Notebook):**
```bash
jupyter notebook notebooks/14_UpdateForecast.ipynb
```

**Programmatic:**
```python
from src.models.forecast_generator import ForecastGenerator
from src.utils.forecast_db_updater import ForecastDBUpdater

# Generate forecasts
generator = ForecastGenerator()
forecasts = generator.generate_all_forecasts()

# Store in database
updater = ForecastDBUpdater(conn)
updater.insert_forecasts(forecasts)
```

**Via Examples:**
```bash
python examples/query_forecast.py
```

## 📓 Notebooks

Explore the `notebooks/` directory for interactive development and analysis:

1. **`01_notebook.ipynb`** - Initial project exploration
2. **`02_RAG.ipynb`** - RAG system testing with OpenAI embeddings
3. **`03_RAG_Ollama.ipynb`** - RAG system with local Ollama models
4. **`04_download_data.ipynb`** - Climate data download from Open-Meteo
5. **`05_training_colab.ipynb`** - LSTM model training (Google Colab)
6. **`06_performance_model.ipynb`** - Model evaluation and metrics
7. **`07_connectDB.ipynb`** - Database connection testing
8. **`08_checkDB.ipynb`** - Database integrity checks
9. **`09_SQLAgent.ipynb`** - SQL agent experimentation
10. **`10_restructure_datetime.ipynb`** - Database schema optimization
11. **`11_UpdateDB.ipynb`** - Interactive database updates
12. **`12_CheckDB.ipynb`** - Database status verification
13. **`13_Forecast.ipynb`** - Wind speed forecasting with LSTM models
14. **`14_UpdateForecast.ipynb`** - Generate and update forecasts in database
15. **`15_CheckForecast.ipynb`** - Verify and analyze forecast accuracy

## 📦 LSTM Forecasting System

The system includes a complete wind speed forecasting pipeline using deep learning LSTM models.

### Pre-trained Models
Pre-trained LSTM models are available in `data/models/LSTM/` for all 13 municipalities:
- **Training Data**: 2015-2025 historical wind speed data (hourly)
- **Optimization**: Optuna hyperparameter tuning (results in `data/models/optuna-LSTM/`)
- **Architecture**: LSTM with Random Fourier Features (DenseRFF)
- **Format**: PyTorch (.pt) model files
- **Coverage**: All 13 La Guajira municipalities

### Forecast Pipeline

**1. Model Training** (`notebooks/13_Forecast.ipynb`)
- Hyperparameter optimization with Optuna
- Training on historical data
- Model evaluation and validation

**2. Forecast Generation** (`src/models/forecast_generator.py`)
- Loads trained LSTM models
- Uses last 48 hours of historical data
- Generates 24-hour ahead predictions
- Batch processing for all municipalities

**3. Database Storage** (`src/utils/forecast_db_updater.py`)
- Stores predictions in SQL Server `Forecast` table
- Automatic cleanup of old forecasts
- JSON format for input/output arrays

**4. Forecast Queries** (Agent Tools)
- `obtener_prediccion_municipio`: Get text forecast
- `graficar_prediccion_municipio`: Visual forecast with historical context

### Forecast Structure

Each forecast contains:
- **Input**: 48 hourly values (recent historical data)
- **Output**: 24 hourly predictions (next day forecast)
- **Metadata**: Municipality, start datetime, creation timestamp

### Example Usage

```python
# Generate forecasts for all municipalities
from src.models.forecast_generator import ForecastGenerator

generator = ForecastGenerator()
forecasts = generator.generate_all_forecasts()
```

```python
# Query via agent
"What is the wind forecast for Riohacha?"
"Show me a forecast graph for Uribia"
```

### Notebooks
- **`13_Forecast.ipynb`** - Model training and evaluation
- **`14_UpdateForecast.ipynb`** - Generate and store forecasts
- **`15_CheckForecast.ipynb`** - Verify forecast accuracy

## 🔒 Security Features

The agent includes multiple security layers:

### Prompt Injection Protection
- System prompts with maximum priority rules
- Rejection of attempts to modify agent behavior
- Filtering of malicious instructions in user input
- Separation of data sources from instructions

### Best Practices
- Environment variables for sensitive credentials (never hardcoded)
- SQL injection prevention through parameterized queries
- Input validation and sanitization
- Comprehensive error handling without exposing internal details

### Security Rules
The agent is designed to:
- Reject role-playing requests outside its domain
- Ignore instructions embedded in documents or tool outputs
- Maintain focus on climate and wind energy topics only
- Never expose system prompts or internal logic

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

### Monitoring & Logging

The system includes comprehensive logging:

**Log Files:**
- `logs/scheduler.log` - Database update scheduler activity
- `logs/errors.log` - Error tracking and debugging
- `logs/telegram_bot.log` - Telegram bot operations
- `logs/cron_updates_YYYYMM.log` - Monthly cron execution logs

**Monitoring Commands:**
```bash
# Watch scheduler logs in real-time
tail -f logs/scheduler.log

# Check database update status
python scripts/update_db_simple.py --status

# View conversation checkpoints
sqlite3 data/checkpoints/telegram_checkpoints.db "SELECT * FROM checkpoints;"

# Monitor API usage
grep "API" logs/*.log | wc -l
```

**Metrics to Track:**
- API usage and costs (OpenAI + Open-Meteo)
- Database update frequency and success rate
- User activity (messages, image requests)
- Conversation state size
- Image storage usage

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
- Check logs: `tail -f logs/telegram_bot.log`

**Database connection errors:**
- Verify SQL Server credentials in `.env`
- Ensure SQL Server is accessible from your network
- Check that `climate_observations` table exists
- Test connection: `python notebooks/07_connectDB.ipynb`

**Images not being sent:**
- Verify write permissions in `data/user_images/` directory
- Check matplotlib installation: `uv sync`
- Ensure `src/agents/climate_guajira/images/` exists
- Review logs for specific error messages

**Conversation history not persisting:**
- Check `data/checkpoints/telegram_checkpoints.db` exists
- Verify SQLite permissions
- Check checkpointer initialization in logs

### Database Update Issues

**Updates failing:**
- Verify internet connection to Open-Meteo API
- Check database credentials and connectivity
- Review `logs/scheduler.log` for specific errors
- Test manual update: `python scripts/update_db_simple.py`

**Scheduler not running:**
- For cron: Check crontab with `crontab -l`
- Verify script has execute permissions: `chmod +x scripts/update_db.sh`
- Check cron logs: `grep CRON /var/log/syslog` (Linux)
- For Python scheduler: Ensure it's running in background

**Duplicate data:**
- The system uses MERGE to prevent duplicates automatically
- If duplicates exist, check database constraints
- Run database integrity check: `notebooks/12_CheckDB.ipynb`

### General Issues

**Import errors:**
- Run `uv sync` to install all dependencies
- Verify Python version: `python --version` (must be 3.11+)
- Check that you're in the correct virtual environment
- Clear Python cache: `find . -type d -name __pycache__ -exec rm -rf {} +`

**OpenAI API errors:**
- Verify `OPENAI_API_KEY` is valid and not expired
- Check your OpenAI account has credits
- Monitor rate limits and quotas
- Consider using caching to reduce API calls

**Vector store errors:**
- Ensure ChromaDB directory exists: `data/embeddings/`
- Regenerate embeddings if corrupted: `notebooks/02_RAG.ipynb`
- Check OpenAI embeddings API access

**LSTM model issues:**
- Verify model files exist in `data/models/LSTM/`
- Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`
- See training notebooks: `05_training_colab.ipynb`, `06_performance_model.ipynb`

**Forecast issues:**
- Check if `Forecast` table exists in database
- Verify forecasts are being generated: `notebooks/14_UpdateForecast.ipynb`
- Check forecast data: `python examples/query_forecast.py`
- Ensure LSTM models are loaded correctly
- Review forecast accuracy: `notebooks/15_CheckForecast.ipynb`

### Performance Issues

**Slow responses:**
- Check database query performance (add indexes if needed)
- Monitor API latency
- Consider caching frequently requested data
- Reduce `retrieval_k` for RAG queries

**High memory usage:**
- Clear old conversation checkpoints periodically
- Run image cleanup: Clean `data/user_images/` of old files
- Monitor ChromaDB size

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

## 📖 Additional Documentation

For more detailed information, see:

### Documentation Files
- **`docs/chatbot-doctorade.svg`** - System architecture diagram (visual overview)
- **`docs/DATABASE_UPDATE_GUIDE.md`** - Complete guide to the automatic update system (500+ lines)
- **`docs/ORGANIZATION_SUMMARY.md`** - System architecture and organization overview
- **`scripts/README.md`** - Scripts usage and cron configuration guide
- **Inline Docstrings** - All modules include comprehensive docstrings

### Key Features Documentation

- **Agent Tools**: See docstrings in `src/agents/climate_guajira/tools.py`
- **Security**: Review `SYSTEM_PROMPT` in `src/agents/climate_guajira/prompts.py`
- **Database Schema**: Check `notebooks/08_checkDB.ipynb` and `10_restructure_datetime.ipynb`
- **Update System**: Read `docs/DATABASE_UPDATE_GUIDE.md`

## 🌟 Acknowledgments

### Data Sources
- **[Colombian Wind Atlas](https://atlas.ideam.gov.co/visorAtlasVientos.html)** - Wind energy potential data for Colombia
- **[Open-Meteo](https://open-meteo.com/)** - Historical climate data API (2015-2025)
- **IDEAM** - Colombian Institute of Hydrology, Meteorology and Environmental Studies

### Technologies & Frameworks
- **[LangChain](https://langchain.com/)** - LLM application framework
- **[LangGraph](https://github.com/langchain-ai/langgraph)** - Agent orchestration and state management
- **[OpenAI](https://openai.com/)** - GPT-4 language model and embeddings
- **[ChromaDB](https://www.trychroma.com/)** - Vector database for embeddings
- **[python-telegram-bot](https://python-telegram-bot.org/)** - Telegram Bot API wrapper

### Research Context
This project is part of doctoral research on **renewable energy systems and climate analysis** in La Guajira, Colombia, focusing on the application of Generative AI and Large Language Models to climate data analysis and decision support systems.

---

Built with ❤️ for renewable energy research in La Guajira, Colombia
