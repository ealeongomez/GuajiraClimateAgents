# GuajiraClimateAgents

An intelligent AI agent system for climate analysis and wind energy potential assessment in La Guajira, Colombia.

## 🌟 Overview

GuajiraClimateAgents is a GenAI-powered platform that combines Retrieval-Augmented Generation (RAG) with historical climate databases to provide insights about wind energy potential and climate patterns in La Guajira region. The system uses LangGraph agents to intelligently query both the Colombian Wind Atlas and real-time climate observations database.

## ✨ Features

- **🤖 Intelligent Agent**: LangGraph-powered agent that automatically selects the right tools to answer questions
- **📚 RAG System**: Query the Colombian Wind Atlas (Atlas Eólico de Colombia) using semantic search
- **📊 Climate Database**: Access historical climate data from 13 municipalities in La Guajira
- **🌬️ Wind Analysis**: Compare wind speeds, analyze seasonal patterns, and identify optimal hours for wind energy generation
- **⚡ Temporal Optimization**: Efficient queries using indexed temporal columns (year, month, day, hour)
- **🧠 LSTM Models**: Pre-trained models for wind speed forecasting (available for all municipalities)

## 🏗️ Project Structure

```
GuajiraClimateAgents/
├── config/                  # Configuration files (YAML)
├── src/                     # Source code
│   ├── agents/             # LangGraph agent implementation
│   │   └── climate_guajira/ # Main climate agent
│   ├── llm/                # LLM clients (Claude, GPT)
│   ├── prompt_engineering/ # Prompt templates and chains
│   └── utils/              # Utilities (vector store, caching, etc.)
├── data/                    # Data repository
│   ├── embeddings/         # Vector embeddings storage
│   ├── models/             # Trained LSTM models
│   ├── wind/               # Wind speed CSV files
│   └── pdfs/               # Source documents
├── examples/               # Example implementations
├── notebooks/              # Jupyter notebooks for analysis
└── handlers/               # Error and response handlers
```

## 🛠️ Technologies

- **LangChain & LangGraph**: Agent orchestration and tool calling
- **OpenAI GPT-4**: Language model for agent reasoning
- **ChromaDB**: Vector database for document embeddings
- **SQL Server**: Historical climate observations database
- **PyTorch**: LSTM models for time series forecasting
- **Python 3.11+**: Core programming language

## 📋 Requirements

- Python 3.11 or higher
- OpenAI API key
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
OPENAI_API_KEY=your_openai_key
DB_SERVER=your_sql_server
DB_PORT=1433
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=ClimateDB
```

4. (Optional) Set up the vector database:
Run the notebooks in `notebooks/` to create embeddings from the Wind Atlas PDF.

## 💻 Usage

### Interactive CLI Agent

Run the main agent in interactive mode:

```bash
python -m src.agents.climate_guajira
```

Or use the example console:

```bash
python examples/console_ClimateAgent.py
```

### Example Questions

The agent can answer questions like:

- "What is the wind energy potential in La Guajira?"
- "Compare wind speeds between Riohacha and Maicao"
- "What are the windiest hours in Uribia during January 2024?"
- "Show me monthly wind statistics for Manaure in 2023"
- "Which municipalities have the highest average wind speed?"

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

## 📓 Notebooks

Explore the `notebooks/` directory for:
- RAG system testing
- Model training and evaluation
- Database connectivity examples
- Data analysis and visualization

## 📦 LSTM Models

Pre-trained LSTM models are available in `data/models/LSTM/` for all 13 municipalities. These models were optimized using Optuna hyperparameter tuning.

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Eder Arley León Gómez**
- GitHub: [@ealeongomez](https://github.com/ealeongomez)

## 🤝 Contributing

This is a doctoral research project. For questions or collaborations, please open an issue on GitHub.

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

---

Built with ❤️ for renewable energy research in La Guajira, Colombia
