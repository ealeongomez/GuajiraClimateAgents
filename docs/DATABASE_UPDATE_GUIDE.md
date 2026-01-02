# 🔄 Climate Database Update Guide

**Project:** GuajiraClimateAgents  
**Author:** Eder Arley León Gómez  
**Last Updated:** January 2026

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Main Modules](#main-modules)
4. [Update Methods](#update-methods)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This system automates the update of the `ClimateDB` database with climate data from La Guajira, Colombia, obtained from the Open-Meteo API.

### ✨ Key Features

- ✅ **Incremental Updates**: Only downloads new data
- ✅ **No Duplicates**: Uses SQL `MERGE` to avoid duplicate records
- ✅ **Efficient**: Batch processing of 1000 records
- ✅ **Robust**: Per-municipality error handling
- ✅ **Flexible**: Multiple execution methods
- ✅ **Monitored**: Complete logging system

---

## Architecture

```
GuajiraClimateAgents/
├── src/
│   ├── utils/
│   │   ├── db_updater.py          # ⭐ Main update module
│   │   ├── climate_data.py        # API data download
│   │   └── logger.py              # Logging system
│   └── scheduler/
│       ├── __init__.py
│       └── update_scheduler.py    # ⭐ Automatic scheduler
├── scripts/
│   ├── update_db.sh               # ⭐ Bash script for cron
│   ├── update_db_simple.py        # Simple Python script
│   └── README.md                  # Scripts documentation
├── notebooks/
│   └── 11_UpdateDB.ipynb          # Interactive notebook
└── logs/
    ├── scheduler.log              # Scheduler logs
    ├── errors.log                 # Error logs
    └── cron_updates_YYYYMM.log    # Cron execution logs
```

---

## Main Modules

### 1. `src/utils/db_updater.py`

Main module containing the `ClimateDBUpdater` class.

#### `ClimateDBUpdater` Class

```python
class ClimateDBUpdater:
    """Database update handler."""
    
    def __init__(self, server, database, user, password, port="1433")
    def connect() -> None
    def close() -> None
    def get_last_dates() -> Dict[str, datetime]
    def bulk_insert_climate_data(df, municipio) -> int
    def update_municipality(municipio, last_date=None) -> Dict
    def update_all_municipalities(end_date=None) -> Dict
    def get_database_status() -> pd.DataFrame
```

#### Convenience Function

```python
def update_database_from_env() -> Dict:
    """Updates using environment variables."""
```

### 2. `src/scheduler/update_scheduler.py`

Scheduler for automatic periodic executions.

#### Main Functions

```python
def update_database_task() -> Dict
    """Update task (called by the scheduler)."""

def run_scheduler(cron_expression="5 * * * *", run_immediately=False)
    """Runs the scheduler with cron expression."""
```

### 3. `src/utils/climate_data.py`

Downloads data from the Open-Meteo API (already existing).

---

## Update Methods

### Method 1: Cron (⭐ RECOMMENDED)

**Advantages:**
- Native to the operating system
- Very reliable
- Doesn't require a constantly running process
- Month-separated logs

**Setup:**

1. **Edit crontab:**
```bash
crontab -e
```

2. **Add task:**
```cron
# Every hour at minute 5
5 * * * * /Users/guane/Documentos/Doctorate/GuajiraClimateAgents/scripts/update_db.sh

# Every 6 hours
5 */6 * * * /Users/guane/Documentos/Doctorate/GuajiraClimateAgents/scripts/update_db.sh

# Daily at 6 AM
5 6 * * * /Users/guane/Documentos/Doctorate/GuajiraClimateAgents/scripts/update_db.sh
```

3. **Verify:**
```bash
crontab -l
```

### Method 2: Python Scheduler with APScheduler

**Advantages:**
- Full control from Python
- Detailed integrated logs
- Easy to modify at runtime

**Usage:**

```bash
# Run scheduler (every hour by default)
python src/scheduler/update_scheduler.py

# With custom cron expression
python src/scheduler/update_scheduler.py --cron "*/30 * * * *"

# With immediate initial update
python src/scheduler/update_scheduler.py --run-now

# Single update (no scheduler)
python src/scheduler/update_scheduler.py --once
```

**Keep running in background:**

```bash
# Option 1: nohup
nohup python src/scheduler/update_scheduler.py > logs/scheduler_output.log 2>&1 &

# Option 2: screen
screen -S climate_scheduler
python src/scheduler/update_scheduler.py
# Ctrl+A, D to detach
# screen -r climate_scheduler to reconnect
```

### Method 3: Simple Script

For quick manual execution:

```bash
# Bash
./scripts/update_db.sh

# Python
python scripts/update_db_simple.py
```

### Method 4: Interactive Notebook

For exploration and debugging:

```bash
jupyter notebook notebooks/11_UpdateDB.ipynb
```

### Method 5: Programmatic Usage

```python
from src.utils.db_updater import ClimateDBUpdater, update_database_from_env

# Option A: Using environment variables
results = update_database_from_env()

# Option B: With explicit parameters
with ClimateDBUpdater(
    server="localhost",
    database="ClimateDB",
    user="sa",
    password="password"
) as updater:
    # Update all
    results = updater.update_all_municipalities()
    print(f"Inserted: {results['total_inserted']} records")
    
    # Or update a specific one
    result = updater.update_municipality("riohacha")
    
    # View status
    status = updater.get_database_status()
    print(status)
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Database Configuration
DB_SERVER=localhost
DB_PORT=1433
DB_USER=sa
DB_PASSWORD=your_secure_password
DB_NAME=ClimateDB
```

### Dependencies Installation

```bash
# If using uv (recommended)
uv sync

# Or with pip
pip install apscheduler pymssql pandas python-dotenv pytz requests
```

---

## Usage Examples

### Example 1: Complete Manual Update

```python
import os
from dotenv import load_dotenv
from src.utils.db_updater import ClimateDBUpdater

load_dotenv()

with ClimateDBUpdater(
    server=os.getenv("DB_SERVER"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
) as updater:
    # View current status
    print("📊 Current status:")
    status = updater.get_database_status()
    print(status)
    
    # Update all
    print("\n🔄 Updating...")
    results = updater.update_all_municipalities()
    
    # Show results
    print(f"\n✅ Completed:")
    print(f"  📥 Downloaded: {results['total_downloaded']:,}")
    print(f"  💾 Inserted: {results['total_inserted']:,}")
    print(f"  ✅ Successful: {results['successful']}/{results['total']}")
```

### Example 2: Update Only One Municipality

```python
from src.utils.db_updater import ClimateDBUpdater

with ClimateDBUpdater(...) as updater:
    result = updater.update_municipality("riohacha")
    
    if result['status'] == 'success':
        print(f"✅ {result['municipio']}: {result['inserted']} records")
    else:
        print(f"❌ Error: {result['status']}")
```

### Example 3: Update with Specific Range

```python
from datetime import datetime, timedelta
from src.utils.db_updater import ClimateDBUpdater

# Update only the last month
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

with ClimateDBUpdater(...) as updater:
    result = updater.update_municipality(
        municipio="riohacha",
        last_date=start_date,
        end_date=end_date
    )
```

---

## Monitoring

### Log Locations

| File | Content |
|------|---------|
| `logs/scheduler.log` | General scheduler log |
| `logs/errors.log` | Errors only |
| `logs/cron_updates_YYYYMM.log` | Cron executions |
| `logs/telegram_bot.log` | Bot logs (separate) |

### Monitoring Commands

```bash
# View logs in real-time
tail -f logs/scheduler.log

# View last 50 lines
tail -50 logs/scheduler.log

# View errors only
tail -f logs/errors.log

# View current month cron logs
tail -f logs/cron_updates_$(date +%Y%m).log

# Search for errors in logs
grep "ERROR" logs/scheduler.log

# Check last successful execution
grep "UPDATE COMPLETED" logs/scheduler.log | tail -1
```

### Verify Cron Jobs

```bash
# View scheduled tasks
crontab -l

# View system logs (macOS)
log show --predicate 'process == "cron"' --last 1h --style syslog

# View system logs (Linux)
sudo grep CRON /var/log/syslog | tail -20
```

---

## Troubleshooting

### Problem 1: Script doesn't execute in cron

**Solution:**

1. Verify permissions:
```bash
chmod +x scripts/update_db.sh
ls -la scripts/update_db.sh
```

2. Verify that `.env` file exists:
```bash
ls -la .env
```

3. Test manually:
```bash
./scripts/update_db.sh
```

4. Check cron logs:
```bash
tail -50 logs/cron_updates_$(date +%Y%m).log
```

### Problem 2: Database connection error

**Symptoms:**
```
pymssql.OperationalError: (20009, b'...')
```

**Solution:**

1. Verify SQL Server is running:
```bash
# macOS/Linux
ps aux | grep sqlservr

# Or test connection
telnet localhost 1433
```

2. Verify credentials in `.env`:
```bash
cat .env | grep DB_
```

3. Test manual connection:
```python
import pymssql
conn = pymssql.connect(
    server='localhost',
    port='1433',
    user='sa',
    password='your_password',
    database='ClimateDB'
)
print("✅ Connected")
conn.close()
```

### Problem 3: No new data downloaded

**Symptoms:**
```
⚠️  No new data downloaded for municipality
```

**Explanation:**

This is **normal** if the database is already up to date. The system only downloads data from the last recorded date until now.

**Verify:**

```python
from src.utils.db_updater import ClimateDBUpdater

with ClimateDBUpdater(...) as updater:
    last_dates = updater.get_last_dates()
    print("Last dates:")
    for municipio, fecha in last_dates.items():
        print(f"  {municipio}: {fecha}")
```

### Problem 4: Module import error

**Symptoms:**
```
ModuleNotFoundError: No module named 'apscheduler'
```

**Solution:**

```bash
# Install dependencies
uv sync

# Or with pip
pip install apscheduler
```

### Problem 5: Scheduler stops by itself

**Solution:**

Use `nohup` or `screen` to keep it running:

```bash
# With nohup
nohup python src/scheduler/update_scheduler.py > logs/scheduler_output.log 2>&1 &

# Save PID
echo $! > scheduler.pid

# Stop when needed
kill $(cat scheduler.pid)
```

---

## 📚 References

- **Open-Meteo API:** https://open-meteo.com/
- **APScheduler Docs:** https://apscheduler.readthedocs.io/
- **Cron Syntax:** https://crontab.guru/

---

## 🔐 Security

### Recommendations

1. **Don't version the `.env` file**
   - Already in `.gitignore`
   - Never commit credentials

2. **Use secure passwords**
   - Minimum 12 characters
   - Combine uppercase, lowercase, numbers and symbols

3. **Limit access to logs**
   ```bash
   chmod 600 .env
   chmod 700 logs/
   ```

4. **Rotate logs periodically**
   - Cron logs already rotate monthly
   - Consider archiving old logs

---

## 📈 Next Steps

1. ✅ Configure automatic update with cron
2. ✅ Monitor logs the first week
3. ⏳ Configure email alerts on errors (optional)
4. ⏳ Implement automatic database backup (optional)
5. ⏳ Monitoring dashboard (optional)

---

**Questions or problems?** Check the logs in `logs/` or consult this guide.

