# DocuMind AI

Intelligent documentation assistant that transforms your Confluence pages into an AI-powered Q&A system with Slack integration.

## Features

• **Smart Search**: Vector similarity + keyword matching across Confluence pages  
• **AI Answers**: DeepSeek LLM provides contextual responses from your docs  
• **Cost Efficient**: Runs on AWS t3.micro Spot instances (~$3-5/month)  
• **Slack Integration**: Complete AI chatbot workflow integrated with team communication  
• **Simple Setup**: Pure Python, minimal dependencies  

## Infrastructure & Costs

**Recommended Setup:**
• **Compute**: AWS EC2 t3.micro Spot instance ($3-4/month)  
• **Storage**: 11 GB gp3 EBS volume ($1/month)  
• **Database**: MongoDB Atlas Free Tier (500 MB limit)  
• **AI**: DeepSeek API (token-based, very cheap ~$0.001/1K tokens)  

**Total estimated cost: $4-6/month**

⚠️ **Ingestion Note**: `ingest.py` can be CPU-intensive. If t3.micro throttles during ingestion, temporarily use t3.small or t3.medium, then scale back to t3.micro.  

## Prerequisites

• Ubuntu 24.04  
• Python 3.12  

**Required API Keys & Services:**
• **DeepSeek API Token** - Get from [DeepSeek Platform](https://platform.deepseek.com/)
• **Confluence API Token** - Generate from your Atlassian account  
• **MongoDB Database** - Free Atlas cluster or self-hosted
• **Slack Tokens** - App and Bot tokens (for Slack integration only)  

## Quick Start

1. **System Setup**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv git
```

2. **Clone & Setup**
```bash
git clone <your-repo-url>
cd confluence-qa
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. **Configuration**
```bash
cp .env.example .env
cp config.example.py config.py
# Edit .env with your credentials:
# - DEEPSEEK_API_KEY (required)
# - CONFLUENCE_BASE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN (required)
# - MONGODB_URI (required) 
# - SLACK_APP_TOKEN, SLACK_BOT_TOKEN (optional, for Slack bot)
```

4. **Ingest Data** (may take 1-5 minutes depends on content length)
```bash
python ingest.py
```

5. **Test System**
```bash
# Start API server in background
python serve.py &

# Test with curl
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"what is our deployment process"}'

# Or use CLI directly  
python ask.py "what is our deployment process"
```

## How It Works

**Workflow:** `slack_bot.py` → `serve.py` → `ask.py`

1. **Data Ingestion** (`ingest.py`): Fetches Confluence pages, chunks content, generates embeddings, stores in MongoDB
2. **API Server** (`serve.py`): Loads embeddings into memory, serves HTTP API for fast responses  
3. **Q&A Engine** (`ask.py`): Searches vectors, retrieves relevant context, queries DeepSeek LLM
4. **Slack Bot** (`slack_bot.py`): Listens for mentions, forwards questions to API server, returns answers

## Slack Integration

### Create a Slack App

1. Go to [Slack API – Your Apps](https://api.slack.com/apps)
2. Create new app → From scratch  
3. **Socket Mode**: Enable → Generate App Token (xapp-...)
   • Scope: `connections:write`
4. **OAuth & Permissions** → Bot Token Scopes:
   • `app_mentions:read`
   • `chat:write`
   • (optional) `channels:history`
5. **Event Subscriptions**:
   • Enable Events
   • Subscribe to `app_mention` event
6. Reinstall app to workspace
7. Copy Bot User OAuth Token (xoxb-...)

### Configure & Run
```bash
# Add tokens to .env
SLACK_APP_TOKEN=xapp-...
SLACK_BOT_TOKEN=xoxb-...

# Start bot
python slack_bot.py
```

## Production Deployment

### Systemd Services

**API Server** (`/etc/systemd/system/confluence-qa.service`):
```ini
[Unit]
Description=Confluence QA API Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/path/to/repo
EnvironmentFile=/path/to/repo/.env
ExecStart=/path/to/repo/.venv/bin/python serve.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

**Slack Bot** (`/etc/systemd/system/confluence-qa-slack.service`):
```ini
[Unit]
Description=Confluence QA Slack Bot
After=network.target confluence-qa.service
Requires=confluence-qa.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/path/to/repo
EnvironmentFile=/path/to/repo/.env
ExecStart=/path/to/repo/.venv/bin/python slack_bot.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### Service Commands
```bash
# Install services
sudo systemctl daemon-reload
sudo systemctl enable confluence-qa.service
sudo systemctl enable confluence-qa-slack.service

# Start services
sudo systemctl start confluence-qa.service
sudo systemctl start confluence-qa-slack.service

# Check status
sudo systemctl status confluence-qa.service
journalctl -u confluence-qa.service -f
```

## Configuration

All configuration is via environment variables in `.env`. See `.env.example` for required variables.

**Required Environment Variables:**
• `DEEPSEEK_API_KEY` - Your DeepSeek API token
• `CONFLUENCE_BASE_URL` - Your Confluence instance URL  
• `CONFLUENCE_USERNAME` - Your Confluence email
• `CONFLUENCE_API_TOKEN` - Confluence API token
• `MONGODB_URI` - MongoDB connection string
• `PAGE_IDS` - Comma-separated Confluence page IDs to ingest

**Optional (for Slack):**
• `SLACK_APP_TOKEN` - Slack app token (xapp-...)
• `SLACK_BOT_TOKEN` - Slack bot token (xoxb-...)

## Files

• `ingest.py` — Load Confluence pages into vector database  
• `ask.py` — CLI question answering  
• `serve.py` — HTTP API server (keeps embeddings warm)  
• `slack_bot.py` — Slack bot integration  
• `config.py` — Configuration (copy from config.example.py)  

## System Architecture
```bash
[Confluence Pages] → [ingest.py] → [MongoDB Vector DB]
                                                ↓
[Slack User] → [slack_bot.py] → [serve.py] → [ask.py] → [DeepSeek LLM]
                                                ↑
                                [Vector Search + Context]
```