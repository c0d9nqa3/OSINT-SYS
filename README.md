# OSINT Intelligence Collection System

## Project Overview

An **experimental open-source intelligence collection system** designed for research and educational purposes. This is a demo system that integrates various AI technologies to collect, verify, and analyze publicly available information.

**Current Status**: This is an experimental project. AI-related models are still under development, including psychological vulnerability analysis and social engineering research.

**Collaboration Welcome**: We welcome researchers, developers, and security professionals to collaborate on building advanced intelligence collection systems.

## Key Features

### Multi-Source Data Collection
- Search Engine Integration (Google Search API)
- Social Media Data Collection
- Intelligent Information Filtering
- Compliance Checking

### AI Identity Verification (Under Development)
- Multi-modal Verification
- Confidence Scoring
- Evidence Chain Tracking

### Social Network Analysis
- Relationship Discovery
- Network Visualization
- Community Detection

### Resume Organization
- Information Extraction
- Timeline Construction
- Multi-format Export

## Technical Stack

### Backend
- FastAPI, PostgreSQL, Neo4j, Redis
- OpenAI API, Transformers, spaCy
- pandas, numpy, networkx

### Frontend
- Vanilla JavaScript, Bootstrap 5
- D3.js for visualization

### Data Collection
- Scrapy, Selenium, BeautifulSoup
- API integrations (Google, Twitter, LinkedIn)

## Quick Start

### Requirements
- Python 3.8+
- PostgreSQL, Neo4j, Redis

### Setup
```bash
git clone <repository-url>
cd osint-system
pip install -r requirements.txt
cp config.env.example config.env
# Edit config.env with your API keys
python main.py
```

### Access
- Web Interface: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Configuration
```bash
# Required API Keys
GOOGLE_API_KEY=your-google-api-key
OPENAI_API_KEY=your-openai-api-key
DATABASE_URL=postgresql://user:password@localhost/osint_db
```

## Usage

### Web Interface
1. Enter target person's name
2. Start investigation
3. View results and download reports

### API Usage
```bash
# Create investigation
curl -X POST "http://localhost:8000/api/v1/investigations" \
  -H "Content-Type: application/json" \
  -d '{"target_name": "John Smith", "user_id": "user123"}'

# Get results
curl "http://localhost:8000/api/v1/investigations/{id}/results"
```

## System Architecture

```
Web Frontend ←→ API Service ←→ Data Collectors
                     ↓
              AI Modules + Databases
```

## Important Notes

**Experimental Purpose**: This is a demo system for research and educational purposes only.

**AI Development**: AI-related models, including psychological vulnerability analysis and social engineering research, are currently under development.

**Legal Compliance**: Only collects publicly available information. Respects robots.txt and implements rate limiting.

**Collaboration**: We welcome researchers, developers, and security professionals to collaborate on this project.

## Development

### Project Structure
```
osint-system/
├── app/                    # Application
├── static/                 # Static files
├── templates/              # HTML templates
├── requirements.txt        # Dependencies
└── main.py                # Main program
```

### Contributing
1. Follow code standards
2. Add tests
3. Update documentation
4. Pass compliance checks

## License

MIT License - see LICENSE file for details.

## Disclaimer

This tool is for legitimate open-source intelligence collection only. Users must ensure compliance with local laws and respect website terms of use. Developers bear no responsibility for misuse.

---

**OSINT Intelligence Collection System** - Experimental research platform for intelligence gathering 