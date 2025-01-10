# AcquiredAgent

## About AcquiredAgent

AcquiredAgent is an AI-powered conversational agent that leverages the rich content from Acquired FM, the #1 Technology show on Apple Podcasts and Spotify. This bot allows users to interact with and gain insights from the extensive knowledge base of Acquired's in-depth episodes on great companies and their strategies.

## About Acquired FM

Acquired tells the stories and strategies of great companies. Each episode reaches over one million listeners, and The Wall Street Journal describes it as "the business world's favorite podcast."

Key features of Acquired FM:
- Depth-first approach with 3-4 hour long episodes
- Described as "conversational audiobooks" rather than traditional podcasts
- Occasionally features guests such as founders/CEOs of NVIDIA, Berkshire Hathaway, Starbucks, Meta, Spotify, Uber, Zoom, CAA, Sequoia Capital, and all five Benchmark partners

## Project Overview

AcquiredAgent aims to make the wealth of knowledge from Acquired FM's episodes more accessible and interactive. By ingesting and processing the transcripts from Acquired FM episodes, this bot can answer questions, provide insights, and engage in conversations about the companies and strategies discussed in the show.

### Key Features

1. Transcript Ingestion: Automatically scrapes and processes transcripts from Acquired FM episodes.
2. Knowledge Base Creation: Builds a searchable index of the processed transcripts.
3. Conversational AI: Allows users to ask questions and receive informed responses based on the content of Acquired FM episodes.
4. Continuous Learning: Updates its knowledge base as new episodes are released.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```
git clone https://github.com/your-username/AcquiredAgent.git && cd AcquiredAgent
```
2. Install required packages:
```
pip install -r requirements.txt
```
3. Set up environment variables in a `.env` file

### Usage

1. Run ingestion script: python src/ingest.py
2. Start conversational agent: python src/process_and_query.py
3. Interact with the agent via command line or preferred chat interface

## Contributing

Contributions welcome. See contribution guidelines for details.

## License

MIT License - see LICENSE file

## Acknowledgments

- Acquired FM for their content
- Open-source community for tools and libraries

## Disclaimer

This project is not officially affiliated with Acquired FM.
