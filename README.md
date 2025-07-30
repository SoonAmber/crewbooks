# Library Recommendation System

An AI-powered library recommendation system built with crewAI that recommends books based on user topics of interest.

## Features

- Multiple AI agents with different perspectives analyze user topics
- Employee agents search for, organize, and recommend relevant books
- Integration with local library database
- Online search capability for books not in the database
- Database management to add new books

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Run the application:
   ```
   python src/app.py
   ```

## System Architecture

### Reader Agents
- Knowledge Expander: Explores new and emerging areas related to the topic
- Inherent Knowledge Keeper: Focuses on core knowledge and authoritative sources
- Multidimensional Integrator: Connects knowledge across different disciplines

### Employee Agents
- Demand Assistant: Analyzes user needs and extracts keywords
- Retrieval Assistant: Searches for relevant books
- Organization Assistant: Evaluates and filters resources
- Collection Assistant: Manages database updates
- Recommendation Assistant: Creates final personalized recommendations

## Usage

Enter a topic of interest when prompted. The system will analyze it from multiple perspectives and provide tailored book recommendations.