# Chess Player Analyzer Development Guidelines

This document provides guidelines and information for developers working on the Chess Player Analyzer project.

## Build and Configuration Instructions

### Local Development Setup

1. **Clone the repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development/testing
   ```

3. **Environment Variables**:
   The following environment variables are required:
   - `DATABASE_URL`: PostgreSQL connection string (e.g., `postgresql+psycopg://chess:chess@localhost:5432/chessdb`)
   - `REDIS_URL`: Redis connection string (e.g., `redis://localhost:6379/0`)
   - `STOCKFISH_PATH`: Path to Stockfish executable (e.g., `/usr/games/stockfish`)
   - `STOCKFISH_DEPTH`: Depth for Stockfish analysis (e.g., `12`)

   You can set these in a `.env` file at the project root.

4. **Database Setup**:
   ```bash
   python -m app.init_db
   ```

### Docker Setup

1. **Build and start the containers**:
   ```bash
   docker-compose up -d
   ```

   This will start the following services:
   - PostgreSQL database
   - Redis for caching and message brokering
   - Database migration service
   - FastAPI backend
   - Celery worker for background tasks

2. **Access the API**:
   The API will be available at `http://localhost:8000`

## Testing Information

### Running Tests

The project uses pytest for testing. To run the tests:

```bash
cd backend
python -m pytest
```

To run specific tests:

```bash
python -m pytest tests/test_health.py
python -m pytest tests/test_player_flow.py
```

To run tests with verbose output:

```bash
python -m pytest -v
```

### Test Configuration

Tests use a separate database configuration defined in `tests/conftest.py`. The default test database URL is `postgresql+psycopg://postgres:postgres@localhost/testdb`.

Celery tasks are configured to run eagerly (synchronously) during tests using:
```python
celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)
```

### Writing Tests

1. **Create a new test file** in the `tests` directory with a name starting with `test_`.

2. **Use the provided fixtures**:
   - `client`: A FastAPI TestClient instance
   - `setup_db`: Automatically sets up and tears down the test database

3. **Example test**:

```python
def test_example(client):
    """
    Example test to demonstrate how to write tests for this project.
    This test checks if the /health endpoint returns the expected response.
    """
    # Make a GET request to the health endpoint
    response = client.get("/health")
    
    # Check that the response status code is 200 (OK)
    assert response.status_code == 200
    
    # Parse the JSON response
    data = response.json()
    
    # Check that the response contains the expected data
    assert "status" in data
    assert data["status"] == "healthy"
    
    # You can add more assertions as needed
    # For example, checking response headers
    assert response.headers["content-type"] == "application/json"
```

4. **Test data**: Place test data files (like PGN files) in the `tests/data` directory.

## Development Information

### Project Structure

- `app/`: Main application code
  - `main.py`: FastAPI application and endpoints
  - `models.py`: SQLModel database models
  - `schemas.py`: Pydantic schemas for API requests/responses
  - `celery_app.py`: Celery configuration and tasks
  - `analysis/`: Chess analysis modules
- `tests/`: Test files
- `docker-compose.yml`: Docker Compose configuration

### Code Style

- The project uses type hints throughout the codebase
- Function and method names use snake_case
- Class names use PascalCase
- Constants use UPPER_CASE
- Docstrings are provided for public functions and methods

### API Endpoints

The main API endpoints are:
- `/analyze`: Analyze a chess game
- `/games/{game_id}`: Get information about a specific game
- `/players/{username}`: Get information about a specific player
- `/players/{username}/analyze`: Analyze a player's games
- `/metrics/game/{game_id}`: Get metrics for a specific game
- `/metrics/player/{username}`: Get metrics for a specific player

### Background Tasks

The project uses Celery for background tasks:
- Game analysis
- Player analysis

Tasks are queued in Redis and processed by Celery workers.

### Debugging

For debugging, you can:
1. Use the `/tasks/{task_id}` endpoint to check the status of background tasks
2. Check the logs of the various Docker containers
3. Use the debug_results directory for storing debug information
