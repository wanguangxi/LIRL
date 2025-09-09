# LIRL

## OpenAI Client Implementation

This repository contains a corrected OpenAI API client for generating academic papers.

### Fixed Issues

The original code had several issues that have been corrected:

1. **Incorrect API method**: Changed from `client.responses.create()` to `client.chat.completions.create()`
2. **Incorrect parameter name**: Changed from `input` to `messages`
3. **Unformatted output**: Changed from `response.json()` to properly formatted text output
4. **Added error handling**: Proper exception handling for API calls
5. **Environment variable support**: API key can be set via `OPENAI_API_KEY` environment variable

### Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your API key (optional, falls back to hardcoded key):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. Run the client:
   ```python
   from openai_client import response_openai
   
   # Generate formatted academic paper
   result = response_openai()
   ```

### Files

- `openai_client.py` - Main implementation with corrected OpenAI API usage
- `test_openai_client.py` - Test script to validate the implementation
- `requirements.txt` - Python dependencies

### Functions

- `response_openai()` - Main function that returns formatted text output
- `response_openai_json()` - Alternative function that returns full JSON response for debugging