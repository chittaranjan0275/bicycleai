## Setup Instructions

1. Clone the repository:
   ```sh
   git clone https://github.com/chittaranjan0275/bicycleai.git
   cd bicycleai
   ```

2. Create a virtual environment (conda) and activate it:
   ```sh
   conda create --name your-env-name python=3.8
   conda activate your-env-name
   ```

3. Install the dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Set your OpenAI API key:
   ```sh
   export OPENAI_API_KEY=your_openai_api_key
   ```

5. Run the `graph_llm.py` script:
   ```sh
   python graph_llm.py /path/to/your/csvfile.csv
   ```
