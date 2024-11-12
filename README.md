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

5. Run the `graph_llm.py` script to process and clean the data:
   ```sh
   python graph_llm.py /path/to/your/csvfile.csv
   ```

6. Launch the Streamlit chat interface:
   ```sh
   streamlit run streamlit_chat.py
   ```
   This will open a web browser with the chat interface where you can interact with your aviation database. The interface will be available at `http://localhost:8501`.

Note: Ensure that the `aviation_test.db` file has been created by running `graph_llm.py` before launching the Streamlit interface.
