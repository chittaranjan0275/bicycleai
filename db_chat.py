from typing import List
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
import os

def load_csvs_to_db(csv_paths: List[str], db_path: str) -> List[str]:
    """
    Load multiple CSV files into SQLite database tables.
    
    Args:
        csv_paths: List of paths to CSV files
        db_path: Path where SQLite database should be created
    
    Returns:
        List of created table names
    """
    # Ensure db_path has .db extension
    if not db_path.endswith('.db'):
        db_path = db_path + '.db'
    
    # Create directory if it doesn't exist
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    engine = create_engine(f'sqlite:///{db_path}')
    table_names = []
    
    for csv_path in csv_paths:
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Generate table name from file name
            table_name = os.path.splitext(os.path.basename(csv_path))[0].lower()
            
            # Load to database
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            table_names.append(table_name)
            
            print(f"Successfully loaded {csv_path} into table '{table_name}'")
            print(f"Columns: {', '.join(df.columns)}")
            print(f"Rows: {len(df)}\n")
            
        except Exception as e:
            print(f"Error loading {csv_path}: {str(e)}\n")
    
    return table_names

async def chat_with_data(db_path: str, query: str):
    """
    Chat with the data using LangChain SQL Agent.
    """
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    llm = ChatOpenAI(temperature=0)
    
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True
    )
    
    result = await agent_executor.ainvoke({"input": query})
    return result['output']

async def main():
    print("Welcome to the Database Chat Interface!")
    
    while True:
        # Get database path
        db_path = input("\nEnter the path for the SQLite database (e.g., my_data.db): ")
        
        # Add .db extension if missing
        if not db_path.endswith('.db'):
            db_path = db_path + '.db'
        
        # Create directory if it doesn't exist
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir)
            except Exception as e:
                print(f"Error creating directory: {str(e)}")
                continue
        
        # Verify if path is writable
        try:
            with open(db_path, 'a'):  # Try to open file for writing
                pass
            break
        except Exception as e:
            print(f"Error: Cannot write to database path: {str(e)}")
            print("Please try a different path")
    
    # Get CSV files
    csv_paths = []
    while True:
        path = input("\nEnter path to a CSV file (or 'done' to finish): ")
        if path.lower() == 'done':
            break
        if os.path.exists(path):
            csv_paths.append(path)
        else:
            print(f"File not found: {path}")
    
    if not csv_paths:
        print("No valid CSV files provided.")
        return
    
    # Load CSVs into database
    print("\nLoading CSV files into database...")
    table_names = load_csvs_to_db(csv_paths, db_path)
    
    if not table_names:
        print("No tables were created. Exiting.")
        return
    
    # Print available tables
    print("\nAvailable tables in database:")
    for table in table_names:
        print(f"- {table}")
    
    # Start chat interface
    print("\nChat Interface Ready!")
    print("Type 'exit' to quit")
    print("Type 'tables' to see available tables")
    
    while True:
        query = input("\nAsk a question about your data: ")
        
        if query.lower() == 'exit':
            break
        
        if query.lower() == 'tables':
            print("\nAvailable tables:")
            for table in table_names:
                print(f"- {table}")
            continue
        
        try:
            response = await chat_with_data(db_path, query)
            print("\nResponse:", response)
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 