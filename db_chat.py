from typing import List
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
import os
import glob
import shutil

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
            # Read CSV with error handling
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # Clean table name: remove special characters and spaces
            table_name = os.path.splitext(os.path.basename(csv_path))[0].lower()
            table_name = ''.join(e for e in table_name if e.isalnum() or e == '_')
            
            # Verify DataFrame is not empty
            if df.empty:
                print(f"Warning: {csv_path} is empty, skipping...")
                continue
                
            # Clean column names
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            
            # Load to database with transaction
            with engine.begin() as connection:
                df.to_sql(table_name, connection, if_exists='replace', index=False)
            
            table_names.append(table_name)
            
            print(f"Successfully loaded {csv_path} into table '{table_name}'")
            print(f"Columns: {', '.join(df.columns)}")
            print(f"Rows: {len(df)}\n")
            
        except pd.errors.EmptyDataError:
            print(f"Error: {csv_path} is empty or has no valid data\n")
        except pd.errors.ParserError:
            print(f"Error: Could not parse {csv_path}. Please check if it's a valid CSV file\n")
        except Exception as e:
            print(f"Error loading {csv_path}: {str(e)}\n")
            continue
    
    engine.dispose()  # Properly close the database connection
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
    
    # Set database path to current working directory
    db_name = "aviation_test.db"
    db_path = os.path.join(os.getcwd(), db_name)
    
    # Delete existing database if it exists
    if os.path.exists(db_path):
        print(f"\nRemoving existing database: {db_path}")
        try:
            os.remove(db_path)
        except Exception as e:
            print(f"Error removing existing database: {str(e)}")
            return
    
    # Find all cleaned CSV files in current directory
    csv_paths = glob.glob(os.path.join(os.getcwd(), "*_cleaned.csv"))
    
    if not csv_paths:
        print("No cleaned CSV files found in current directory.")
        print("Please ensure files end with '_cleaned.csv'")
        return
    
    # Print found files
    print("\nFound the following cleaned CSV files:")
    for path in csv_paths:
        print(f"- {os.path.basename(path)}")
    
    # Load CSVs into database
    print("\nLoading CSV files into database...")
    table_names = load_csvs_to_db(csv_paths, db_path)
    
    if not table_names:
        print("No tables were created. Exiting.")
        return
    
    # Print database info
    print(f"\nDatabase created at: {db_path}")
    print("\nTables created in database:")
    for table in table_names:
        print(f"- {table}")
    
    # Start chat interface
    print("\nChat Interface Ready!")
    print("Available commands:")
    print("- 'exit': Quit the application")
    print("- 'tables': Show available tables")
    print("- 'schema': Show table schemas")
    
    while True:
        query = input("\nAsk a question about your data: ")
        
        if query.lower() == 'exit':
            break
        
        if query.lower() == 'tables':
            print("\nAvailable tables:")
            for table in table_names:
                print(f"- {table}")
            continue
        
        if query.lower() == 'schema':
            try:
                # Get schema for each table
                db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
                for table in table_names:
                    schema = db.get_table_info(table)
                    print(f"\nSchema for {table}:")
                    print(schema)
            except Exception as e:
                print(f"Error getting schema: {str(e)}")
            continue
        
        try:
            response = await chat_with_data(db_path, query)
            print("\nResponse:", response)
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 