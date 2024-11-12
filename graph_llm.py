from typing import Dict, TypedDict, List, Any, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph, StateGraph
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import os
import subprocess

# Define state schema
class GraphState(TypedDict):
    messages: List[Any]
    data: Dict[str, Any]
    cleaning_steps: List[str]
    current_step: str
    issues_found: Dict[str, Any]
    cleaned: bool
    csv_path: str

# Initialize LLM
# llm = ChatOpenAI(temperature=0)

llm = ChatOpenAI(temperature=0)

# Node Functions
def csv_input(state: GraphState) -> GraphState:
    """Load CSV data from the specified path."""
    try:
        if not os.path.exists(state['csv_path']):
            raise FileNotFoundError(f"CSV file not found at path: {state['csv_path']}")

        data = pd.read_csv(state['csv_path'])
        # data = pd.read_csv(state['csv_path'], nrows=20)

        state['data'] = {'original': data, 'current': data.copy()}
        state['messages'].append(AIMessage(content=f"Data loaded successfully from {state['csv_path']}"))
    except Exception as e:
        state['messages'].append(AIMessage(content=f"Error loading data: {str(e)}"))
    return state


def data_existence_check(state: GraphState) -> GraphState:
    """Check if data exists and is valid."""
    if 'data' not in state or 'current' not in state['data'] or state['data']['current'].empty:
        state['messages'].append(AIMessage(content="No data found"))
        return state

    state['messages'].append(AIMessage(content=f"Data exists and is valid. Shape: {state['data']['current'].shape}"))
    return state


def data_profiling(state: GraphState) -> GraphState:
    """Profile the data and suggest column name improvements."""
    df = state['data']['current']
    
    # Original profiling logic
    issues = {
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    state['issues_found'] = issues

    # Add column name analysis
    column_analysis_prompt = f"""
    Analyze these column names and suggest improvements:
    {list(df.columns)}
    
    Consider:
    1. Are they human readable?
    2. Do they follow consistent naming?
    3. Are abbreviations clear?
    
    Provide reasoning for each suggestion.
    """
    
    # Create message for column name suggestions
    messages = [
        HumanMessage(content=column_analysis_prompt)
    ]
    
    # Get column name suggestions from LLM
    response = llm.invoke(messages)
    
    # Add to state messages
    summary = f"Data profiling results:\n"
    summary += f"Missing values: {issues['missing_values']}\n"
    summary += f"Duplicate rows: {issues['duplicates']}\n"
    summary += f"Data types: {issues['dtypes']}\n"
    summary += f"\nColumn name analysis:\n{response.content}"

    state['messages'].append(AIMessage(content=summary))
    return state


def rename_columns(state: GraphState) -> GraphState:
    """Rename columns based on LLM suggestions for better readability."""
    df = state['data']['current']
    
    # Create prompt for column renaming
    system_message = """You are a data cleaning assistant. Given a list of column names, 
    return a Python dictionary mapping old column names to new improved names.
    Follow these rules:
    - Use clear, descriptive names
    - Replace abbreviations with full words
    - Use lowercase with underscores
    - Keep names concise but meaningful
    Return only the Python dictionary, nothing else."""
    
    user_message = f"Create a rename mapping dictionary for these columns: {list(df.columns)}"
    
    # Create messages for LLM
    messages = [
        HumanMessage(content=system_message),
        HumanMessage(content=user_message)
    ]
    
    try:
        # Get rename mapping from LLM
        response = llm.invoke(messages)
        
        # Safely evaluate the LLM response to get the dictionary
        rename_mapping = eval(response.content)
        
        # Apply the renaming
        df = df.rename(columns=rename_mapping)
        state['data']['current'] = df
        
        # Log the changes
        changes = "\nColumn names updated:\n"
        for old, new in rename_mapping.items():
            changes += f"- {old} → {new}\n"
        state['messages'].append(AIMessage(content=changes))
        
    except Exception as e:
        state['messages'].append(AIMessage(content=f"Error renaming columns: {str(e)}"))
    
    return state


def identify_cleaning_steps(state: GraphState) -> GraphState:
    """Identify necessary cleaning steps based on profiling results."""
    issues = state['issues_found']
    steps = []

    if any(issues['missing_values'].values()):
        steps.append('missing_value_imputation')
    if issues['duplicates'] > 0:
        steps.append('duplicate_removal')

    df = state['data']['current']
    date_columns = []
    for col in df.columns:
        try:
            pd.to_datetime(df[col], errors='raise')
            date_columns.append(col)
        except:
            continue

    if date_columns:
        steps.append('data_type_conversion')

    state['cleaning_steps'] = steps
    state['current_step'] = steps[0] if steps else 'done'

    state['messages'].append(AIMessage(content=f"Identified cleaning steps: {steps}"))
    return state


def missing_value_imputation(state: GraphState) -> GraphState:
    """Handle missing values using KNN imputation for numeric and mode for categorical."""
    if 'missing_value_imputation' not in state['cleaning_steps']:
        return state

    df = state['data']['current']

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        imputer = KNNImputer(n_neighbors=2)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "MISSING", inplace=True)

    state['data']['current'] = df
    state['messages'].append(AIMessage(content="Missing values imputed"))
    return state


def duplicate_removal(state: GraphState) -> GraphState:
    """Remove duplicate rows."""
    if 'duplicate_removal' not in state['cleaning_steps']:
        return state

    df = state['data']['current']
    original_count = len(df)
    df = df.drop_duplicates()
    removed_count = original_count - len(df)

    state['data']['current'] = df
    state['messages'].append(AIMessage(content=f"Removed {removed_count} duplicate rows"))
    return state


def data_type_conversion(state: GraphState) -> GraphState:
    """Convert data types where necessary."""
    if 'data_type_conversion' not in state['cleaning_steps']:
        return state

    df = state['data']['current']

    for column in df.columns:
        try:
            if df[column].dtype == 'object':
                pd.to_datetime(df[column], errors='raise')
                df[column] = pd.to_datetime(df[column])
                state['messages'].append(AIMessage(content=f"Converted {column} to datetime"))
        except:
            continue

    state['data']['current'] = df
    return state


def post_cleaning_check(state: GraphState) -> GraphState:
    """Verify the cleaning results."""
    df = state['data']['current']
    issues = {
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }

    state['issues_found'] = issues
    summary = f"Post-cleaning check results:\n"
    summary += f"Remaining missing values: {issues['missing_values']}\n"
    summary += f"Remaining duplicate rows: {issues['duplicates']}\n"
    summary += f"Final data types: {issues['dtypes']}"

    state['messages'].append(AIMessage(content=summary))
    return state


def check_satisfactory_results(state: GraphState) -> str:
    """Determine if cleaning results are satisfactory."""
    issues = state['issues_found']
    if any(issues['missing_values'].values()) or issues['duplicates'] > 0:
        return 'continue_cleaning'
    return 'complete'


def csv_output(state: GraphState) -> GraphState:
    """Export cleaned data to CSV."""
    cleaned_df = state['data']['current']

    input_path = state['csv_path']
    base_path = os.path.splitext(input_path)[0]
    output_path = f"{base_path}_cleaned.csv"

    cleaned_df.to_csv(output_path, index=False)

    state['cleaned'] = True
    state['messages'].append(AIMessage(content=f"Cleaned data saved to: {output_path}"))
    return state


def create_data_cleaning_graph() -> Graph:
    # Create a new StateGraph instance
    workflow = StateGraph(GraphState)

    # Define the entry point
    workflow.add_node("start", csv_input)

    # Add remaining nodes
    workflow.add_node("data_existence_check", data_existence_check)
    workflow.add_node("data_profiling", data_profiling)
    workflow.add_node("rename_columns", rename_columns)
    workflow.add_node("identify_cleaning_steps", identify_cleaning_steps)
    workflow.add_node("missing_value_imputation", missing_value_imputation)
    workflow.add_node("duplicate_removal", duplicate_removal)
    workflow.add_node("data_type_conversion", data_type_conversion)
    workflow.add_node("post_cleaning_check", post_cleaning_check)
    workflow.add_node("csv_output", csv_output)

    # Add edges starting from the entry point
    workflow.set_entry_point("start")
    workflow.add_edge("start", "data_existence_check")
    workflow.add_edge("data_existence_check", "data_profiling")
    workflow.add_edge("data_profiling", "rename_columns")
    workflow.add_edge("rename_columns", "identify_cleaning_steps")
    workflow.add_edge("identify_cleaning_steps", "missing_value_imputation")
    workflow.add_edge("missing_value_imputation", "duplicate_removal")
    workflow.add_edge("duplicate_removal", "data_type_conversion")
    workflow.add_edge("data_type_conversion", "post_cleaning_check")

    # Add conditional edges
    workflow.add_conditional_edges(
        "post_cleaning_check",
        check_satisfactory_results,
        {
            "continue_cleaning": "identify_cleaning_steps",
            "complete": "csv_output"
        }
    )

    # Set the end nodes
    workflow.set_finish_point("csv_output")

    return workflow.compile()


async def main(csv_path: str):
    """
    Run the data cleaning workflow for a single file.
    """
    # Create and compile the graph
    graph = create_data_cleaning_graph()

    # Initialize state
    state = GraphState(
        messages=[],
        data={},
        cleaning_steps=[],
        current_step="",
        issues_found={},
        cleaned=False,
        csv_path=csv_path
    )

    # Run the graph asynchronously
    final_state = await graph.ainvoke(state)

    # Print cleaning results
    print("\nData Cleaning Workflow Results:")
    print("-" * 40)
    for message in final_state['messages']:
        print(message.content)
        print("-" * 40)

    return final_state['cleaned']

if __name__ == "__main__":
    import sys
    import asyncio

    # Default file paths
    default_files = [
        "/home/spartan/Desktop/bicycleai/Flight Bookings.csv",
        "/home/spartan/Desktop/bicycleai/Airline ID to Name.csv"
    ]

    # Use command line arguments or default to sample paths
    if len(sys.argv) > 2:
        file_paths = [sys.argv[1], sys.argv[2]]
    else:
        print("No files provided as arguments. Using default file paths:")
        print(f"File 1: {default_files[0]}")
        print(f"File 2: {default_files[1]}")
        file_paths = default_files

    # Process all files and track success
    all_files_cleaned = True
    for csv_path in file_paths:
        print(f"\nProcessing file: {csv_path}")
        cleaned = asyncio.run(main(csv_path))
        all_files_cleaned = all_files_cleaned and cleaned

    # Launch db_chat.py only after all files are cleaned
    if all_files_cleaned:
        print("\nAll files cleaned! Launching database chat interface...")
        print("-" * 40)
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            db_chat_path = os.path.join(script_dir, 'db_chat.py')
            subprocess.run(['python', db_chat_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error launching database chat interface: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
    else:
        print("\nSome files were not cleaned successfully. Skipping database chat launch.")