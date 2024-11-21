# Importing necessary libraries
import pandas as pd

def load_data_in_chunks(file_path, chunksize):
    """
    Loads a dataset in chunks and displays its content.

    Parameters:
        file_path (str): The path to the CSV file.
        chunksize (int): Number of rows to read per chunk.

    Returns:
        None
    """
    try:
        # Reading the CSV file in chunks
        chunk_reader = pd.read_csv(file_path, chunksize=chunksize)
        
        print(f"Dataset successfully loaded from {file_path} in chunks of size {chunksize}.\n")

        # Displaying each chunk
        for i, chunk in enumerate(chunk_reader):
            print(f"Chunk {i + 1}:")
            print(chunk)  # Displays the chunk
            print("\nColumns in the chunk:")
            print(chunk.columns.tolist())  # Displays all column names
            print("\n---\n")

            # Optional: Stop after displaying 3 chunks for demonstration
            if i == 2:
                print("Displayed 3 chunks for demonstration. Stop here.")
                break

    except FileNotFoundError:
        print(f"File not found at {file_path}. Please check the path.")
        raise
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        raise

# Define the file path and chunk size
file_path = 'C:/Users/Utilisateur/credit-banque/loan-data-673b233f1c1cb921157550.csv'  # Replace with the actual file path
chunksize = 5  # Number of rows per chunk

# Set Pandas display options to show all columns
pd.set_option('display.max_columns', None)

# Load and display the dataset in chunks
load_data_in_chunks(file_path, chunksize)
