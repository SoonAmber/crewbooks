import pandas as pd
from langchain.tools import tool

class LibraryTools:
    def __init__(self, library_path="data/lite_library.csv"):
        self.library_path = library_path
    
    def _load_library(self):
        try:
            return pd.read_csv(self.library_path)
        except Exception as e:
            print(f"Error loading library: {str(e)}")
            return pd.DataFrame(columns=["Number", "Title", "Call Number", "Author", 
                                        "Publication Information", "Content and Summary", "status"])
    
    def _save_library(self, df):
        df.to_csv(self.library_path, index=False)
    
    @tool("Search in library database")
    def search_library(self, query):
        """Search for books in the library database based on the query."""
        try:
            library = self._load_library()
            
            if library.empty:
                return "The library database is empty or could not be loaded."
            
            # Search in title, author, and content summary
            results = library[
                library['Title'].str.contains(query, case=False, na=False) |
                library['Author'].str.contains(query, case=False, na=False) |
                library['Content and Summary'].str.contains(query, case=False, na=False)
            ]
            
            if results.empty:
                return f"No books found in the library for query: {query}"
            
            output = []
            for _, row in results.iterrows():
                output.append({
                    "Number": row['Number'],
                    "Title": row['Title'],
                    "Call Number": row['Call Number'],
                    "Author": row['Author'],
                    "Publication Information": row['Publication Information'],
                    "Content and Summary": row['Content and Summary'],
                    "Status": row['status']
                })
            
            return output
        except Exception as e:
            print(f"Error searching library: {e}")
            return f"Error searching library: {str(e)}"
    
    @tool("Check if book exists in library")
    def check_book_exists(self, title):
        """Check if a book with the given title exists in the library."""
        try:
            library = self._load_library()
            
            if library.empty:
                return False
            
            exists = library['Title'].str.contains(title, case=False, na=False).any()
            return exists
        except Exception as e:
            print(f"Error checking if book exists: {e}")
            return False
    
    @tool("Add book to library")
    def add_to_library(self, book_data):
        """Add a new book to the library database."""
        try:
            library = self._load_library()
            
            new_number = int(library['Number'].max() + 1) if not library.empty else 1
            
            new_book = {
                "Number": new_number,
                "Title": book_data.get("Title", "Unknown Title"),
                "Call Number": book_data.get("Call Number", "Unknown"),
                "Author": book_data.get("Author", "Unknown Author"),
                "Publication Information": book_data.get("Publication Information", "Unknown"),
                "Content and Summary": book_data.get("Content and Summary", "No description available"),
                "status": book_data.get("status", "available")
            }
            
            library = pd.concat([library, pd.DataFrame([new_book])], ignore_index=True)
            self._save_library(library)
            
            return f"Added book '{new_book['Title']}' to library with Number {new_number}"
        except Exception as e:
            print(f"Error adding book to library: {e}")
            return f"Error adding book to library: {str(e)}"