import time
from main import LibrarySystem

def main():
    """Main application entry point"""
    print("=" * 50)
    print("       AI Library Recommendation System           ")
    print("=" * 50)
    
    # Ask for model preference
    print("\nSelect an LLM model to use:")
    print("1. Ollama/llama3 (local)")
    print("2. GPT-3.5 Turbo (requires API key)")
    print("3. GPT-4 (requires API key)")
    print("4. Other (specify)")
    
    while True:
        choice = input("Enter choice (1-4): ").strip()
        if choice == "1":
            model = "ollama/llama3"
            break
        elif choice == "2":
            model = "gpt-3.5-turbo"
            break
        elif choice == "3":
            model = "gpt-4"
            break
        elif choice == "4":
            model = input("Enter model name (e.g., ollama/mistral): ").strip()
            break
        else:
            print("Invalid choice. Please enter 1-4.")
    
    print(f"\nInitializing system with {model}...")
    
    try:
        library_system = LibrarySystem(model_name=model)
        
        while True:
            print("\n" + "=" * 50)
            topic = input("\nEnter a topic you're interested in (or 'exit' to quit): ")
            
            if topic.lower() == 'exit':
                print("\nThank you for using the AI Library Recommendation System!")
                break
            
            start_time = time.time()
            print("\nProcessing your request... This may take several minutes.\n")
            
            try:
                results = library_system.recommend_books(topic)
                
                print("\n" + "=" * 50)
                print("             REQUIREMENTS ANALYSIS             ")
                print("=" * 50)
                print(results["requirements"])
                
                print("\n" + "=" * 50)
                print("             BOOK RECOMMENDATIONS              ")
                print("=" * 50)
                print(results["recommendations"])
                
                elapsed_time = time.time() - start_time
                print(f"\nProcess completed in {elapsed_time:.2f} seconds.")
            except Exception as e:
                print(f"\n❌ Error occurred: {str(e)}")
                print("Please try again with a different topic.")
                
    except Exception as e:
        print(f"\n❌ Error initializing system: {str(e)}")
        if "ollama" in str(model).lower():
            print("1. Make sure Ollama is installed and running")
            print("2. Check if the requested model is available in Ollama")
            print("3. Run 'ollama list' to see available models")
            print("4. Run 'ollama pull llama3' to download the llama3 model")
        else:
            print("Check your API keys and configuration and try again.")

if __name__ == "__main__":
    main()