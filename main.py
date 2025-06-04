import sys
from src.rolling_window_analyzer import main

sys.path.append('src')

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        print("Make sure all required packages are installed and the src directory exists.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")