from src.xpu import XPU

def main():
    # Initialize your XPU processor
    processor = XPU()
    
    # Add your application logic here
    print("Application started")
    processor.run()

if __name__ == "__main__":
    main()