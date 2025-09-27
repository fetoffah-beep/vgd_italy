
def log_message(message, file_path):
    print(f"{message}", file=file_path)
    file_path.flush()
    
    
    
    