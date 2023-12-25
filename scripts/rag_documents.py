from utils.rag import save_documents_to_db
import sys

if __name__ == "__main__":
    path = sys.argv[1]
    save_documents_to_db(path)