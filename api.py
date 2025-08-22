# api.py
from fastapi import FastAPI, UploadFile, Form
from typing import List, Optional
import os
import shutil

from Groq_CLI import MarkdownChatbot, find_markdown_files

app = FastAPI()

# Store chatbot instance globally (so we don’t reload every request)
chatbot = None


@app.post("/init")
async def init_chatbot(api_key: str = Form(...), files: Optional[List[UploadFile]] = None, directory: Optional[str] = None):
    """
    Initialize chatbot with Groq API key and markdown files.
    """
    global chatbot
    chatbot = MarkdownChatbot(api_key)

    files_to_process = []

    # Save uploaded files temporarily
    if files:
        os.makedirs("uploaded_files", exist_ok=True)
        for f in files:
            file_path = os.path.join("uploaded_files", f.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            files_to_process.append(file_path)

    # Add directory files if provided
    if directory and os.path.isdir(directory):
        dir_files = find_markdown_files(directory)
        files_to_process.extend(dir_files)

    # Filter markdown files
    unique_files = list(set(f for f in files_to_process if f.endswith(".md")))
    if not unique_files:
        return {"error": "No markdown files found."}

    if not chatbot.load_markdown_files(unique_files):
        return {"error": "Failed to load documents."}

    return {"status": "Chatbot initialized", "files_loaded": unique_files}


@app.post("/ask")
async def ask_question(question: str = Form(...), show_citations: bool = True, show_themes: bool = True):
    """
    Ask a question to the chatbot (must be initialized first).
    """
    global chatbot
    if not chatbot:
        return {"error": "Chatbot not initialized. Call /init first."}

    result = chatbot.answer_question(question, show_citations, show_themes)
    return result
