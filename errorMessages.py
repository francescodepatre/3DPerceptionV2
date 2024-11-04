import tkinter as tk
from tkinter import messagebox
import logging
import traceback

class ErrorManager:

    def log_error(message):
        logging.error(message)

    def log_exception(e):
        error_message = f"Exception: {e}\nTraceback:\n{traceback.format_exc()}"
        logging.exception(error_message)

    def message_error(self, message):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror('Errore', message)
        root.destroy()
