"""
LinerNet desktop UI (tkinter)
-----------------------------
Features:
  - Enter Gemini API key once and save to .env
  - Select/upload 4 required CSV files
  - One-click run for full pipeline
"""

import os
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

import sys
sys.path.insert(0, PROJECT_ROOT)

from run_pipeline import run_all
from utils.env import get_gemini_key, save_gemini_key, load_dotenv


REQUIRED_FILES = {
    "ports.csv": "ports.csv",
    "dist_dense.csv": "dist_dense.csv",
    "fleet_data.csv": "fleet_data.csv",
    "demand_worldsmall.csv": "demand_worldsmall.csv",
}


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LinerNet One-Click Runner")
        self.root.geometry("980x680")

        load_dotenv(PROJECT_ROOT)
        self.file_vars = {k: tk.StringVar(value="") for k in REQUIRED_FILES}

        self.key_var = tk.StringVar(value=get_gemini_key(PROJECT_ROOT))
        self.status_var = tk.StringVar(value="Ready.")

        self._build()

    def _build(self):
        top = tk.Frame(self.root, padx=12, pady=10)
        top.pack(fill="x")

        tk.Label(top, text="Gemini API Key (.env):").grid(row=0, column=0, sticky="w")
        tk.Entry(top, textvariable=self.key_var, width=70, show="*").grid(row=0, column=1, sticky="we", padx=6)
        tk.Button(top, text="Save Key", command=self.save_key).grid(row=0, column=2, padx=4)
        top.grid_columnconfigure(1, weight=1)

        files = tk.LabelFrame(self.root, text="Upload / Select the 4 data CSV files", padx=12, pady=8)
        files.pack(fill="x", padx=12, pady=8)

        for i, fname in enumerate(REQUIRED_FILES):
            tk.Label(files, text=fname, width=20, anchor="w").grid(row=i, column=0, sticky="w")
            tk.Entry(files, textvariable=self.file_vars[fname], width=80).grid(row=i, column=1, sticky="we", padx=6)
            tk.Button(files, text="Browse", command=lambda f=fname: self.pick_file(f)).grid(row=i, column=2, padx=4)
        files.grid_columnconfigure(1, weight=1)

        actions = tk.Frame(self.root, padx=12, pady=8)
        actions.pack(fill="x")
        tk.Button(actions, text="Copy CSVs to data/", command=self.copy_data_files, bg="#e5f3ff").pack(side="left", padx=4)
        tk.Button(actions, text="Run Full Pipeline", command=self.run_pipeline, bg="#d2ffd2").pack(side="left", padx=8)
        tk.Label(actions, textvariable=self.status_var).pack(side="left", padx=12)

        logf = tk.LabelFrame(self.root, text="Logs", padx=8, pady=8)
        logf.pack(fill="both", expand=True, padx=12, pady=10)
        self.log = ScrolledText(logf, height=24)
        self.log.pack(fill="both", expand=True)

        self._write("LinerNet UI started.")
        self._write("1) Save Gemini key once")
        self._write("2) Select 4 CSV files")
        self._write("3) Click 'Run Full Pipeline'")

    def _write(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.root.update_idletasks()

    def pick_file(self, required_name: str):
        path = filedialog.askopenfilename(
            title=f"Select {required_name}",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.file_vars[required_name].set(path)

    def save_key(self):
        key = self.key_var.get().strip()
        if not key:
            messagebox.showerror("Missing key", "Please enter a Gemini API key.")
            return
        save_gemini_key(PROJECT_ROOT, key)
        self.status_var.set("Gemini key saved to .env")
        self._write("Saved GEMINI_API_KEY to .env")

    def copy_data_files(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        for required_name, target_name in REQUIRED_FILES.items():
            src = self.file_vars[required_name].get().strip()
            if not src:
                raise ValueError(f"Please select file for {required_name}")
            if not os.path.isfile(src):
                raise FileNotFoundError(f"File not found: {src}")
            dst = os.path.join(DATA_DIR, target_name)
            shutil.copy2(src, dst)
            self._write(f"Copied {required_name} -> data/{target_name}")
        self.status_var.set("CSV files copied.")

    def run_pipeline(self):
        def task():
            try:
                self.status_var.set("Running...")
                if self.key_var.get().strip():
                    save_gemini_key(PROJECT_ROOT, self.key_var.get().strip())
                self.copy_data_files()

                def logger(line: str):
                    self._write(line)

                summary = run_all(gemini_key=None, log=logger)
                self._write(f"Pipeline summary: {summary}")
                self.status_var.set("Completed.")
                messagebox.showinfo("Done", "Pipeline completed. Check outputs/ folder.")
            except Exception as e:
                self._write(f"ERROR: {type(e).__name__}: {e}")
                self.status_var.set("Failed.")
                messagebox.showerror("Failed", str(e))

        threading.Thread(target=task, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

