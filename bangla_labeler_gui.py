import os
import sys
import threading
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Configuration
MODEL_PATH = r"./hf_bangla_multilabel_best"
TEXT_COLUMN = 'text'  # CSV column containing text

LABEL_MAPPING = {
    "LABEL_0": "bully",
    "LABEL_1": "sexual",
    "LABEL_2": "religious",
    "LABEL_3": "threat",
    "LABEL_4": "spam"
}

LABEL_COLORS = {
    "bully": "#FF6B6B",
    "sexual": "#4ECDC4",
    "religious": "#45B7D1",
    "threat": "#96CEB4",
    "spam": "#FFEAA7"
}

# check libs
try:
    import chardet
except Exception:
    chardet = None

# Global model variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer = None
_model = None
_model_loaded = False
_model_load_error = None
_model_load_lock = threading.Lock()

# Model loading with proper synchronization
def load_model_async(model_path, status_callback=None):
    global _tokenizer, _model, _model_loaded, _model_load_error
    
    try:
        if status_callback:
            status_callback("Loading tokenizer...")
        
        with _model_load_lock:
            _tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            
        if status_callback:
            status_callback("Loading model (this may take a while)...")
        
        with _model_load_lock:
            _model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            _model.to(device)
            _model.eval()
            _model_loaded = True
            
        if status_callback:
            status_callback("Model loaded successfully")
            
    except Exception as e:
        with _model_load_lock:
            _model_load_error = str(e)
            _model_loaded = False
        if status_callback:
            status_callback(f"Model load failed: {e}")

def is_model_loaded():
    """Check if model is ready to use"""
    with _model_load_lock:
        return _model_loaded and _tokenizer is not None and _model is not None

def wait_for_model(timeout=30):
    """Wait for model to load with timeout"""
    import time
    start_time = time.time()
    while not is_model_loaded():
        if time.time() - start_time > timeout:
            return False
        time.sleep(0.1)
    return True

# Prediction utilities

def detect_encoding(path):
    try:
        raw = open(path, 'rb').read()
        if chardet:
            enc = chardet.detect(raw).get('encoding')
            if enc:
                return enc
        # fallback guesses
        for enc in ('utf-8', 'utf-8-sig', 'utf-16', 'latin1', 'cp1252'):
            try:
                raw.decode(enc)
                return enc
            except Exception:
                continue
    except Exception:
        pass
    return 'utf-8'

def read_csv_auto(path):
    enc = detect_encoding(path)
    return pd.read_csv(path, encoding=enc)

def predict_text_local(text):
    """Return dict label->score"""
    global _tokenizer, _model, _model_loaded
    
    # Check if model is loaded
    if not is_model_loaded():
        if _model_load_error:
            raise RuntimeError(f"Model failed to load: {_model_load_error}")
        else:
            raise RuntimeError("Model not loaded yet. Please wait until status shows 'Model loaded'.")
    
    try:
        with _model_load_lock:
            inputs = _tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256).to(device)
            with torch.no_grad():
                outputs = _model(**inputs)
                logits = outputs.logits.squeeze(0)
                probs = torch.sigmoid(logits)
        
        scores = probs.tolist()
        if isinstance(scores, float):
            scores = [scores]
        return {LABEL_MAPPING[f"LABEL_{i}"]: float(round(scores[i], 4)) for i in range(len(scores))}
    
    except Exception as e:
        raise RuntimeError(f"Prediction error: {str(e)}")

# UI auto-resizing
class ModernApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bangla Toxic Comment Classifier")
        self.geometry('1200x800')  # Slightly larger default size
        self.minsize(1000, 600)   # Minimum window size
        self.configure(bg='#f6f7fb')
        
        # Make the window resizable
        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self._create_styles()
        
        # Start model loading immediately
        self.model_status_text = tk.StringVar(value="Model: loading...")
        threading.Thread(target=load_model_async, args=(MODEL_PATH, self._update_model_status), daemon=True).start()
        
        self._build_ui()
        self.csv_results = None
        self.csv_df = None
        
        # Start status polling
        self.after(500, self._poll_model_status)
        
        # Bind resize event
        self.bind('<Configure>', self._on_resize)

    def _on_resize(self, event):
        """Handle window resize events"""
        if hasattr(self, 'main'):
            pass

    def _update_model_status(self, message):
        """Callback for model loading status updates"""
        self.model_status_text.set(f"Model: {message}")

    def _create_styles(self):
        style = ttk.Style(self)
        style.theme_use('default')
        style.configure('Sidebar.TFrame', background='#263238')
        style.configure('Card.TFrame', background='white', relief='flat')
        style.configure('Accent.TButton', background='#1976D2', foreground='white', font=('Segoe UI', 10, 'bold'))
        style.map('Accent.TButton', background=[('active', '#115293')])

    def _build_ui(self):
        # Main container with grid layout for proper resizing
        main_container = tk.Frame(self, bg='#f6f7fb')
        main_container.grid(row=0, column=0, sticky='nsew')
        main_container.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)

        # Sidebar - fixed width but stretches vertically
        sidebar = tk.Frame(main_container, bg='#263238', width=220)
        sidebar.grid(row=0, column=0, sticky='ns')
        sidebar.rowconfigure(1, weight=1)  # Make space between buttons and status stretchable
        
        # Prevent sidebar from shrinking too much
        sidebar.grid_propagate(False)

        title = tk.Label(sidebar, text='Bangla Classifier', bg='#263238', fg='white', 
                        font=('Segoe UI', 14, 'bold'), anchor='w')
        title.pack(fill='x', padx=12, pady=18)

        # Sidebar buttons container with flexible spacing
        button_container = tk.Frame(sidebar, bg='#263238')
        button_container.pack(fill='x', pady=10)

        self.btn_home = self._create_sidebar_button(button_container, 'Home', self.show_home)
        self.btn_text = self._create_sidebar_button(button_container, 'Single Text', self.show_text)
        self.btn_csv = self._create_sidebar_button(button_container, 'CSV Upload', self.show_csv)
        self.btn_about = self._create_sidebar_button(button_container, 'About', self.show_about)

        # Flexible spacer to push status to bottom
        spacer = tk.Frame(sidebar, bg='#263238', height=20)
        spacer.pack(fill='x', expand=True)

        # Model status at bottom
        self.model_status = tk.Label(sidebar, textvariable=self.model_status_text, bg='#263238', 
                                   fg='white', font=('Segoe UI', 9), anchor='w')
        self.model_status.pack(fill='x', side='bottom', padx=12, pady=12)

        # Main content area - stretches in both directions
        self.main = tk.Frame(main_container, bg='#f6f7fb')
        self.main.grid(row=0, column=1, sticky='nsew')
        self.main.rowconfigure(1, weight=1)  # Content area stretches
        self.main.columnconfigure(0, weight=1)

        # Header card - fixed height
        header = tk.Frame(self.main, bg='#ffffff')
        header.grid(row=0, column=0, sticky='ew', padx=18, pady=18)
        header.columnconfigure(0, weight=1)
        
        tk.Label(header, text='Bangla Toxic Content Detector', bg='white', fg='#2c3e50', 
                font=('Segoe UI', 18, 'bold')).grid(row=0, column=0, sticky='w', padx=12, pady=8)
        tk.Label(header, text='multi-label classification (bully, sexual, religious, threat, spam)', 
                bg='white', fg='#6b7280', font=('Segoe UI', 10)).grid(row=1, column=0, sticky='w', padx=12, pady=(0,12))

        # Content container - stretches to fill available space
        self.container = tk.Frame(self.main, bg='#f6f7fb')
        self.container.grid(row=1, column=0, sticky='nsew', padx=18, pady=(0,18))
        self.container.rowconfigure(0, weight=1)
        self.container.columnconfigure(0, weight=1)

        # Create pages
        self.pages = {}
        for Page in (HomePage, TextPage, CSVPage, AboutPage):
            page = Page(self.container, self)
            self.pages[Page.__name__] = page
            page.grid(row=0, column=0, sticky='nsew')
            page.rowconfigure(0, weight=1)
            page.columnconfigure(0, weight=1)

        self.show_home()

    def _create_sidebar_button(self, parent, text, command):
        """Create a styled sidebar button"""
        btn = tk.Button(parent, text=text, command=command, bg='#455A64', fg='white', 
                       relief='flat', anchor='w', font=('Segoe UI', 10))
        btn.pack(fill='x', padx=12, pady=6, ipady=8)
        return btn

    def _poll_model_status(self):
        """Poll model status and update UI accordingly"""
        if is_model_loaded():
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model_status_text.set(f"Model: loaded ({device_type})")
        elif _model_load_error:
            self.model_status_text.set(f"Model: error - check path")
        else:
            # Still loading - update the dots animation
            current = self.model_status_text.get()
            if current.endswith('...'):
                self.model_status_text.set('Model: loading')
            else:
                dots = current.count('.') + 1
                if dots > 3:
                    dots = 1
                self.model_status_text.set('Model: loading' + '.' * dots)
        
        self.after(800, self._poll_model_status)

    def show_home(self):
        self._raise_page('HomePage')

    def show_text(self):
        self._raise_page('TextPage')

    def show_csv(self):
        self._raise_page('CSVPage')

    def show_about(self):
        self._raise_page('AboutPage')

    def _raise_page(self, name):
        self.pages[name].tkraise()

    def is_model_ready(self):
        """Check if model is ready and show message if not"""
        if not is_model_loaded():
            if _model_load_error:
                messagebox.showerror("Model Error", f"Model failed to load:\n{_model_load_error}\n\nPlease check your MODEL_PATH configuration.")
            else:
                messagebox.showwarning("Model Loading", "Model is still loading. Please wait until the status shows 'Model loaded'.")
            return False
        return True
    
# Pages with auto-resizing
class HomePage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg='#f6f7fb')
        self.app = app
        
        # Configure grid for resizing
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        
        # Main content card that stretches
        card = tk.Frame(self, bg='white', bd=0)
        card.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        card.rowconfigure(1, weight=1)  # Content area stretches
        card.columnconfigure(0, weight=1)
        
        # Title
        tk.Label(card, text='Welcome', font=('Segoe UI', 16, 'bold'), bg='white'
                ).grid(row=0, column=0, sticky='w', padx=12, pady=12)
        
        # Content area that can scroll if needed
        content_frame = tk.Frame(card, bg='white')
        content_frame.grid(row=1, column=0, sticky='nsew', padx=12, pady=12)
        content_frame.columnconfigure(0, weight=1)
        
        tk.Label(content_frame, text='Use the sidebar to analyze single texts or batch CSV files.', 
                bg='white', fg='#374151', font=('Segoe UI', 11)
                ).grid(row=0, column=0, sticky='w', pady=4)
        
        # Add model status info
        status_frame = tk.Frame(content_frame, bg='white')
        status_frame.grid(row=1, column=0, sticky='w', pady=20)
        
        tk.Label(status_frame, text='Current Status:', font=('Segoe UI', 11, 'bold'), 
                bg='white').grid(row=0, column=0, sticky='w')
        tk.Label(status_frame, textvariable=app.model_status_text, bg='white', 
                fg='#666', font=('Segoe UI', 10)).grid(row=1, column=0, sticky='w', pady=2)

class TextPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg='#f6f7fb')
        self.app = app
        
        # Configure grid for proper resizing
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        
        # Main card that stretches
        card = tk.Frame(self, bg='white', bd=0)
        card.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        card.rowconfigure(1, weight=1)  # Text area stretches
        card.columnconfigure(0, weight=1)

        # Top controls - fixed height
        top_frame = tk.Frame(card, bg='white')
        top_frame.grid(row=0, column=0, sticky='ew', padx=12, pady=8)
        top_frame.columnconfigure(1, weight=1)  # Textbox will expand

        tk.Label(top_frame, text='Enter text (one per line for multiple):', bg='white', 
                font=('Segoe UI', 11, 'bold')).grid(row=0, column=0, columnspan=3, sticky='w', pady=(0,8))

        # Text area with scrollbar - stretches
        text_frame = tk.Frame(card, bg='white')
        text_frame.grid(row=1, column=0, sticky='nsew', padx=12, pady=(0,8))
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        
        self.textbox = scrolledtext.ScrolledText(text_frame, font=('Segoe UI', 11), wrap=tk.WORD)
        self.textbox.grid(row=0, column=0, sticky='nsew')

        # Controls frame - fixed height
        controls = tk.Frame(card, bg='white')
        controls.grid(row=2, column=0, sticky='ew', padx=12, pady=6)
        controls.columnconfigure(1, weight=1)  # Spacer between controls and status

        tk.Label(controls, text='Threshold:', bg='white').grid(row=0, column=0, sticky='w')
        self.threshold = tk.Entry(controls, width=6)
        self.threshold.insert(0, '0.5')
        self.threshold.grid(row=0, column=1, sticky='w', padx=6)
        
        self.analyze_btn = tk.Button(controls, text='Analyze', bg='#1976D2', fg='white', 
                                   command=self.run_analysis)
        self.analyze_btn.grid(row=0, column=2, sticky='w', padx=10)
        
        # Add model status indicator
        self.status_indicator = tk.Label(controls, text='', bg='white', fg='red', 
                                       font=('Segoe UI', 9))
        self.status_indicator.grid(row=0, column=3, sticky='w', padx=10)
        
        # Spacer to push status to the right
        controls.columnconfigure(4, weight=1)

        # Results area - stretches to fill remaining space
        self.results_area = tk.Frame(card, bg='white')
        self.results_area.grid(row=3, column=0, sticky='nsew', padx=12, pady=8)
        self.results_area.rowconfigure(0, weight=1)
        self.results_area.columnconfigure(0, weight=1)
        
        # Update status indicator
        self.after(1000, self._update_status_indicator)

    def _update_status_indicator(self):
        """Update the model status indicator"""
        if is_model_loaded():
            self.status_indicator.config(text='✓ Ready', fg='green')
            self.analyze_btn.config(state='normal', bg='#1976D2')
        elif _model_load_error:
            self.status_indicator.config(text='✗ Error', fg='red')
            self.analyze_btn.config(state='disabled', bg='gray')
        else:
            self.status_indicator.config(text='⏳ Loading...', fg='orange')
            self.analyze_btn.config(state='disabled', bg='gray')
        
        self.after(1000, self._update_status_indicator)

    def run_analysis(self):
        # Check if model is ready
        if not self.app.is_model_ready():
            return
            
        text_content = self.textbox.get('1.0', tk.END).strip()
        if not text_content:
            messagebox.showerror('Error', 'Enter text to analyze')
            return
        try:
            th = float(self.threshold.get())
            if not (0 <= th <= 1):
                raise ValueError("Threshold must be between 0 and 1")
        except Exception as e:
            messagebox.showerror('Error', f'Threshold must be a number between 0 and 1\n{str(e)}')
            return
            
        texts = [t.strip() for t in text_content.split('\n') if t.strip()]
        
        # Show progress
        self.analyze_btn.config(text='Analyzing...', state='disabled', bg='gray')
        
        # run predictions in background so UI doesn't freeze
        threading.Thread(target=self._predict_and_show, args=(texts, th), daemon=True).start()

    def _predict_and_show(self, texts, threshold):
        # clear previous results
        for w in self.results_area.winfo_children():
            w.destroy()
            
        results = []
        try:
            total_texts = len(texts)
            progress_frame = tk.Frame(self.results_area, bg='white')
            progress_frame.grid(row=0, column=0, sticky='ew', pady=10)
            progress_frame.columnconfigure(0, weight=1)
            
            progress_label = tk.Label(progress_frame, text=f"Processing {total_texts} texts...", bg='white')
            progress_label.grid(row=0, column=0, sticky='w')
            progress_bar = ttk.Progressbar(progress_frame, orient='horizontal', mode='determinate', maximum=total_texts)
            progress_bar.grid(row=1, column=0, sticky='ew', pady=5)
            
            for i, text in enumerate(texts):
                pred = predict_text_local(text)
                results.append({
                    'text': text, 
                    'predictions': pred, 
                    'has_toxic': any(v >= threshold for v in pred.values())
                })
                progress_bar['value'] = i + 1
                progress_label.config(text=f"Processed {i+1}/{total_texts} texts...")
                self.update_idletasks()
                
            # Remove progress bar
            progress_frame.destroy()
            
        except Exception as e:
            messagebox.showerror('Error', f'Prediction failed: {e}')
            self.analyze_btn.config(text='Analyze', state='normal', bg='#1976D2')
            return

        # Show results
        self._display_results(results, threshold)
        self.analyze_btn.config(text='Analyze', state='normal', bg='#1976D2')

    def _display_results(self, results, threshold):
        # Clear results area
        for w in self.results_area.winfo_children():
            w.destroy()

        # Create a container for results that can scroll if needed
        results_container = tk.Frame(self.results_area, bg='white')
        results_container.grid(row=0, column=0, sticky='nsew')
        results_container.rowconfigure(0, weight=1)
        results_container.columnconfigure(0, weight=1)

        # Create notebook for multiple texts
        if len(results) > 1:
            notebook = ttk.Notebook(results_container)
            notebook.grid(row=0, column=0, sticky='nsew')
            
            for i, res in enumerate(results):
                tab = ttk.Frame(notebook)
                notebook.add(tab, text=f'Text {i+1}')
                self._create_result_tab(tab, res, threshold, i+1)
        else:
            # Single result
            self._create_result_tab(results_container, results[0], threshold, 1)

        # Export buttons at bottom
        exf = tk.Frame(self.results_area, bg='white')
        exf.grid(row=1, column=0, sticky='ew', pady=8)
        exf.columnconfigure(0, weight=1)  # Center the buttons
        
        button_frame = tk.Frame(exf, bg='white')
        button_frame.grid(row=0, column=0)
        
        tk.Button(button_frame, text='Export PDF', bg='#4CAF50', fg='white', 
                 command=lambda: export_to_pdf(results, filedialog.asksaveasfilename(defaultextension='.pdf'))).pack(side='left', padx=6)
        tk.Button(button_frame, text='Export Image', bg='#2196F3', fg='white', 
                 command=lambda: export_to_image(results, filedialog.asksaveasfilename(defaultextension='.png'))).pack(side='left', padx=6)
        tk.Button(button_frame, text='Export Excel', bg='#FF9800', fg='white', 
                 command=lambda: export_multiple_to_excel(results, filedialog.asksaveasfilename(defaultextension='.xlsx'))).pack(side='left', padx=6)

    def _create_result_tab(self, parent, result, threshold, text_num):
        # Configure the parent for resizing
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)
        
        # Text content
        text_frame = tk.Frame(parent, bg='white')
        text_frame.grid(row=0, column=0, sticky='ew', pady=5)
        text_frame.columnconfigure(0, weight=1)
        
        tk.Label(text_frame, text=f'Text {text_num}:', bg='white', font=('Segoe UI', 10, 'bold')
                ).grid(row=0, column=0, sticky='w')
        
        text_display = scrolledtext.ScrolledText(text_frame, height=3, font=('Segoe UI', 10), wrap=tk.WORD)
        text_display.grid(row=1, column=0, sticky='ew', pady=2)
        text_display.insert('1.0', result['text'])
        text_display.config(state='disabled')

        # Results table
        table_frame = tk.Frame(parent, bg='white')
        table_frame.grid(row=1, column=0, sticky='nsew', pady=5)
        table_frame.rowconfigure(1, weight=1)
        table_frame.columnconfigure(0, weight=1)
        
        tk.Label(table_frame, text='Classification Results:', bg='white', font=('Segoe UI', 10, 'bold')
                ).grid(row=0, column=0, sticky='w')
        
        # Create a frame for the table with scrollbars
        table_container = tk.Frame(table_frame, bg='white')
        table_container.grid(row=1, column=0, sticky='nsew')
        table_container.rowconfigure(0, weight=1)
        table_container.columnconfigure(0, weight=1)
        
        # Create treeview with scrollbars
        tree_frame = tk.Frame(table_container, bg='white')
        tree_frame.grid(row=0, column=0, sticky='nsew')
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        
        table = ttk.Treeview(tree_frame, columns=('label','score','status'), show='headings', height=6)
        table.heading('label', text='Label')
        table.heading('score', text='Score')
        table.heading('status', text='Status')
        table.column('label', width=120)
        table.column('score', width=80)
        table.column('status', width=120)
        
        # Add scrollbars
        v_scroll = ttk.Scrollbar(tree_frame, orient='vertical', command=table.yview)
        h_scroll = ttk.Scrollbar(tree_frame, orient='horizontal', command=table.xview)
        table.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        table.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        
        for label, score in result['predictions'].items():
            status = '🚩 FLAGGED' if score >= threshold else '✅ Clean'
            table.insert('', 'end', values=(label, f'{score:.4f}', status))

        # Summary
        summary_frame = tk.Frame(parent, bg='white')
        summary_frame.grid(row=2, column=0, sticky='ew', pady=5)
        
        summary_text = "🚨 Contains toxic content" if result['has_toxic'] else "✅ Clean text"
        color = "red" if result['has_toxic'] else "green"
        tk.Label(summary_frame, text=summary_text, bg='white', fg=color, 
                font=('Segoe UI', 11, 'bold')).pack()

class CSVPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg='#f6f7fb')
        self.app = app
        
        # Configure grid for resizing
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        
        card = tk.Frame(self, bg='white')
        card.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        card.rowconfigure(1, weight=1)  # Results area stretches
        card.columnconfigure(0, weight=1)

        # Top controls
        top = tk.Frame(card, bg='white')
        top.grid(row=0, column=0, sticky='ew', pady=8, padx=12)
        top.columnconfigure(1, weight=1)  # Path entry stretches
        
        tk.Label(top, text='Select CSV file (must contain a "text" column):', bg='white'
                ).grid(row=0, column=0, sticky='w')
        
        self.path_entry = tk.Entry(top)
        self.path_entry.grid(row=0, column=1, sticky='ew', padx=6)
        
        tk.Button(top, text='Browse', command=self.browse).grid(row=0, column=2, sticky='w', padx=6)
        
        self.analyze_btn = tk.Button(top, text='Analyze', bg='#27ae60', fg='white', command=self.analyze)
        self.analyze_btn.grid(row=0, column=3, sticky='w', padx=6)
        
        # Model status indicator
        self.status_indicator = tk.Label(top, text='', bg='white', fg='red', font=('Segoe UI', 9))
        self.status_indicator.grid(row=0, column=4, sticky='w', padx=10)

        # Settings and progress
        settings_progress = tk.Frame(card, bg='white')
        settings_progress.grid(row=1, column=0, sticky='ew', padx=12, pady=8)
        settings_progress.columnconfigure(1, weight=1)
        
        tk.Label(settings_progress, text='Threshold:', bg='white').grid(row=0, column=0, sticky='w')
        self.threshold = tk.Entry(settings_progress, width=6)
        self.threshold.insert(0, '0.5')
        self.threshold.grid(row=0, column=1, sticky='w', padx=6)

        # Progress bar
        self.prog = ttk.Progressbar(settings_progress, orient='horizontal', mode='determinate')
        self.prog.grid(row=1, column=0, columnspan=5, sticky='ew', pady=8)

        # Results area - stretches
        self.results_container = tk.Frame(card, bg='white')
        self.results_container.grid(row=2, column=0, sticky='nsew', padx=12, pady=8)
        self.results_container.rowconfigure(0, weight=1)
        self.results_container.columnconfigure(0, weight=1)
        
        # results list
        self.tree = None
        
        # Update status indicator
        self.after(1000, self._update_status_indicator)

    def _update_status_indicator(self):
        """Update the model status indicator"""
        if is_model_loaded():
            self.status_indicator.config(text='✓ Ready', fg='green')
            self.analyze_btn.config(state='normal', bg='#27ae60')
        elif _model_load_error:
            self.status_indicator.config(text='✗ Error', fg='red')
            self.analyze_btn.config(state='disabled', bg='gray')
        else:
            self.status_indicator.config(text='⏳ Loading...', fg='orange')
            self.analyze_btn.config(state='disabled', bg='gray')
        
        self.after(1000, self._update_status_indicator)

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[('CSV files','*.csv'), ('All files','*.*')])
        if path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, path)

    def analyze(self):
        # Check if model is ready
        if not self.app.is_model_ready():
            return
            
        path = self.path_entry.get().strip()
        if not path:
            messagebox.showerror('Error', 'Choose CSV file')
            return
        try:
            th = float(self.threshold.get())
            if not (0 <= th <= 1):
                raise ValueError("Threshold must be between 0 and 1")
        except Exception as e:
            messagebox.showerror('Error', f'Threshold must be a number between 0 and 1\n{str(e)}')
            return
            
        self.analyze_btn.config(text='Analyzing...', state='disabled', bg='gray')
        threading.Thread(target=self._analyze_bg, args=(path, th), daemon=True).start()

    def _analyze_bg(self, path, threshold):
        try:
            df = read_csv_auto(path)
        except Exception as e:
            messagebox.showerror('Error', f'Could not read CSV: {e}')
            self.analyze_btn.config(text='Analyze', state='normal', bg='#27ae60')
            return
            
        if TEXT_COLUMN not in df.columns:
            messagebox.showerror('Error', f"CSV must contain a '{TEXT_COLUMN}' column")
            self.analyze_btn.config(text='Analyze', state='normal', bg='#27ae60')
            return
            
        n = len(df)
        self.prog['maximum'] = n
        self.prog['value'] = 0
        
        results = []
        out_rows = []
        
        try:
            for i, txt in enumerate(df[TEXT_COLUMN].astype(str)):
                pred = predict_text_local(txt)
                row = {k + '_score': v for k, v in pred.items()}
                for k, v in pred.items():
                    row[k + '_flag'] = v >= threshold
                out_rows.append(row)
                results.append({
                    'text': txt, 
                    'predictions': pred, 
                    'has_toxic': any(v >= threshold for v in pred.values())
                })
                self.prog['value'] = i + 1
                self.update_idletasks()
                
            # merge out_rows into df
            for col in out_rows[0].keys():
                df[col] = [r[col] for r in out_rows]
                
            self.app.csv_results = results
            self.app.csv_df = df
            self._show_table(results)
            messagebox.showinfo('Done', 'CSV analysis finished. Use Export buttons to save results.')
            
        except Exception as e:
            messagebox.showerror('Error', f'Analysis failed: {e}')
            
        finally:
            self.analyze_btn.config(text='Analyze', state='normal', bg='#27ae60')

    def _show_table(self, results):
        # Clear previous content
        for w in self.results_container.winfo_children():
            w.destroy()
            
        # Create a frame for table and buttons
        main_frame = tk.Frame(self.results_container, bg='white')
        main_frame.grid(row=0, column=0, sticky='nsew')
        main_frame.rowconfigure(0, weight=1)  # Table stretches
        main_frame.columnconfigure(0, weight=1)

        # Create table with scrollbars
        table_frame = tk.Frame(main_frame, bg='white')
        table_frame.grid(row=0, column=0, sticky='nsew', pady=(0,10))
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        
        cols = ['text'] + list(next(iter(results))['predictions'].keys())
        self.tree = ttk.Treeview(table_frame, columns=cols, show='headings')
        
        # Configure columns with auto-width
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=150, minwidth=100)
            
        # Add scrollbars
        v_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        h_scroll = ttk.Scrollbar(table_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        self.tree.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        
        # Populate table
        for r in results:
            vals = [r['text']] + [r['predictions'][k] for k in r['predictions']]
            self.tree.insert('', 'end', values=vals)
            
        # export buttons
        exf = tk.Frame(main_frame, bg='white')
        exf.grid(row=1, column=0, sticky='ew', pady=6)
        exf.columnconfigure(0, weight=1)  # Center the buttons
        
        button_frame = tk.Frame(exf, bg='white')
        button_frame.grid(row=0, column=0)
        
        tk.Button(button_frame, text='Export CSV', bg='#6c757d', fg='white', 
                 command=self._export_csv).pack(side='left', padx=6)
        tk.Button(button_frame, text='Export Excel', bg='#FF9800', fg='white', 
                 command=self._export_excel).pack(side='left', padx=6)
        tk.Button(button_frame, text='Export PDF Summary', bg='#4CAF50', fg='white', 
                 command=lambda: export_to_pdf(self.app.csv_results, filedialog.asksaveasfilename(defaultextension='.pdf'), 'csv')).pack(side='left', padx=6)

    def _export_csv(self):
        if self.app.csv_df is None:
            messagebox.showerror('Error', 'No results to export')
            return
        path = filedialog.asksaveasfilename(defaultextension='.csv')
        if path:
            self.app.csv_df.to_csv(path, index=False, encoding='utf-8')
            messagebox.showinfo('Saved', f'Saved to {path}')

    def _export_excel(self):
        if self.app.csv_df is None:
            messagebox.showerror('Error', 'No results to export')
            return
        path = filedialog.asksaveasfilename(defaultextension='.xlsx')
        if path:
            self.app.csv_df.to_excel(path, index=False)
            messagebox.showinfo('Saved', f'Saved to {path}')

class AboutPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg='#f6f7fb')
        self.app = app
        
        # Configure grid
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        
        card = tk.Frame(self, bg='white')
        card.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        card.columnconfigure(0, weight=1)
        
        content = tk.Frame(card, bg='white')
        content.grid(row=0, column=0, sticky='w', padx=12, pady=12)
        
        tk.Label(content, text='About', font=('Segoe UI', 14, 'bold'), bg='white'
                ).grid(row=0, column=0, sticky='w', pady=(0,10))
        tk.Label(content, text=
    "Bangla Toxic Content Classifier\n"
    "Version: 1.0.0\n\n"

    "AI-powered desktop application for detecting\n"
    "toxic Bangla text using multi-label NLP.\n\n"

    "Model Details:\n"
    "• Transformer-based (HuggingFace)\n"
    "• Multi-label classification\n"
    "• Labels: Bully, Sexual, Religious, Threat, Spam\n"
    "• Training samples: ~XX,000 Bangla texts\n"
    "• Validation accuracy: ~XX%\n\n"

    "System Info:\n"
    "• Build date: 2025\n"
    "• Inference: Offline (local model)\n"
    "• Device: Auto-detect (CPU / GPU)\n\n"

    "Author:\n"
    "• Chatur\n"
    "• Aspiring Machine Learning / NLP Engineer\n\n"

    "License:\n"
    "• MIT License\n\n"

    "Built with Python, PyTorch & Tkinter",
                bg='white').grid(row=1, column=0, sticky='w', pady=2)
        tk.Label(content, text=f'Model path: {MODEL_PATH}', bg='white', 
                font=('Segoe UI', 9)).grid(row=2, column=0, sticky='w', pady=5)
        
# Export helper wrappers (same as before)
def export_to_pdf(data, filename, export_type='text'):
    if not filename:
        return
    try:
        doc = SimpleDocTemplate(filename, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = [Paragraph('Bangla Classification Report', styles['Title']), Spacer(1,12)]
        
        if export_type == 'text':
            for i, item in enumerate(data, 1):
                elements.append(Paragraph(f'Text {i}', styles['Heading2']))
                elements.append(Paragraph(item['text'], styles['Normal']))
                table_data = [['Label', 'Score']]
                for k,v in item['predictions'].items():
                    table_data.append([k, f'{v:.4f}'])
                t = Table(table_data)
                t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),1,colors.black)]))
                elements.append(t)
                elements.append(Spacer(1,12))
        else:
            # CSV summary
            table_data = [['Label', 'Flagged Count', 'Percentage']]
            total = len(data)
            for label in LABEL_MAPPING.values():
                count = sum(1 for item in data if item['predictions'][label] >= 0.5)
                percentage = (count / total) * 100 if total > 0 else 0
                table_data.append([label, count, f'{percentage:.1f}%'])
            t = Table(table_data)
            t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),1,colors.black)]))
            elements.append(t)
            
        doc.build(elements)
        messagebox.showinfo('Saved', f'PDF saved to {filename}')
    except Exception as e:
        messagebox.showerror('Error', f'Failed to export PDF: {e}')

def export_to_image(data, filename):
    if not filename:
        return
    try:
        first = data[0]
        labels = list(first['predictions'].keys())
        scores = list(first['predictions'].values())
        colors = [LABEL_COLORS.get(l, '#888') for l in labels]
        plt.figure(figsize=(8,4))
        plt.bar(labels, scores, color=colors)
        plt.ylim(0,1)
        plt.ylabel('Confidence Score')
        plt.title('Text Classification Results')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        messagebox.showinfo('Saved', f'Image saved to {filename}')
    except Exception as e:
        messagebox.showerror('Error', f'Failed to export image: {e}')

def export_multiple_to_excel(results, filename):
    if not filename:
        return
    try:
        out = []
        for i, r in enumerate(results):
            row = {'text_number': i+1, 'text': r['text']}
            row.update({f'{k}_score': v for k, v in r['predictions'].items()})
            out.append(row)
        pd.DataFrame(out).to_excel(filename, index=False)
        messagebox.showinfo('Saved', f'Excel saved to {filename}')
    except Exception as e:
        messagebox.showerror('Error', f'Failed to export Excel: {e}')

# Run
if __name__ == '__main__':
    # Check if model path exists
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model path '{MODEL_PATH}' does not exist.")
        print("Please update the MODEL_PATH variable in the code.")
    
    app = ModernApp()
    app.mainloop()