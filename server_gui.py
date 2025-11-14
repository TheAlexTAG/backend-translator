import threading
import tkinter as tk
from tkinter import ttk, messagebox
import uvicorn
from server import create_app

server = None
server_thread = None

def start_server():
    global server, server_thread
    if server_thread and server_thread.is_alive():
        messagebox.showinfo("Info", "Server is already running.")
        return

    port_text = port_entry.get().strip() or "8000"
    try:
        port = int(port_text)
    except ValueError:
        messagebox.showerror("Error", f"Invalid port: {port_text}")
        return

    mode = cors_mode.get()
    if mode == "all":
        allow_all = True
        origins = []
    else:
        allow_all = False
        raw = origins_entry.get().strip()
        origins = [o.strip() for o in raw.split(",") if o.strip()]
        if not origins:
            messagebox.showerror("Error", "Please enter at least one origin, or choose 'Allow all'.")
            return

    app = create_app(allow_all_origins=allow_all, specific_origins=origins)
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)

    def run_server():
        try:
            server.run()
        except Exception as e:
            print("Server error:", e)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    status_var.set(f"Running on 0.0.0.0:{port} (CORS: {'*' if allow_all else ', '.join(origins)})")
    start_btn.config(state="disabled")
    stop_btn.config(state="normal")

def stop_server():
    global server, server_thread
    if not server:
        return
    server.should_exit = True
    status_var.set("Stopping server...")
    # we don't join the thread here to avoid freezing the UI
    server = None
    start_btn.config(state="normal")
    stop_btn.config(state="disabled")
    status_var.set("Server stopped.")

# --- Tkinter UI setup ---
root = tk.Tk()
root.title("MT Backend Server")

main_frame = ttk.Frame(root, padding=10)
main_frame.grid(row=0, column=0, sticky="nsew")

# Port
ttk.Label(main_frame, text="Port:").grid(row=0, column=0, sticky="w")
port_entry = ttk.Entry(main_frame, width=10)
port_entry.insert(0, "8000")
port_entry.grid(row=0, column=1, sticky="w", padx=(5, 0))

# CORS mode
cors_mode = tk.StringVar(value="all")
ttk.Label(main_frame, text="CORS mode:").grid(row=1, column=0, sticky="w", pady=(8, 0))

all_radio = ttk.Radiobutton(main_frame, text="Allow all origins (*)", variable=cors_mode, value="all")
all_radio.grid(row=1, column=1, sticky="w", pady=(8, 0))

specific_radio = ttk.Radiobutton(main_frame, text="Specific origins", variable=cors_mode, value="specific")
specific_radio.grid(row=2, column=1, sticky="w")

# Origins entry (only used when "specific" is selected)
ttk.Label(main_frame, text="Origins (comma-separated):").grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
origins_entry = ttk.Entry(main_frame, width=50)
origins_entry.insert(0, "https://freefu.it, https://fifahub.vercel.app")
origins_entry.grid(row=4, column=0, columnspan=2, sticky="we")

# Buttons
btn_frame = ttk.Frame(main_frame)
btn_frame.grid(row=5, column=0, columnspan=2, pady=(10, 0), sticky="we")

start_btn = ttk.Button(btn_frame, text="Start server", command=start_server)
start_btn.grid(row=0, column=0, padx=(0, 5))

stop_btn = ttk.Button(btn_frame, text="Stop server", command=stop_server, state="disabled")
stop_btn.grid(row=0, column=1)

# Status label
status_var = tk.StringVar(value="Server not running.")
status_label = ttk.Label(main_frame, textvariable=status_var, foreground="gray")
status_label.grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))

root.mainloop()
