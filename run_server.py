# run_server.py
import uvicorn
from server import create_app

def main():
    print("=== MT Server Launcher ===")
    # choose port
    port_input = input("Port [8000]: ").strip()
    try:
        port = int(port_input) if port_input else 8000
    except ValueError:
        print("Invalid port, using 8000.")
        port = 8000

    # choose CORS mode
    mode = input("Allow all origins? [Y/n]: ").strip().lower()
    allow_all = (mode == "" or mode == "y" or mode == "yes")

    if allow_all:
        print("→ CORS: allowing ALL origins (*)")
        app = create_app(allow_all_origins=True)
    else:
        origins_str = input(
            "Enter allowed origins (comma-separated, e.g. https://freefu.it,https://fifahub.vercel.app):\n> "
        ).strip()
        origins = [o.strip() for o in origins_str.split(",") if o.strip()]
        if not origins:
            print("No valid origins provided, falling back to allow-all.")
            app = create_app(allow_all_origins=True)
        else:
            print("→ CORS: specific origins:", origins)
            app = create_app(allow_all_origins=False, specific_origins=origins)

    print(f"Starting server on 0.0.0.0:{port} ...")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
