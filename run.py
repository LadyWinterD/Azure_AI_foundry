import subprocess
import time
import threading
from pyngrok import ngrok

# Ngrok configuration
NGROK_AUTH_TOKEN = "your_auth_token"

def start_ngrok(port=8001, auth_token=None):
    """Set up ngrok tunnel."""
    if auth_token:
        ngrok.set_auth_token(auth_token)
    
    print(f"Starting ngrok tunnel on port {port}...")
    public_url = ngrok.connect(port).public_url
    print(f"Public URL for Chainlit: {public_url}")
    return public_url

def stop_ngrok():
    """Stop ngrok tunnel."""
    print("Stopping ngrok...")
    ngrok.kill()

def run_chainlit():
    """Run the Chainlit application."""
    try:
        subprocess.run(
            ["chainlit", "run", "client.py", "--port", "8001"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running Chainlit: {e}")
    except KeyboardInterrupt:
        print("Chainlit process interrupted")

def main():
    """Launches Chainlit application with ngrok tunnel"""
    print("Starting Floor Plan Generator...")

    # Start ngrok tunnel for Chainlit
    public_url = start_ngrok(port=8001, auth_token=NGROK_AUTH_TOKEN)

    # Start Chainlit in a separate thread
    print("Starting Chainlit application...")
    chainlit_thread = threading.Thread(target=run_chainlit)
    chainlit_thread.daemon = True
    chainlit_thread.start()

    try:
        while True:
            time.sleep(1)  # Keep the main thread running
    except KeyboardInterrupt:
        print("Shutting down...")

    finally:
        # Stop ngrok when done
        stop_ngrok()

if __name__ == "__main__":
    main()
