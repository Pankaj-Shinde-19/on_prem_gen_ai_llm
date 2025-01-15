import sys
import time
import logging
import asyncio
import os
# import PDFHandler
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler
from pdf_filehandler import PDFHandler

if __name__ == "__main__":
    # Configure logging for the scriptpython
    logging.basicConfig(
        level=logging.INFO,  # Set log level to INFO
        format="%(asctime)s - %(message)s",  # Log message format
        datefmt="%Y-%m-%d %H:%M:%S",  # Date format for log entries
    )

    # Path to the directory containing PDF files
    path = r"C:\Users\PankajShinde\PycharmProjects\testing-on-premise-gen-ai\pdfs"

    # Initialize an instance of the PDFHandler class
    pdf_handler = PDFHandler()

    # Process existing PDF files in the specified directory
    if os.path.exists(path):
        import glob  # Import glob for pattern matching

        # Find all PDF files in the directory
        pdf_files = glob.glob(os.path.join(path, "*.pdf"))

        # Iterate through each PDF file and process it
        for pdf_file in pdf_files:
            print(f"Processing existing file: {pdf_file}")
            asyncio.run(pdf_handler.process_pdf_for_embeddings(pdf_file))

    # Set up Watchdog to monitor the directory for new PDF files
    event_handler = LoggingEventHandler()  # Default event handler for logging file events
    observer = Observer()  # Create a Watchdog Observer instance

    # Schedule the observer to watch the specified directory
    observer.schedule(pdf_handler, path, recursive=True)  # Recursive=True to monitor subdirectories
    observer.start()  # Start the observer

    try:
        # Keep the script running indefinitely
        while True:
            time.sleep(1)  # Sleep to avoid high CPU usage
    except KeyboardInterrupt:
        # Stop the observer if the script is interrupted
        observer.stop()
    observer.join()  # Wait for the observer thread to finish
