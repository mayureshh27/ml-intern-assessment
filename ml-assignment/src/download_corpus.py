"""
Download and prepare corpus from Project Gutenberg.

This script downloads "Alice's Adventures in Wonderland" from Project Gutenberg
and saves it to the data directory for training the trigram model.
"""

import requests
import os
import re


def download_gutenberg_book(book_id, output_path):
    """
    Download a book from Project Gutenberg with robust error handling.
    
    Args:
        book_id (int): The Project Gutenberg book ID
        output_path (str): Path where the book should be saved
    
    Returns:
        bool: True if download successful, False otherwise
    """
    # Try multiple URL formats
    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]
    
    print(f"📥 Attempting to download book {book_id} from Project Gutenberg...")
    print(f"   Will try {len(urls)} different URL formats\n")
    
    for i, url in enumerate(urls, 1):
        try:
            print(f"[Attempt {i}/{len(urls)}] Trying: {url}")
            
            # Use streaming to handle large files and connection issues better
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()
            
            print(f"   ✓ Connection successful (Status: {response.status_code})")
            
            # Get content length if available
            content_length = response.headers.get('content-length')
            if content_length:
                print(f"   ✓ File size: {int(content_length):,} bytes")
            
            # Download in chunks to handle large files
            print(f"   ⏳ Downloading content...")
            chunks = []
            downloaded = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    chunks.append(chunk)
                    downloaded += len(chunk)
            
            text = b''.join(chunks).decode('utf-8', errors='ignore')
            print(f"   ✓ Downloaded {downloaded:,} bytes")
            
            # Extract the main text (remove Project Gutenberg header/footer)
            print(f"   🧹 Cleaning text (removing headers/footers)...")
            text = clean_gutenberg_text(text)
            print(f"   ✓ Cleaned text length: {len(text):,} characters")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save to file
            print(f"   💾 Saving to: {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"\n✅ SUCCESS! Book downloaded and saved successfully!")
            print(f"   Final text length: {len(text):,} characters")
            print(f"   Word count: ~{len(text.split()):,} words")
            return True
            
        except requests.exceptions.Timeout as e:
            print(f"   ❌ Timeout error: Connection took too long")
            print(f"      Details: {str(e)}")
            
        except requests.exceptions.ConnectionError as e:
            print(f"   ❌ Connection error: Could not connect to server")
            print(f"      Details: {str(e)}")
            
        except requests.exceptions.HTTPError as e:
            print(f"   ❌ HTTP error: {e.response.status_code}")
            if e.response.status_code == 404:
                print(f"      This URL format doesn't exist for book {book_id}")
            else:
                print(f"      Details: {str(e)}")
                
        except requests.exceptions.ChunkedEncodingError as e:
            print(f"   ❌ Download interrupted: Connection broken during transfer")
            print(f"      Details: {str(e)}")
            
        except UnicodeDecodeError as e:
            print(f"   ❌ Encoding error: Could not decode text")
            print(f"      Details: {str(e)}")
            
        except Exception as e:
            print(f"   ❌ Unexpected error: {type(e).__name__}")
            print(f"      Details: {str(e)}")
        
        print()  # Blank line between attempts
    
    print(f"❌ FAILED: All {len(urls)} download attempts failed")
    print(f"   Possible issues:")
    print(f"   - Unstable internet connection")
    print(f"   - Project Gutenberg server issues")
    print(f"   - Firewall or proxy blocking the connection")
    print(f"\n   💡 Suggestion: You can manually download the book from:")
    print(f"      https://www.gutenberg.org/ebooks/{book_id}")
    print(f"      and save it as: {output_path}")
    return False


def clean_gutenberg_text(text):
    """
    Remove Project Gutenberg header and footer from the text.
    
    Args:
        text (str): Raw text from Project Gutenberg
    
    Returns:
        str: Cleaned text with header/footer removed
    """
    # Find the start of the actual book content
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT"
    ]
    
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Find the end of this line
            start_idx = text.find('\n', idx) + 1
            break
    
    # Find the end of the actual book content
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg"
    ]
    
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break
    
    # Extract the main content
    cleaned_text = text[start_idx:end_idx].strip()
    
    return cleaned_text


def main():
    """Main function to download Alice's Adventures in Wonderland."""
    # Alice's Adventures in Wonderland - Book ID: 11
    book_id = 11
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    output_path = os.path.join(data_dir, 'alice_in_wonderland.txt')
    
    success = download_gutenberg_book(book_id, output_path)
    
    if success:
        print("\n✓ Corpus download complete!")
        print(f"  You can now train your model using: python generate.py")
    else:
        print("\n✗ Failed to download corpus")
        print("  Please check your internet connection and try again")


if __name__ == "__main__":
    main()
