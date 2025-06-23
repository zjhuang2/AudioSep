#!/usr/bin/env python3
"""
Scrape sounds from Freesound.org using their API
Requires: pip install freesound-python requests

Setup for OAuth2 (full file downloads):
1. Create app at: https://freesound.org/apiv2/apply/
   - Set Callback URL to: http://localhost:8080/callback
2. Add to secret_keys.py:
   FREESOUND_CLIENT_ID = "your_client_id"
   FREESOUND_CLIENT_SECRET = "your_client_secret"
   FREESOUND_API_KEY = "your_api_key"  # Still needed for basic auth
3. Run with --oauth flag
4. Browser opens → Authorize → Automatically captures the code

Without OAuth2: Downloads high-quality previews using API key only
"""

import os
import json
import time
import requests
from pathlib import Path
import argparse
import webbrowser
from urllib.parse import urlencode, parse_qs, urlparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
try:
    from secret_keys import FREESOUND_API_KEY, FREESOUND_CLIENT_ID, FREESOUND_CLIENT_SECRET
except ImportError:
    FREESOUND_API_KEY = None
    FREESOUND_CLIENT_ID = None
    FREESOUND_CLIENT_SECRET = None


# Global variable to store the authorization code
auth_code = None

class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth2 callback"""
    def do_GET(self):
        global auth_code
        
        # Parse the URL to get the code
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        
        if 'code' in params:
            auth_code = params['code'][0]
            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
                <html>
                <body>
                <h1>Authorization successful!</h1>
                <p>You can close this window and return to the terminal.</p>
                </body>
                </html>
            """)
        else:
            # Send error response
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"<h1>Error: No authorization code received</h1>")
    
    def log_message(self, format, *args):
        """Suppress log messages"""
        pass


def get_oauth2_token(client_id, client_secret, token_file="freesound_token.json"):
    """
    Get OAuth2 access token for Freesound downloads.
    This performs the full OAuth2 flow and saves the token for reuse.
    
    NOTE: Set your Freesound app's callback URL to: http://localhost:8080/callback
    """
    global auth_code
    auth_code = None
    
    # Check if we have a saved token
    if os.path.exists(token_file):
        try:
            with open(token_file, 'r') as f:
                token_data = json.load(f)
            
            # Test if token still works
            headers = {'Authorization': f'Bearer {token_data["access_token"]}'}
            resp = requests.get('https://freesound.org/apiv2/sounds/1/', headers=headers)
            
            if resp.status_code == 200:
                print("Using existing OAuth2 token")
                return token_data['access_token']
            else:
                print(f"Existing token invalid (HTTP {resp.status_code})")
                os.remove(token_file)
                
        except Exception as e:
            print(f"Token check failed: {e}")
            if os.path.exists(token_file):
                os.remove(token_file)
    
    # OAuth2 endpoints
    auth_base = 'https://freesound.org/apiv2/oauth2/authorize/'
    token_url = 'https://freesound.org/apiv2/oauth2/access_token/'
    redirect_uri = 'http://localhost:8080/callback'
    
    # Start local server to catch callback
    server = HTTPServer(('localhost', 8080), OAuthCallbackHandler)
    server_thread = threading.Thread(target=server.handle_request)
    server_thread.daemon = True
    server_thread.start()
    
    # Step 1: Build authorization URL
    auth_params = {
        'client_id': client_id,
        'response_type': 'code',
        'redirect_uri': redirect_uri
    }
    
    auth_url = f"{auth_base}?{urlencode(auth_params)}"
    
    print(f"\nStarting local server on http://localhost:8080")
    print(f"Opening browser for Freesound authorization...")
    print(f"If browser doesn't open, visit:\n{auth_url}")
    
    webbrowser.open(auth_url)
    
    # Wait for callback
    print("\nWaiting for authorization callback...")
    server_thread.join(timeout=300)  # 5 minute timeout
    
    if not auth_code:
        raise Exception("No authorization code received")
    
    print(f"Received authorization code!")
    
    # Step 3: Exchange code for access token
    token_data = {
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': 'authorization_code',
        'code': auth_code,
        'redirect_uri': redirect_uri
    }
    
    resp = requests.post(token_url, data=token_data)
    
    if resp.status_code != 200:
        raise Exception(f"Token exchange failed: {resp.text}")
    
    token_response = resp.json()
    
    # Save token for future use
    with open(token_file, 'w') as f:
        json.dump(token_response, f)
    
    print("OAuth2 token obtained and saved!")
    return token_response['access_token']


def download_from_freesound(api_key, query, num_sounds=50, output_dir="finetune_dataset", use_oauth=False):
    """
    Download sounds from Freesound.org
    
    Get API key from: https://freesound.org/apiv2/apply/
    For OAuth2: Also need client_id and client_secret from your Freesound app
    """
    import freesound
    
    client = freesound.FreesoundClient()
    
    if use_oauth:
        # Use OAuth2 for full file downloads
        if not FREESOUND_CLIENT_ID or not FREESOUND_CLIENT_SECRET:
            raise ValueError("OAuth2 requires FREESOUND_CLIENT_ID and FREESOUND_CLIENT_SECRET in secret_keys.py")
        
        access_token = get_oauth2_token(FREESOUND_CLIENT_ID, FREESOUND_CLIENT_SECRET)
        # Set OAuth2 token on the client
        client.set_token(access_token)
        oauth_headers = {'Authorization': f'Bearer {access_token}'}
    else:
        # Use API token (standard flow)
        client.set_token(api_key)
        oauth_headers = None
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Search for sounds
    if use_oauth:
        # Use direct API call for OAuth2 with pagination
        all_results = []
        page = 1
        page_size = 50  # Max allowed by Freesound API
        
        while len(all_results) < num_sounds:
            search_url = "https://freesound.org/apiv2/search/text/"
            params = {
                'query': query,
                'fields': 'id,name,description,duration,download',
                'page_size': page_size,
                'page': page
            }
            resp = requests.get(search_url, headers=oauth_headers, params=params)
            if resp.status_code != 200:
                raise Exception(f"Search failed: {resp.text}")
            
            search_data = resp.json()
            
            # Break if no more results
            if not search_data['results']:
                break
            
            # Convert to simple objects
            class SimpleSound:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)
            
            all_results.extend([SimpleSound(sound) for sound in search_data['results']])
            
            # Break if we've got all results
            if search_data['next'] is None:
                break
                
            page += 1
            time.sleep(0.2)  # Be respectful to the API
        
        results = all_results[:num_sounds]  # Limit to requested number
    else:
        results = client.text_search(query=query, fields="id,name,description,duration,download")
    
    downloaded = []
    count = 0
    
    for sound in results:
        if count >= num_sounds:
            break
            
        try:
            # Download sound
            filename = f"{sound.id}_{sound.name}.wav"
            filepath = os.path.join(output_dir, filename)
            
            # Skip if already downloaded
            if os.path.exists(filepath):
                print(f"Already exists: {filename}")
                downloaded.append({
                    "wav": filepath,
                    "caption": sound.description[:100]  # Truncate long descriptions
                })
                count += 1
                continue
            
            # Download the sound
            if use_oauth and hasattr(sound, 'download'):
                # OAuth2 download - get download URL and use requests
                download_url = sound.download
                resp = requests.get(download_url, headers=oauth_headers)
                if resp.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(resp.content)
                    print(f"Downloaded: {filename}")
                else:
                    print(f"Download failed for {filename}: HTTP {resp.status_code}")
                    continue
            else:
                # Use standard retrieve method (works with both API key and OAuth2)
                try:
                    sound.retrieve(filepath)
                    print(f"Downloaded: {filename}")
                except Exception as e:
                    print(f"Download failed for {filename}: {e}")
                    continue
            
            downloaded.append({
                "wav": filepath,
                "caption": sound.description[:100]
            })
            
            count += 1
            time.sleep(0.5)  # Be respectful to the API
            
        except Exception as e:
            print(f"Error downloading {sound.name}: {e}")
    
    # Save metadata
    metadata_file = os.path.join(output_dir, f"{query}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump({"data": downloaded}, f, indent=2)
    
    print(f"\nDownloaded {len(downloaded)} sounds")
    print(f"Metadata saved to: {metadata_file}")
    return downloaded


def download_youtube_audio(url, output_path):
    """
    Download audio from YouTube using yt-dlp
    Requires: pip install yt-dlp
    """
    import subprocess
    
    cmd = [
        'yt-dlp',
        '-x',  # Extract audio
        '--audio-format', 'wav',
        '--audio-quality', '0',
        '-o', output_path,
        url
    ]
    
    subprocess.run(cmd, check=True)


def batch_download_youtube(urls_file, output_dir="finetune_dataset"):
    """
    Download multiple YouTube videos from a text file
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    with open(urls_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    downloaded = []
    
    for i, url in enumerate(urls):
        try:
            output_path = os.path.join(output_dir, f"youtube_{i:04d}.wav")
            print(f"Downloading {url}...")
            download_youtube_audio(url, output_path)
            
            # You'll need to add captions manually or extract from video title
            downloaded.append({
                "wav": output_path,
                "caption": f"youtube audio {i}"
            })
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    
    # Save metadata
    with open(os.path.join(output_dir, "youtube_metadata.json"), 'w') as f:
        json.dump({"data": downloaded}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape sounds from internet")
    parser.add_argument("--source", choices=["freesound", "youtube"], required=True)
    parser.add_argument("--api-key", help="Freesound API key (optional if FREESOUND_API_KEY in secret_keys.py)")
    parser.add_argument("--query", help="Search query for Freesound")
    parser.add_argument("--urls-file", help="File with YouTube URLs")
    parser.add_argument("--output-dir", default="finetune_dataset")
    parser.add_argument("--num-sounds", type=int, default=50)
    parser.add_argument("--oauth", action="store_true", help="Use OAuth2 for full file downloads (requires client_id/secret)")
    
    args = parser.parse_args()
    
    if args.source == "freesound":
        # Use provided API key or fall back to imported one
        api_key = args.api_key or FREESOUND_API_KEY
        
        if not api_key:
            print("Error: No API key found. Provide --api-key or add FREESOUND_API_KEY to secret_keys.py")
            exit(1)
        if not args.query:
            print("Error: --query required for Freesound")
            exit(1)
        download_from_freesound(api_key, args.query, args.num_sounds, args.output_dir, args.oauth)
    
    elif args.source == "youtube":
        if not args.urls_file:
            print("Error: --urls-file required for YouTube")
            exit(1)
        batch_download_youtube(args.urls_file, args.output_dir)