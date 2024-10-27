import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
from collections import Counter

def duckduckgo_search(query):
    # DuckDuckGo search URL with added headers to mimic a browser request
    search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}+datasheet+electronics"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(search_url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        print("Error during DuckDuckGo search:", response.status_code)
        return None

def extract_links(html):
    # Extract relevant links from DuckDuckGo search results
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for link in soup.find_all('a', href=True):
        url = link['href']
        if any(keyword in url for keyword in ['datasheet', 'component', 'electronic', 'digikey', 'mouser', 'farnell']):
            links.append(url)
    return links

def select_top_domain(links, top_n=10):
    # Limit links to top 10, parse domains, and count their frequencies
    top_links = links[:top_n]
    domains = [urlparse(link).netloc for link in top_links]
    domain_counts = Counter(domains)
    most_common_domain = domain_counts.most_common(1)
    if most_common_domain:
        selected_domain = most_common_domain[0][0]
        # Return the first link with the most common domain
        for link in top_links:
            if urlparse(link).netloc == selected_domain:
                return link
    return None

def main():
    query = input("Enter a keyword to search for datasheets and electronics info: ")
    search_results_html = duckduckgo_search(query)
    if search_results_html:
        links = extract_links(search_results_html)
        selected_link = select_top_domain(links, top_n=10)
        if selected_link:
            print("Selected Link:", selected_link)
        else:
            print("No relevant link found.")
    time.sleep(1)

if __name__ == "__main__":
    main()
