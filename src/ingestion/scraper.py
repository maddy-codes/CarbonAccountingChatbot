import os
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

TARGET_URLS = [
    # UK Pillar: SECR and Environmental Guidelines (Corrected 2026 Links)
    "https://www.gov.uk/government/publications/environmental-reporting-guidelines-including-mandatory-greenhouse-gas-emissions-reporting-guidance",
    "https://www.gov.uk/government/publications/sustainability-reporting-guidance-2025-26",
    # UK Technical Pillar: Emission Conversion Factors
    "https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting",
    # Global Pillar: GHG Protocol Standards
    "https://ghgprotocol.org/corporate-standard",
    # US Pillar: EPA Emissions Hub (Already working, but here for completeness)
    "https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
    # EU Pillar: ETS Directives
    "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32023L0959",
    "https://www.gov.uk/government/publications/environmental-reporting-guidelines-including-streamlined-energy-and-carbon-reporting-guidelines",
    "https://www.epa.gov/ghgreporting/rulemaking-notices-ghg-reporting",
]

SAVE_DIR = "data/raw"

os.makedirs(SAVE_DIR, exist_ok=True)


def download_pdfs(url):
    print(f"starting scrape for {url}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

    except Exception as e:
        print(f"Failed to reach {url}: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    # finding all anchor tags with href attribues
    links = soup.find_all("a", href=True)

    pdf_count = 0

    for link in links:
        href = link["href"]

        # checking if the link is a pdf
        if href.lower().endswith(".pdf"):
            # convert relative URLs to absolute
            pdf_url = urljoin(url, href)
            file_name = os.path.join(SAVE_DIR, href.split("/")[-1])

            # downloading the file
            print(f"Downloading the file {pdf_url}")
            try:
                pdf_res = requests.get(pdf_url, stream=True)
                with open(file_name, "wb") as f:
                    f.write(pdf_res.content)
                pdf_count += 1
            except Exception as e:
                print(f"Could not download {pdf_url} : {e}")

            # ETHICAL SCRAPPING: Wait 2 secs between the downloads
            time.sleep(2)

    print(f"Finished! Downloaded {pdf_count} PDFs from this page.")


if __name__ == "__main__":
    for target in TARGET_URLS:
        download_pdfs(target)
