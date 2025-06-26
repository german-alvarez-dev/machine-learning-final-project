import requests
import pandas as pd
from bs4 import BeautifulSoup
import re



def parse_duration_to_minutes(duration_str: str) -> float:
    """
    Convert duration string like '2:30', '~3:00', or '90 seconds' to float minutes.
    """
    if not duration_str or not isinstance(duration_str, str):
        return None

    duration_str = duration_str.strip().lower()
    duration_str = duration_str.replace("~", "").replace("approx.", "").replace("ca.", "")
    duration_str = duration_str.replace("about", "").strip()

    # mm:ss
    match = re.match(r"(\d+):(\d+)", duration_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return round(minutes + seconds / 60, 2)

    # '90 seconds'
    match = re.match(r"(\d+)\s*seconds?", duration_str)
    if match:
        return round(int(match.group(1)) / 60, 2)

    # '3 minutes'
    match = re.match(r"(\d+)\s*minutes?", duration_str)
    if match:
        return float(match.group(1))

    return None



def get_wikipedia_page_title(query):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json"
    }
    res = requests.get(url, params=params)
    results = res.json().get("query", {}).get("search", [])
    if results:
        return results[0]["title"]  # best match
    return None

from bs4 import BeautifulSoup
import requests
import re

def fetch_wikipedia_ride_metadata(ride_name: str, theme_park: str) -> dict:
    try:
        query = f"{ride_name} {theme_park}"
        print(f"\n🔍 Buscando metadata para: {query}")

        search_title = get_wikipedia_page_title(query)
        if not search_title:
            print("❌ No se encontró el título en Wikipedia")
            return {"ride_type": None, "duration_min": None}

        print(f"🌐 Título encontrado: {search_title}")
        url = f"https://en.wikipedia.org/wiki/{search_title.replace(' ', '_')}"
        print(f"🔗 URL: {url}")

        res = requests.get(url, timeout=10)
        res.raise_for_status()

        soup = BeautifulSoup(res.text, "html.parser")
        infobox = soup.find("table", {"class": "infobox"})

        if not infobox:
            print("⚠️ No se encontró tabla infobox")
            return {"ride_type": None, "duration_min": None}

        ride_type, duration = None, None

        for row in infobox.find_all("tr"):
            header = row.find("th")
            data = row.find("td")

            if header and data:
                header_text = header.get_text(strip=True).lower()
                data_text = data.get_text(strip=True)

                if "type" in header_text and not ride_type:
                    ride_type = data_text
                elif "duration" in header_text and not duration:
                    duration = data_text

        duration_min = parse_duration_to_minutes(duration)
        print(f"⏱️ Duration (min): {duration_min}")

        return {
            "ride_type": ride_type,
            "duration_min": duration_min
        }

    except Exception as e:
        print(f"🛑 Error processing {ride_name}: {e}")
        return {"ride_type": None, "duration_min": None}
