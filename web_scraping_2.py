import json
import os
import requests
from bs4 import BeautifulSoup

DATASETS_PATH = r"new_data"
DATASETS_MICROLABS_USA = os.path.join(DATASETS_PATH, "microlabs_usa")

URLS_FILE_PATH = r"drug_urls.json"

def load_drug_urls():
    """Load drug URLs from the saved JSON file."""
    with open(URLS_FILE_PATH, "r", encoding="utf-8") as f:
        drug_urls = json.load(f)
    print(f"Loaded {len(drug_urls)} drug URLs.")
    return drug_urls


def setup_prescribing_info_urls(urls_map):
    """
    Given a dict of product name (e.g. "Celecoxib capsules") and its corresponding URL, load the URL content,
    parse with BeautifulSoup to obtain the child URL for "Prescribing Information". Retrieve the child html content
    and create a soup object. Create a new dict with all these info and return these.
    :param urls_map: dict of product name (e.g. "Celecoxib capsules") and its corresponding URL
    :return: a dict updated_urls that maps product name to its url, child url and child soup where the child is
             the node containing "Prescribing Information"
    """
    updated_urls = {}

    for key, value in urls_map.items():
        print(f"Processing: {key}...")  # Progress statement
        got = False
        updated_urls[key] = {
            "product_url": value,
        }
        data = requests.get(value)
        soup = BeautifulSoup(data.text, "html.parser")
        h2 = soup.findAll("h2")  # we know that "Prescribing Information" is enclosed by <h2> <a .../>

        for h2_item in h2:
            txt = h2_item.get_text()
            if txt is not None:
                if txt.strip().lower() == "Prescribing Information".lower():
                    child_url = h2_item.findAll("a")
                    if child_url:
                        href = child_url[0].get("href")
                        updated_urls[key]["prescribing_info_url"] = href
                        # print(href)
                        html = requests.get(href)
                        prescribing_soup = BeautifulSoup(html.text, "html.parser")
                        # print(prescribing_soup)
                        updated_urls[key]["prescribing_soup"] = prescribing_soup
                        got = True
            if got:  # we got the url and soup for "Prescribing Information" and so we break
                break
        
        if got:
            print(f"Found Prescribing Information for: {key}")  # Progress statement
        else:
            print(f"Prescribing Information not found for: {key}")  # Progress statement

    return updated_urls


def find_elements_with_text(soup):
    # Find all elements that have text content
    elements_with_text = []
    for element in soup.find_all(True):  # True captures all tags
        if element.name not in ["script", "style"]:
            if element.string or element.get_text(strip=True):  # Check for non-empty text
                elements_with_text.append(element)

    # Print the elements and their text content
    print("Found elements with text:")
    for elem in elements_with_text:
        print(f"Tag: {elem.name}, Text: {elem.get_text(strip=True)}")

    return


def get_text_below_anchor_with_special_handling(a_tag):
    result_text = []

    # Loop through all siblings after the <a> tag
    for sibling in a_tag.find_next_siblings():
        if sibling.name == "div":
            childs = sibling.children
            for child in childs:
                if child.name is not None:
                    if child.name.lower() == "table":
                        # print("processing table...")
                        # Process table content
                        table_content = []
                        rows = sibling.find_all('tr')
                        for row in rows:
                            cells = [cell.get_text(strip=False) for cell in row.find_all(['td', 'th'])]
                            # Join cells with a single space separator
                            table_content.append(" ".join(cells))
                        result_text.append("\n".join(table_content))

                    elif child.name.lower() == "img":
                        # Process image content
                        img_src = sibling.get('src', 'No src attribute')
                        img_alt = sibling.get('alt', 'No alt text')
                        result_text.append(f"Image: [src={img_src}, alt={img_alt}]")
                    else:
                        result_text.append(sibling.get_text(strip=True))

    # Join and return the result
    return "\n".join(result_text)


def get_all_sections(soup):
    atags = soup.findAll("a")
    info = dict()

    for atag in atags:
        if atag:
            at = atag.get("id")

            if at and at.startswith("anch_dj_dj-dj"):
                txt = get_text_below_anchor_with_special_handling(atag)
                info[atag.get_text()] = txt

    return info


def process_prescribing_soup(name, soup):
    """
    This takes input as product name and its soup and returns the parsed content for "Prescribing Information"
    :param name: product name
    :param soup: bs4 soup object
    :return: parsed content as dict
    """
    print(f"Processing Prescribing Information for {name}...")  # Progress statement
    results = get_all_sections(soup)
    results["product_name"] = name
    return results


def create_dataset_file(pth, result):
    fname = os.path.join(pth, result["product_name"] + ".json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    print(f"Created file: {fname}")  # Progress statement
    return


if __name__ == '__main__':
    print("Starting process...")  # Progress statement

    # Load drug URLs
    drug_urls = load_drug_urls()
    if not drug_urls:
        print("No drug URLs found. Exiting process.")
        exit()

    # Proceed with the rest of the script
    modified_urls = setup_prescribing_info_urls(drug_urls)
    for k, v in modified_urls.items():
        results = process_prescribing_soup(k, v["prescribing_soup"])
        create_dataset_file(DATASETS_MICROLABS_USA, results)

    print("Process completed.")  # Progress statement

