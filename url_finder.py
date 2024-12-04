from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import json

# Set up Selenium WebDriver
def setup_driver():
    # Ensure you have the correct ChromeDriver installed for your browser
    service = Service("/Users/diyam1/Downloads/chromedriver-mac-arm64/chromedriver")  # Update path to ChromeDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run browser in headless mode (no GUI)
    options.add_argument("--disable-gpu")
    return webdriver.Chrome(service=service, options=options)

def get_drug_urls(base_url):
    """
    Scrape all drug names and their URLs after clicking 'All Products' using Selenium.
    :param base_url: URL of the main products page
    :return: A dictionary of drug names and their URLs
    """
    driver = setup_driver()
    driver.get(base_url)

    # Wait for the 'All Products' button to load and click it
    wait = WebDriverWait(driver, 20)
    all_products_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "show_all_productss")))
    all_products_button.click()
    print("Clicked 'All Products' button.")

    # Wait for the product list to load
    time.sleep(15)  # Adjust delay based on load time

    # Parse the updated page source with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, "html.parser")
    drug_dict = {}

    # Find all drug links in the product listing
    drug_items = soup.select("ul.product_listing li h3 a")
    for drug_item in drug_items:
        drug_name = drug_item.text.strip()
        relative_url = drug_item.get("href")  # Get the relative URL part
        full_url = f"https://www.microlabsusa.com/products/{relative_url}"  # Combine with the base URL
        drug_dict[drug_name] = full_url

    driver.quit()
    return drug_dict

def save_urls_to_file(drug_urls, filepath):
    """Save the drug URLs dictionary to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(drug_urls, f, indent=4)
    print(f"Drug URLs saved to {filepath}")

# def main():
#     BASE_URL = "https://www.microlabsusa.com/our-products/"
    
#     # Fetch drug URLs
#     drug_urls = get_drug_urls(BASE_URL)
    
#     # Format the output as required
#     formatted_output = "URLS = {\n"
#     for name, url in drug_urls.items():
#         formatted_output += f'    "{name}": "{url}",\n'
#     formatted_output += "}\n"
    
#     # Print the formatted output
#     print(formatted_output)

# if __name__ == "__main__":
#     main()
    
def main():
    BASE_URL = "https://www.microlabsusa.com/our-products/"
    
    # Fetch drug URLs
    drug_urls = get_drug_urls(BASE_URL)
    
    # File path to save the data
    file_path = "/Users/diyam1/Desktop/llm/drug_urls.json"
    
    # Save drug URLs to a file
    save_urls_to_file(drug_urls, file_path)

if __name__ == "__main__":
    main()