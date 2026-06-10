from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException
import os

# Streamlit app URL from environment variable (or default to your app's URL)
STREAMLIT_URL = os.environ.get("STREAMLIT_APP_URL", "https://objectdetectionrealtime.streamlit.app/")

def main():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        driver.get(STREAMLIT_URL)
        print(f"Opened {STREAMLIT_URL}")

        wait = WebDriverWait(driver, 15)
        try:
            # Look for the wake-up button
            button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'Yes, get this app back up')]"))
            )
            print("Wake-up button found. Clicking...")
            button.click()

            # After clicking, check if it disappears
            try:
                wait.until(EC.invisibility_of_element_located((By.XPATH, "//button[contains(text(),'Yes, get this app back up')]")))
                print("Button clicked and disappeared ✅ (app should be waking up)")
            except TimeoutException:
                print("Button was clicked but did NOT disappear ❌ (possible failure)")
                exit(1)

        except TimeoutException:
            # No button at all → app is assumed to be awake
            print("No wake-up button found. Assuming app is already awake ✅")

    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)
    finally:
        driver.quit()
        print("Script finished.")

if __name__ == "__main__":
    main()
