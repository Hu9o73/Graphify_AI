#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Web scraping module for collecting news articles.
This module provides functions to scrape articles from news websites.
"""

import os
import json
import logging
import time
import random
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Initialize logging
logger = logging.getLogger(__name__)

class NewsArticleScraper:
    """Class for scraping news articles from various websites."""
    
    def __init__(self, output_dir='data/raw'):
        """
        Initialize the scraper.
        
        Args:
            output_dir (str): Directory to save scraped articles
        """
        self.output_dir = output_dir
        self._setup_output_dir()
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
    
    def _setup_output_dir(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory set to: {self.output_dir}")
    
    def _get_timestamp(self):
        """Get current timestamp for filenames."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _random_delay(self, min_seconds=1, max_seconds=5):
        """
        Random delay to avoid being blocked.
        
        Args:
            min_seconds (int): Minimum delay in seconds
            max_seconds (int): Maximum delay in seconds
        """
        delay = random.uniform(min_seconds, max_seconds)
        logger.debug(f"Waiting for {delay:.2f} seconds...")
        time.sleep(delay)
    
    def _save_article(self, article, source):
        """
        Save article to JSON file.
        
        Args:
            article (dict): Article data
            source (str): Source website
        """
        timestamp = self._get_timestamp()
        filename = f"{source}_{timestamp}_{article['id']}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(article, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Saved article to {filepath}")
        return filepath
    
    def scrape_reuters(self, num_articles=10, category='world'):
        """
        Scrape articles from Reuters.
        
        Args:
            num_articles (int): Number of articles to scrape
            category (str): News category
            
        Returns:
            list: List of filepaths to saved articles
        """
        logger.info(f"Scraping {num_articles} articles from Reuters/{category}")
        
        # URLs
        base_url = "https://www.reuters.com"
        category_url = f"{base_url}/{category}/"
        
        # Initialize a headless browser
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        saved_files = []
        
        try:
            # Navigate to the category page
            logger.info(f"Navigating to {category_url}")
            driver.get(category_url)

            # Slowing dowwn navigation to imitate human behavior
            time.sleep(2)
            
            # Wait for the page to load
            WebDriverWait(driver, 30).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-testid='MediaStoryCard']"))
            )
            
            # Get article links
            article_elements = driver.find_elements(By.CSS_SELECTOR, "[data-testid='MediaStoryCard']")
            logger.info(f"FOUND ARTICLES ELEMENTS : {len(article_elements)}")
            article_links = []

            time.sleep(2)
            for idx, article in enumerate(article_elements):
                if idx < num_articles:
                    try:
                        logger.info(f"Dealing with article : {idx+1}")
                        time.sleep(5)
                        # Look for the heading element with href attribute
                        link_element = article.find_element(By.CSS_SELECTOR, "[data-testid='Heading']")
                        href = link_element.get_attribute("href")
                        if href and href.startswith(base_url):
                            article_links.append(href)
                            logger.info(f"Article link added: {href}")
                        else:
                            logger.warning(f"Invalid article link: {href}")
                    except Exception as e:
                        logger.warning(f"Error getting article link: {e}")
                else:
                    break
            
            # Remove duplicates and limit to the requested number
            article_links = list(set(article_links))[:num_articles]
            
            logger.info(f"Found {len(article_links)} article links")
            
            # Process each article
            for i, link in enumerate(article_links):
                try:
                    logger.info(f"Processing article {i+1}/{len(article_links)}: {link}")
                    # Navigate to the article page
                    driver.get(link)
                    
                    # Wait for the article content to load
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-testid='DefaultArticleHeader']"))
                    )
                    
                    # Extract article information
                    title_element = driver.find_element(By.CSS_SELECTOR, "[data-testid='Heading']")
                    title = title_element.text
                    logger.info(f"Title found: {title}")


                    # Try different ways to get the publication date
                    try:
                        date_element = driver.find_element(By.CSS_SELECTOR, "time[data-testid='Body']")
                        pub_date = date_element.get_attribute("datetime")
                    except:
                        pub_date = None
                    
                    logger.info(f"Publication date found: {pub_date}")

                    # Get article content
                    try:
                        # Find the article body container
                        article_body = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, ".article-body__container_3ypuX"))
                        )
                        
                        # Get all paragraph elements with data-testid attribute like 'paragraph-1', 'paragraph-2', etc.
                        paragraphs = article_body.find_elements(By.CSS_SELECTOR, "[data-testid^='paragraph-']")
                        
                        # If no paragraphs found, try alternative selectors
                        if not paragraphs:
                            paragraphs = article_body.find_elements(By.CSS_SELECTOR, ".article-body__content_17Yit p")
                        
                        # Extract text from paragraphs
                        content = "\n".join([p.text for p in paragraphs if p.text])
                        
                    except Exception as e:
                        logger.warning(f"Error extracting content: {e}")
                        content = "Content extraction failed"

                    logger.info(f"Content found: {content[:50]}...")

                    # Create article object
                    article = {
                        "id": link.split("/")[-1],
                        "title": title,
                        "url": link,
                        "source": "reuters",
                        "category": category,
                        "published_date": pub_date,
                        "scraped_date": datetime.now().isoformat(),
                        "content": content
                    }
                    
                    # Save article
                    saved_file = self._save_article(article, "reuters")
                    saved_files.append(saved_file)
                    
                    # Random delay to avoid being blocked
                    self._random_delay(10,15)
                
                except Exception as e:
                    logger.error(f"Error processing article {link}: {e}")
            
            logger.info(f"Scraped {len(saved_files)} articles from Reuters")
        
        except Exception as e:
            logger.error(f"Error scraping Reuters: {e}")
        
        finally:
            driver.quit()
        
        return saved_files
    
    def scrape_bbc(self, num_articles=10, category='news'):
        """
        Scrape articles from BBC.
        
        Args:
            num_articles (int): Number of articles to scrape
            category (str): News category
            
        Returns:
            list: List of filepaths to saved articles
        """
        logger.info(f"Scraping {num_articles} articles from BBC/{category}")
        
        # Implement BBC scraping similar to Reuters
        # This is a placeholder - actual implementation would be similar to Reuters method
        
        return []
    
    def scrape_articles(self, source='reuters', num_articles=10, category='world'):
        """
        Scrape articles from the specified source.
        
        Args:
            source (str): Source website ('reuters', 'bbc', etc.)
            num_articles (int): Number of articles to scrape
            category (str): News category
            
        Returns:
            list: List of filepaths to saved articles
        """
        if source.lower() == 'reuters':
            return self.scrape_reuters(num_articles, category)
        elif source.lower() == 'bbc':
            return self.scrape_bbc(num_articles, category)
        else:
            logger.error(f"Unsupported source: {source}")
            return []

def scrape_articles(source='reuters', num_articles=10, category='world', output_dir='data/raw'):
    """
    Scrape articles from the specified source.
    
    Args:
        source (str): Source website ('reuters', 'bbc', etc.)
        num_articles (int): Number of articles to scrape
        category (str): News category
        output_dir (str): Directory to save scraped articles
        
    Returns:
        list: List of filepaths to saved articles
    """
    scraper = NewsArticleScraper(output_dir)
    return scraper.scrape_articles(source, num_articles, category)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    articles = scrape_articles(source='reuters', num_articles=5)
    print(f"Scraped {len(articles)} articles")