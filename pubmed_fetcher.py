import requests
import xml.etree.ElementTree as ET
import pandas as pd

# Configuration
EMAIL = 'your.email@example.com'
API_KEY = 'your_ncbi_api_key'  # Optional but recommended
BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
SEARCH_URL = BASE_URL + 'esearch.fcgi'
FETCH_URL = BASE_URL + 'efetch.fcgi'

def search_pubmed(query, max_results=100):
    params = {
        'db': 'pubmed',
        'term': query,
        'retmax': max_results,
        'retmode': 'xml',
        'email': EMAIL,
        'api_key': API_KEY
    }
    response = requests.get(SEARCH_URL, params=params)
    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: Unable to fetch data from PubMed.")
    root = ET.fromstring(response.content)
    id_list = [id_elem.text for id_elem in root.findall('.//Id')]
    return id_list

def fetch_details(id_list):
    ids = ','.join(id_list)
    params = {
        'db': 'pubmed',
        'id': ids,
        'retmode': 'xml',
        'email': EMAIL,
        'api_key': API_KEY
    }
    response = requests.get(FETCH_URL, params=params)
    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: Unable to fetch article details.")
    return response.content

def parse_pubmed_xml(xml_data):
    root = ET.fromstring(xml_data)
    articles = []
    for article in root.findall('.//PubmedArticle'):
        article_dict = {}
        # Title
        title = article.find('.//ArticleTitle')
        article_dict['Title'] = title.text if title is not None else None
        # Abstract
        abstract = article.find('.//Abstract/AbstractText')
        article_dict['Abstract'] = abstract.text if abstract is not None else None
        # Authors
        authors = []
        for author in article.findall('.//Author'):
            fore = author.find('ForeName')
            last = author.find('LastName')
            if fore is not None and last is not None:
                authors.append(f"{fore.text} {last.text}")
        article_dict['Authors'] = ', '.join(authors)
        # Publication Date
        pub_date = article.find('.//PubDate')
        if pub_date is not None:
            year = pub_date.find('Year')
            month = pub_date.find('Month')
            day = pub_date.find('Day')
            year_text = year.text if year is not None else '0000'
            month_text = month.text if month is not None else '01'
            day_text = day.text if day is not None else '01'
            article_dict['PublicationDate'] = f"{year_text}-{month_text}-{day_text}"
        else:
            article_dict['PublicationDate'] = None
        # Journal
        journal = article.find('.//Journal/Title')
        article_dict['Journal'] = journal.text if journal is not None else None
        # PMID
        pmid = article.find('.//PMID')
        article_dict['PMID'] = pmid.text if pmid is not None else None
        articles.append(article_dict)
    return articles

def get_pubmed_articles(query, max_results=100):
    pmids = search_pubmed(query, max_results)
    if not pmids:
        return []
    xml_data = fetch_details(pmids)
    articles = parse_pubmed_xml(xml_data)
    return articles

if __name__ == "__main__":
    query = "skin cancer"
    max_results = 50
    articles = get_pubmed_articles(query, max_results)
    df = pd.DataFrame(articles)
    print(df.head())
    df.to_csv('pubmed_skin_cancer_articles.csv', index=False)