import newspaper
from newspaper import Article
import nltk
nltk.download('punkt')

import os
import pandas as pd
from datetime import datetime

import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def summarize_text(text, num_sentences=1):
    # Create a parser for the text
    parser = PlaintextParser.from_string(text, Tokenizer('english'))
    
    # Initialize the LexRank summarizer
    summarizer = LexRankSummarizer()
    
    # Summarize the text to the specified number of sentences
    summary = summarizer(parser.document, num_sentences)
    
    # Convert the summary to a single string
    summary_text = ' '.join([str(sentence) for sentence in summary])
    
    return summary_text


def crawl_articles(website, n_articles=None):
    filename = create_filename(website)
    
    articles = newspaper.build(website)
    print(articles.articles)
    excel_info = []
    
    for idx, article_url in enumerate(articles.article_urls()):
        print(idx, article_url)
        
        if isinstance(n_articles, int) and idx > (n_articles - 1):
            save_excel(excel_info, filename)
            break
        
        # parse the article
        parsed_article = Article(article_url)
        parsed_article.download()
        parsed_article.parse()
        
        # extract info and summary
        parsed_article.nlp()
        
        excel_info.append({
            'Title': parsed_article.title,
            'Date': parsed_article.publish_date,
            'Keywords': ', '.join(parsed_article.keywords),
            'Summary': summarize_text(parsed_article.summary, num_sentences=1)
        })

    save_excel(excel_info, filename)
    return excel_info


def create_filename(website):
    website_name = website.split('/')
    website_name = f'{website_name[2]}'
    today_date = str(datetime.today())[:10]
    filename = f'{website_name}_{today_date}.xlsx'
    
    print('Website name:', website_name)
    print('Created file name:', filename)
    return filename
    


def save_excel(data, filename):
    data = pd.DataFrame(data)
    
    if not os.path.exists('data'):
        os.mkdir('data')
    
    data.to_excel(f'data/{filename}', index=False)


if __name__ == '__main__':
    crawl_articles('https://www.cnbc.com/oil/', 10)