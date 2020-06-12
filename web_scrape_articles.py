import newspaper
from newspaper import Article
from newspaper import Source
import pandas as pd

# Let's say we wanted to download articles from Gamespot (which is a web site that discusses video games)
gamespot = newspaper.build("https://www.gamespot.com//news/", memoize_articles = False) 
# I set memoize_articles to False, because I don't want it to cache and save the articles to memory, run after run.
# Fresh run, everytime we run execute this script essentially

final_df = pd.DataFrame()

for each_article in gamespot.articles:
  
  each_article.download()
  each_article.parse()
  each_article.nlp()
  
  temp_df = pd.DataFrame(columns = ['Title', 'Authors', 'Text',
                                    'Summary', 'published_date', 'Source'])
  
  temp_df['Authors'] = each_article.authors
  temp_df['Title'] = each_article.title
  temp_df['Text'] = each_article.text
  temp_df['Summary'] = each_article.summary
  temp_df['published_date'] = each_article.publish_date
  temp_df['Source'] = each_article.source_url
  
  final_df = final_df.append(temp_df, ignore_index = True)
  
# From here you can export this Pandas DataFrame to a csv file
final_df.to_csv('my_scraped_articles.csv')
