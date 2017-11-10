# coding: utf-8
get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('cd', 'new_data')
get_ipython().run_line_magic('ls', '')
import pandas as pd
artists = pd.read_csv('artists.csv', encoding='iso-8859-2')
artists.head()
composer = pd.read_csv('composer.csv', encoding='iso-8859-2')
composer.head()
artists.drop('index', inplace=True)
artists.drop('index', inplace=True, axis=1)
composer.drop('index', inplace=True, axis=1)
join = artists.merge(composer, on='song_id', how='outer')
join.shape
artists.shape
composer.shape
lyricists = pd.read_csv('lyricist.csv', encoding='iso-8859-2').drop('index', inplace=False, axis=1)
join = join.merge(lyricists, on='song_id', how='outer')
join.shape
lyricists.shape
join.head()
join = join.drop_duplicates()
join.shape
pd.to_csv('joined.csv', index=False)
join.to_csv('joined.csv', index=False)
